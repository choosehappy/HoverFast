#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import math
import os
import time
from typing import Any, Callable

import numpy as np
import skimage.morphology as ndi
import tables
import torch
import torch.nn.functional as F
from skimage.measure import regionprops
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryConfusionMatrix
from tqdm import tqdm

from .augment import randaugment
from .hoverfast import HoverFast


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        fname: str,
        device: torch.device,
        transforms: Callable | None = None,
        edge_weight: bool = False,
    ) -> None:
        """
        Initialize the Dataset object.

        Parameters:
        fname (str): Filename of the HDF5 database.
        device (torch.device): Device to perform computation on (GPU or CPU).
        transforms (callable, optional): Transformations to be applied to the images and labels. Default is None.
        edge_weight (bool, optional): Whether to compute edge weights. Default is False.
        """

        self.fname = fname
        self.edge_weight = edge_weight
        self.device = device
        self.transforms = transforms

        with tables.open_file(self.fname) as db:
            self.numpixels = db.root.numpixels[:]
            self.nitems = db.root.img.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.

        Parameters:
        index (int): Index of the item to retrieve.

        Returns:
        tuple: Tuple containing the image, mask, maps, edge weights, and boundary weights.
        """
        with tables.open_file(self.fname, "r") as db:
            img = db.root.img[index]
            label = db.root.label[index]

        if self.transforms:
            transforms = self.transforms()
            augmented = transforms(image=img, mask=label)
            img = augmented["image"]
            label = augmented["mask"]

        mask = label != 0
        if self.edge_weight:
            eweight = ndi.binary_dilation(mask == 1, ndi.square(5)) & ~mask  # type: ignore[attr-defined]
        else:  # otherwise the edge weight is all ones and thus has no affect
            eweight = np.ones(mask.shape, dtype=mask.dtype)

        maps, bweight = make_maps(label)

        if img.dtype == "uint8":
            img = img / 255

        return (
            torch.from_numpy(img).permute(2, 0, 1),
            torch.from_numpy(mask),
            torch.from_numpy(maps),
            torch.from_numpy(eweight),
            torch.from_numpy(bweight),
        )

    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        Returns:
        int: Number of items in the dataset.
        """
        return self.nitems  # type: ignore[no-any-return]


def asMinutes(s: float) -> str:
    """
    Convert seconds to a string representing minutes and seconds.

    Parameters:
    s (float): Time in seconds.

    Returns:
    str: Time formatted as 'Xm Ys'.
    """
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s}s"


def timeSince(since: float, percent: float) -> str:
    """
    Calculate the elapsed time and estimated remaining time.

    Parameters:
    since (float): Start time in seconds.
    percent (float): Progress percentage (0 to 1).

    Returns:
    str: Elapsed and remaining time formatted as 'Elapsed (- Remaining)'.
    """
    now = time.time()
    s = now - since
    es = s / (percent + 0.00001)
    rs = es - s
    return f"{asMinutes(s)} (- {asMinutes(rs)})"


def make_maps(label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Create feature maps and boundary weight for a given label.

    This function generates horizontal and vertical gradient maps, and computes boundary weights for each region in the label.

    Parameters:
    label (numpy.ndarray): Label image with segmented regions.

    Returns:
    tuple: Tuple containing the maps and the boundary weight.
    """
    maps = np.zeros((2,) + label.shape, np.float32)
    weight = np.ones(label.shape)
    rgs = regionprops(label)  # type: ignore[no-untyped-call]
    for rg in rgs:
        ymin, xmin, ymax, xmax = rg.bbox
        shape = rg.image.shape
        if (ymin == 0) | (xmin == 0) | (ymax == label.shape[0]) | (xmax == label.shape[1]):
            weight[ymin:ymax, xmin:xmax] = 0
        else:
            maps[0, ymin:ymax, xmin:xmax] += rg.image * np.linspace(-1, 1, shape[1])
            maps[1, ymin:ymax, xmin:xmax] += rg.image * np.linspace(-1, 1, shape[0]).reshape((shape[0], 1))
    return maps, weight


def dice_loss(pred: torch.Tensor, true: torch.Tensor, smooth: float = 1e-3) -> torch.Tensor:
    """
    Compute the Dice loss between predicted and true labels.

    Parameters:
    pred (torch.Tensor): Predicted labels, assumed to be of shape NxHxWxC.
    true (torch.Tensor): True labels, assumed to be of shape NxHxWxC.
    smooth (float, optional): Smoothing factor to avoid division by zero. Default is 1e-3.

    Returns:
    torch.Tensor: Computed Dice loss.
    """
    inse = torch.sum(pred * true, (0, 1, 2))
    pred_sum = torch.sum(pred, (0, 1, 2))
    true_sum = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (pred_sum + true_sum + smooth)
    loss = torch.sum(loss)
    return loss


def grad_kernel(size: int = 11) -> torch.Tensor:
    """
    Create a gradient kernel for computing image gradients.

    This function generates a kernel for computing horizontal and vertical gradients
    in an image. The kernel size is adjustable.

    Parameters:
    size (int, optional): Size of the kernel. Default is 11.

    Returns:
    torch.Tensor: A 4D tensor representing the gradient kernel.
    """

    temp = torch.arange(
        -size // 2 + 1,
        size // 2 + 1,
        dtype=torch.float32,
        device="cuda",
        requires_grad=False,
    )
    temp = torch.outer(temp, torch.ones_like(temp))
    stemp = temp * temp
    return (temp / (stemp + stemp.T + 1.0e-15)).view(1, 1, size, size)


class Criterion(nn.Module):
    def __init__(
        self,
        class_weight: torch.Tensor,
        edge_weight: float,
        grad_weight: float,
        hv_weight: float,
        dice_weight: float,
        crossentropy_weight: float,
    ) -> None:
        """
        Initialize the Criterion object for computing loss.

        Parameters:
        class_weight (torch.Tensor): Weights for each class for cross-entropy loss.
        edge_weight (float): Weight for edge loss.
        grad_weight (float): Weight for gradient loss.
        hv_weight (float): Weight for horizontal and vertical map loss.
        dice_weight (float): Weight for dice loss.
        crossentropy_weight (float): Weight for cross-entropy loss.
        """

        super().__init__()
        self.critCEntropy = nn.CrossEntropyLoss(weight=class_weight, ignore_index=-100, reduction="none")
        self.critGrad = nn.MSELoss(reduction="none")
        self.crithv = nn.MSELoss(reduction="none")
        self.edge_weight = edge_weight
        self.class_weight = class_weight
        self.grad_weight = grad_weight
        self.hv_weight = hv_weight
        self.dice_weight = dice_weight
        self.crossentropy_weight = crossentropy_weight
        self.kernel = grad_kernel()

    def forward(
        self,
        x_pred: torch.Tensor,
        hvm_pred: torch.Tensor,
        y: torch.Tensor,
        hvmaps: torch.Tensor,
        y_weight: torch.Tensor,
        hv_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the loss components for model training.

        Parameters:
        x_pred (torch.Tensor): Predicted class scores from the model.
        hvm_pred (torch.Tensor): Predicted horizontal and vertical maps from the model.
        y (torch.Tensor): True class labels.
        hvmaps (torch.Tensor): True horizontal and vertical maps.
        y_weight (torch.Tensor): Weights for the true class labels.
        hv_weight (torch.Tensor): Weights for the horizontal and vertical maps.

        Returns:
        tuple: Tuple containing the horizontal and vertical map loss, gradient loss,
               cross-entropy loss, and dice loss.
        """
        # Cross-entropy loss (Lc)
        loss_matrix = self.critCEntropy(x_pred, y)
        lossCEntropy = (loss_matrix * (self.edge_weight**y_weight)).mean()  # can skip if edge weight==1

        # dice loss (Ld)
        lossD = dice_loss(x_pred.argmax(1), y)

        # HV map loss (La)
        lossHV = self.crithv(hvm_pred.squeeze(), hvmaps)
        weight = self.class_weight[y] * hv_weight
        lossHV[:, 0] *= weight
        lossHV[:, 1] *= weight
        lossHV.permute(0, 2, 3, 1)[(y == 0) & (x_pred.argmax(1) == 1)] = 0
        lossHV = lossHV.mean()

        # gradient loss (Lb)
        grad = torch.cat(
            (
                F.conv2d(hvmaps[:, 0].unsqueeze(1), self.kernel.permute(0, 1, 3, 2), padding="same"),
                F.conv2d(hvmaps[:, 1].unsqueeze(1), self.kernel, padding="same"),
            ),
            dim=1,
        )
        grad_pred = torch.cat(
            (
                F.conv2d(hvm_pred[:, 0].unsqueeze(1), self.kernel.permute(0, 1, 3, 2), padding="same"),
                F.conv2d(hvm_pred[:, 1].unsqueeze(1), self.kernel, padding="same"),
            ),
            dim=1,
        )

        lossGrad = self.critGrad(grad_pred, grad)
        lossGrad[:, 0] *= weight
        lossGrad[:, 1] *= weight
        lossGrad.permute(0, 2, 3, 1)[(y == 0) & (x_pred.argmax(1) == 1)] = 0
        lossGrad = lossGrad.mean()

        return (
            self.hv_weight * lossHV,
            self.grad_weight * lossGrad,
            self.crossentropy_weight * lossCEntropy,
            self.dice_weight * lossD,
        )


def main_train(args: argparse.Namespace) -> None:
    """
    Main function to train the HoverFast model.

    This function sets up the training parameters, initializes the model, loads the dataset,
    and performs training and validation over the specified number of epochs.

    Parameters:
    args (argparse.Namespace): Parsed command-line arguments.
    """
    datapath: str = args.dataset_path
    dataname: str = args.dataname
    outdir: str = args.outdir
    batch_size: int = args.batch_size
    n_process: int = min(batch_size, os.cpu_count() or 1)
    num_epochs: int = args.epoch
    depth: int = args.depth  # Depth of the network
    wf: int = args.width  # wf (int): number of filters in the first layer is 2**wf

    # UNet params
    n_classes: int = 2  # Number of classes in the data mask to predict
    in_channels: int = 3  # Input channels of the data, RGB = 3
    padding: bool = True  # Whether to use padding
    batch_norm: bool = True  # Whether to use batch normalization between layers
    up_mode: str = "upconv"  # 'upconv' for transpose convolution or 'upsample' for interpolation
    conv_block: str = "msunet"  # Convolutional block type, either 'unet' or 'msunet'

    # Training parameters
    edge_weight: float = 1.1  # Boost edge values based on original UNet paper
    phases: list[str] = ["train", "test"]  # Phases for training and testing
    validation_phases: list[str] = ["test"]  # Phases for validation
    grad_weight: float = 1 / 10
    hv_weight: float = 14
    dice_weight: float = 1 / 6
    crossentropy_weight: float = 1

    torch.backends.cudnn.benchmark = True
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = HoverFast(
        n_classes=n_classes,
        in_channels=in_channels,
        padding=padding,
        depth=depth,
        wf=wf,
        up_mode=up_mode,
        batch_norm=batch_norm,
        conv_block=conv_block,
    ).to(device, memory_format=torch.channels_last)  # type: ignore[call-overload]

    # Load dataset and DataLoader
    dataset: dict[str, Dataset] = {}
    dataLoader: dict[str, DataLoader] = {}
    for phase in phases:
        dataset[phase] = Dataset(
            os.path.join(datapath, dataname) + f"_{phase}.pytable",
            device,
            transforms=randaugment,
            edge_weight=bool(edge_weight),
        )
        dataLoader[phase] = DataLoader(
            dataset[phase], batch_size=batch_size, shuffle=True, num_workers=n_process, pin_memory=True, drop_last=True
        )

    optim: torch.optim.Optimizer = torch.optim.Adam(model.parameters())
    class_weight: np.ndarray = dataset["train"].numpixels[1, :]
    f: np.ndarray = np.sum(class_weight) / class_weight
    class_weight = f / np.sum(f)
    class_weight_torch: torch.Tensor = torch.from_numpy(class_weight).type("torch.FloatTensor").to(device)

    print(f"class weight: {class_weight_torch}")  # Display class weights

    criterion: Criterion = Criterion(
        class_weight_torch, edge_weight, grad_weight, hv_weight, dice_weight, crossentropy_weight
    )
    bcm: BinaryConfusionMatrix = BinaryConfusionMatrix().to(device)

    writer: SummaryWriter = SummaryWriter(
        os.path.join(
            outdir, f"hoverfast_{dataname}_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%Hh%M")
        )
    )  # Open the tensorboard visualiser

    best_loss_on_test: float = np.Infinity
    torch.tensor(edge_weight).to(device)
    start_time: float = time.time()
    for epoch in range(num_epochs):
        for phase in phases:
            stats: dict[str, Any] = {}
            stats["loss"] = {}
            for stat in ["total_loss", "hv_loss", "grad_loss", "crossEntropy_loss", "dice_loss"]:
                stats["loss"][stat] = 0
            stats["cmatrix"] = torch.zeros((n_classes, n_classes)).to(device)

            if phase == "train":
                model.train()
            else:
                model.eval()

            for _, (X, y, hvmaps, y_weight, b_weight) in enumerate(tqdm(dataLoader[phase], leave=False)):
                X = X.type("torch.FloatTensor").to(device, memory_format=torch.channels_last)
                y = y.type("torch.LongTensor").to(device)
                hvmaps = hvmaps.to(device)
                y_weight = y_weight.to(device)
                b_weight = b_weight.to(device)
                with torch.set_grad_enabled(phase == "train"):
                    x_pred, hvm_pred = model(X)

                    losses: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = criterion(
                        x_pred, hvm_pred, y, hvmaps, y_weight, b_weight
                    )
                    loss: torch.Tensor = sum(losses)  # type: ignore[assignment]

                    if phase == "train":
                        optim.zero_grad()
                        loss.backward()  # type: ignore[no-untyped-call]
                        optim.step()
                        train_loss: float = loss.item()

                    stats["loss"]["total_loss"] += loss.detach()
                    stats["loss"]["hv_loss"] += losses[0].detach()
                    stats["loss"]["grad_loss"] += losses[1].detach()
                    stats["loss"]["crossEntropy_loss"] += losses[2].detach()
                    stats["loss"]["dice_loss"] += losses[3].detach()

                    if phase in validation_phases:
                        predflat = x_pred.argmax(axis=1).flatten()
                        targetflat = y.flatten()

                        stats["cmatrix"] += bcm(predflat, targetflat).detach()

            n_batches: int = len(dataLoader[phase])
            print(n_batches)
            stats["loss"]["total_loss"] = (stats["loss"]["total_loss"] / n_batches).cpu().numpy()
            stats["loss"]["hv_loss"] = (stats["loss"]["hv_loss"] / n_batches).cpu().numpy()
            stats["loss"]["grad_loss"] = (stats["loss"]["grad_loss"] / n_batches).cpu().numpy()
            stats["loss"]["crossEntropy_loss"] = (stats["loss"]["crossEntropy_loss"] / n_batches).cpu().numpy()
            stats["loss"]["dice_loss"] = (stats["loss"]["dice_loss"] / n_batches).cpu().numpy()

            if phase in validation_phases:
                cm_sum = stats["cmatrix"].sum()
                stats["cmatrix"] = (stats["cmatrix"] / cm_sum if cm_sum > 0 else stats["cmatrix"]).cpu().numpy()

            # Save metrics to tensorboard
            writer.add_scalars(f"{phase}/loss", stats["loss"], epoch)
            if phase in validation_phases:
                cm = stats["cmatrix"]
                writer.add_scalar(f"{phase}/accuracy", cm.trace(), epoch)
                col1_sum = cm[:, 1].sum()
                row1_sum = cm[1].sum()
                row0_sum = cm[0].sum()
                col0_sum = cm[:, 0].sum()
                writer.add_scalar(f"{phase}/precision", cm[1, 1] / col1_sum if col1_sum > 0 else 0.0, epoch)
                writer.add_scalar(f"{phase}/recall", cm[1, 1] / row1_sum if row1_sum > 0 else 0.0, epoch)
                writer.add_scalar(f"{phase}/specificity", cm[0, 0] / row0_sum if row0_sum > 0 else 0.0, epoch)
                writer.add_scalar(
                    f"{phase}/negative predictive value", cm[0, 0] / col0_sum if col0_sum > 0 else 0.0, epoch
                )

            if phase == "train":
                train_loss = stats["loss"]["total_loss"]
            current_loss: float = stats["loss"]["total_loss"]

        print(
            f"{timeSince(start_time, (epoch + 1) / num_epochs)} ([{epoch + 1}/{num_epochs}] {(epoch + 1) / num_epochs * 100:.0f}%), train loss: {train_loss:.4f} test loss: {current_loss:.4f}",
            end="",
        )

        if current_loss < best_loss_on_test:
            best_loss_on_test = current_loss
            print("  **")
            state: dict[str, Any] = {
                "epoch": epoch + 1,
                "model_dict": model.state_dict(),
                "optim_dict": optim.state_dict(),
                "best_loss_on_test": current_loss,
                "n_classes": n_classes,
                "in_channels": in_channels,
                "padding": padding,
                "depth": depth,
                "wf": wf,
                "up_mode": up_mode,
                "batch_norm": batch_norm,
                "conv_block": conv_block,
            }

            torch.save(state, f"{outdir}/{dataname}_best_model.pth")
        else:
            print()
