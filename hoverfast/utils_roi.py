#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import gzip
import math
import multiprocessing
import operator
import os
import time
from collections.abc import Generator
from functools import partial, reduce
from typing import Any, Callable

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
import ujson
from matplotlib.colors import to_rgb
from PIL import Image
from shapely.geometry import Polygon
from shapely.validation import make_valid
from skimage.measure import regionprops
from skimage.segmentation import watershed
from tqdm import tqdm

from .utils_stain_deconv import extract_h_channel_and_stack, hed_to_rgb_torch, rgb_to_hed_torch
from .utils_wsi import load_model, pre_watershed
from .wsi_image_utils import ensure_dirs, setup_logger


def int_coords(x: np.ndarray) -> np.ndarray:
    return np.array(x).round().astype(np.int32)


def divide_batch(items: list[Any], n: int) -> Generator[list[Any], None, None]:
    """
    Split data into batches.

    This helper function divides a list into smaller batches of a specified size.

    Parameters:
    items (list): The list to be divided into batches.
    n (int): The size of each batch.

    Yields:
    list: A batch of the input list with size `n`. The last batch may be smaller if there are fewer than `n` elements remaining.
    """

    for i in range(0, len(items), n):
        yield items[i : i + n]


def load_roi(spaths: list[str]) -> list[np.ndarray]:
    """
    Load regions of interest (ROIs) from given paths.

    This function reads images from the specified paths and converts them to numpy arrays.

    Parameters:
    spaths (list of str): List of file paths to the ROIs.

    Returns:
    list: A list of numpy arrays representing the loaded ROIs.
    """

    out: list[np.ndarray] = []
    for spath in spaths:
        region = np.asarray(Image.open(spath))
        out.append(np.copy(region))
    return out


def multiproc_roi(
    function: Callable[..., Any], arg_list: list[Any], n_process: int, output: bool = True
) -> list[Any] | None:
    """
    Execute a function in parallel using multiprocessing with batching for regions of interest (ROI).

    This function creates a multiprocessing pool with the specified number of processes,
    divides the argument list into batches, and applies the function to each batch in parallel.

    Parameters:
    function (callable): The function to be executed in parallel.
    arg_list (list): A list of arguments to be passed to the function.
    n_process (int): The number of processes to use for multiprocessing.
    output (bool, optional): If True, the output of the function calls is returned. Default is True.

    Returns:
    list: A list of results from the function calls if output is True.
          The results are concatenated if they are lists.
    """

    pool = multiprocessing.Pool(n_process)
    out = list(pool.imap(function, list(divide_batch(arg_list, int(np.ceil(len(arg_list) / n_process))))))
    pool.close()
    pool.join()
    if not output:
        return None
    return reduce(operator.iadd, out, []) if isinstance(out[0], list) else out


def predict_roi_ihc(regions: np.ndarray, model: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform nuclei detection with stain deconvolution on regions using a pre-trained model.

    This function transfers regions to the GPU, performs nuclei detection using the model,
    and processes the output to return binary masks and feature maps.

    Parameters:
    regions (numpy.ndarray): Array of regions to be processed.
    model (torch.nn.Module): Pre-trained model for nuclei detection.
    device (torch.device): Device to perform computation on (GPU or CPU).

    Returns:
    tuple:
        output_cpu (numpy.ndarray): Binary masks indicating detected nuclei.
        maps_final (numpy.ndarray): Feature maps for further processing.
    """

    stride = 168

    # Transfer regions to GPU as a torch tensor
    regions_gpu = torch.from_numpy(regions).half().to(device, memory_format=torch.channels_last)
    regions_shape = regions_gpu.shape
    regions_gpu = regions_gpu / 255
    # Padding operations on GPU
    regions_gpu = torch.nn.functional.pad(
        regions_gpu, (0, 0, stride // 2, stride // 2, stride // 2, stride // 2), mode="reflect"
    )

    # Do color deconvolution
    # Convert to hed
    hed_batch = rgb_to_hed_torch(regions_gpu, device)

    # Extract Hematoxylin channel
    regions_hematoxylin = extract_h_channel_and_stack(hed_batch)

    # Convert HED back to RGB
    reconstructed_rgb_batch = hed_to_rgb_torch(regions_hematoxylin, device)

    # Scale and transpose
    regions_gpu = reconstructed_rgb_batch.permute(0, 3, 1, 2)

    # Model prediction
    output, maps = model(regions_gpu)
    # Post-processing
    output_processed = (
        output[:, :, stride // 2 :, stride // 2 :][:, :, : regions_shape[1], : regions_shape[2]]
        .argmax(axis=1)
        .type(torch.bool)
    )
    maps_processed = maps[:, :, stride // 2 :, stride // 2 :][:, :, : regions_shape[1], : regions_shape[2]]

    # Move the final results to CPU and convert to numpy
    output_cpu = output_processed.detach().cpu().numpy()
    maps_cpu = maps_processed.detach().cpu().numpy()
    maps_final = maps_cpu.astype(np.float32)

    return output_cpu, maps_final


def predict_roi(regions: np.ndarray, model: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform nuclei detection on regions of interest (ROIs) using a pre-trained model.

    This function transfers regions to the GPU, applies padding, performs nuclei detection using the model,
    and processes the output to return binary masks and feature maps.

    Parameters:
    regions (numpy.ndarray): Array of regions to be processed.
    model (torch.nn.Module): Pre-trained model for nuclei detection.
    device (torch.device): Device to perform computation on (GPU or CPU).

    Returns:
    tuple:
        output_cpu (numpy.ndarray): Binary masks indicating detected nuclei.
        maps_final (numpy.ndarray): Feature maps for further processing.
    """

    stride = 168

    # Transfer regions to GPU as a torch tensor
    regions_gpu = torch.from_numpy(regions).half().to(device, memory_format=torch.channels_last)
    regions_shape = regions_gpu.shape
    # Padding operations on GPU
    regions_gpu = torch.nn.functional.pad(
        regions_gpu, (0, 0, stride // 2, stride // 2, stride // 2, stride // 2), mode="reflect"
    )

    # Scale and transpose
    regions_gpu = regions_gpu.permute(0, 3, 1, 2) / 255

    # Model prediction
    output, maps = model(regions_gpu)
    # Post-processing
    output_processed = (
        output[:, :, stride // 2 :, stride // 2 :][:, :, : regions_shape[1], : regions_shape[2]]
        .argmax(axis=1)
        .type(torch.bool)
    )
    maps_processed = maps[:, :, stride // 2 :, stride // 2 :][:, :, : regions_shape[1], : regions_shape[2]]

    # Move the final results to CPU and convert to numpy
    output_cpu = output_processed.detach().cpu().numpy()
    maps_cpu = maps_processed.detach().cpu().numpy()
    maps_final = maps_cpu.astype(np.float32)

    return output_cpu, maps_final


def watershed_object_roi(
    rg: Any,
    dist: np.ndarray,
    submarker: np.ndarray,
    opening: np.ndarray,
    offset: tuple[int, int],
    poly_simplify_tolerance: float,
    threshold: float,
) -> list[np.ndarray]:
    """
    Perform watershed segmentation on detected objects in ROIs.

    This function applies watershed segmentation to divide detected objects and extract their features.
    The features are then processed and saved using simple parameters.

    Parameters:
    rg (skimage.measure._regionprops.RegionProperties): Region properties of the detected object.
    dist (numpy.ndarray): Distance transform for watershed segmentation.
    submarker (numpy.ndarray): Marker image for watershed segmentation.
    opening (numpy.ndarray): Processed mask for watershed segmentation.
    offset (tuple): Offset coordinates for the region.
    poly_simplify_tolerance (float): Tolerance for simplifying polygons.
    threshold (float): Minimum size threshold for nuclei area.

    Returns:
    list: A list of contours representing detected nuclei.
    """

    output: list[np.ndarray] = []

    vals = np.unique(submarker)
    vals = vals[np.nonzero(vals)]
    if vals.size > 1:
        label = watershed(dist, markers=submarker, mask=opening)  # type: ignore[no-untyped-call]
    else:
        label = rg.image.astype(np.uint8)
        vals = [1]
    for val in vals:
        cell = np.uint8(label == val)
        c = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=offset)[0][0]  # type: ignore[call-overload]
        if poly_simplify_tolerance != 0:
            c = cv2.approxPolyDP(c, poly_simplify_tolerance * cv2.arcLength(c, True) / 1000, True)
        if cv2.contourArea(c) <= threshold:
            continue
        poly = Polygon(c.squeeze())
        if not poly.is_valid:
            poly = make_valid(poly)
            while poly.geom_type != "Polygon":
                poly = poly.geoms[np.argmax([p.area for p in poly.geoms])]
            bound = poly.boundary
            if bound.geom_type != "LineString":
                bound = bound.geoms[np.argmax([p.length for p in bound.geoms])]
            c = np.array(bound.coords[:], int).squeeze()
        output.append(c.squeeze())
    return output


def region_feature_roi(
    region: np.ndarray,
    output_mask: np.ndarray,
    dist: np.ndarray,
    marker: np.ndarray,
    opening: np.ndarray,
    poly_simplify_tolerance: float,
    threshold: float,
    color: str,
    width: int,
    outdir: str,
    sname: str,
) -> None:
    """
    Extract features from each detected region of interest (ROI) and save them.

    This function isolates detected objects, performs watershed segmentation,
    and extracts features for each object. The features are then saved to disk.

    Parameters:
    region (numpy.ndarray): Image region being processed.
    output_mask (numpy.ndarray): Binary mask indicating detected nuclei.
    dist (numpy.ndarray): Distance transform for watershed segmentation.
    marker (numpy.ndarray): Marker image for watershed segmentation.
    opening (numpy.ndarray): Processed mask for watershed segmentation.
    poly_simplify_tolerance (float): Tolerance for simplifying polygons.
    threshold (float): Minimum size threshold for nuclei area.
    color (str): Color for drawing contours.
    width (int): Width of the contour lines.
    outdir (str): Output directory.
    sname (str): Slide name.
    """

    rgs = regionprops(ndi.label(output_mask)[0])  # type: ignore[no-untyped-call]
    label: np.ndarray = np.zeros_like(output_mask, np.int32)
    img = np.copy(region)
    color_rgb = (np.array(to_rgb(color)) * 255).astype(int).tolist()
    features: list[dict[str, Any]] = []
    for rg in rgs:
        if rg.area < threshold:
            continue
        ymin, xmin, ymax, xmax = rg.bbox
        temp = watershed_object_roi(
            rg,
            dist[ymin:ymax, xmin:xmax],
            marker[ymin:ymax, xmin:xmax] * rg.image,
            opening[ymin:ymax, xmin:xmax],
            (xmin, ymin),
            poly_simplify_tolerance,
            threshold,
        )
        for i in range(len(temp)):
            poly = temp[i]
            cv2.polylines(img, [poly], -1, color_rgb, width)  # type: ignore[call-overload]
            cv2.fillPoly(label, [poly], len(features) + i + 1)  # type: ignore[call-overload]
        features += [save_poly_dict(poly.astype(float)) for poly in temp]
    # save to Json
    with gzip.open(os.path.join(outdir, "json", sname + ".json.gz"), "wt", encoding="ascii") as zipfile:
        if len(features) != 0:
            ujson.dump(features, zipfile)
    Image.fromarray(img).save(os.path.join(outdir, "overlay", sname + "_overlay.png"))
    Image.fromarray(label).save(os.path.join(outdir, "label_mask", sname + "_label_mask.png"))


def processing_roi(
    regions: np.ndarray,
    names: list[str],
    model: Any,
    device: torch.device,
    batch_to_gpu: int,
    stain: str,
) -> list[tuple[Any, ...]]:
    """
    Process a batch of regions for nuclei detection.

    This function processes a batch of regions for nuclei detection using a pre-trained model.

    Parameters:
    regions (numpy.ndarray): Array of regions to be processed.
    names (list of str): List of names corresponding to the regions.
    model (torch.nn.Module): Pre-trained model for nuclei detection.
    device (torch.device): Device to perform computation on (GPU or CPU).
    batch_to_gpu (int): Target batch size for GPU.

    Returns:
    list: List of tuples containing output masks, feature maps, and region coordinates.
    """
    arg_list1: list[tuple[Any, ...]] = []
    for rgs in tqdm(
        divide_batch(list(regions), batch_to_gpu),
        desc="inner",
        leave=False,
        total=math.ceil(len(regions) / batch_to_gpu),
    ):
        if stain == "ihc_dab":
            output_mask, maps = predict_roi_ihc(rgs, model, device)
        else:
            output_mask, maps = predict_roi(rgs, model, device)

        arg_list1 += [(output_mask[j], maps[j], rgs[j]) for j in range(len(output_mask))]
    return list(map(tuple.__add__, arg_list1, ((x,) for x in names)))


def post_processing_roi(
    data: list[tuple[Any, ...]], outdir: str, threshold: float, poly_simplify_tolerance: float, color: str, width: int
) -> None:
    """
    Post-process the model output to extract nuclei features and save them.

    This function performs watershed segmentation and extracts features from the model output.
    The features are then saved to disk.

    Parameters:
    data (list): List of tuples containing model output masks, feature maps, regions, and names.
    outdir (str): Output directory.
    threshold (float): Minimum size threshold for nuclei area.
    poly_simplify_tolerance (float): Tolerance for simplifying polygons.
    color (str): Color for drawing contours.
    width (int): Width of the contour lines.
    """

    for binmask, maps, region, sname in data:
        dist, marker, opening = pre_watershed(binmask, maps)
        if marker is None:
            continue

        assert dist is not None and opening is not None
        region_feature_roi(
            region, binmask, dist, marker, opening, poly_simplify_tolerance, threshold, color, width, outdir, sname
        )


def save_poly_dict(poly: np.ndarray, object_class: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Serialize a polygon representing a detected object as a dictionary.

    This function converts a polygon and its associated features into a dictionary format.

    Parameters:
    poly (numpy.ndarray): Coordinates of the polygon.
    object_class (dict, optional): Classification information for the object. Default is {'name': 'Nuclei', 'colorRGB': -65536}.
    """

    if object_class is None:
        object_class = {"name": "Nuclei", "colorRGB": -65536}
    feature: dict[str, Any] = {}
    feature["geometry"] = {
        "type": "Polygon",
        "coordinates": (tuple(map(tuple, poly.squeeze())) + (tuple(poly[0].squeeze()),),),
    }
    feature["properties"] = {"object_type": "cell", "classification": object_class, "isLocked": False}
    feature["type"] = "Feature"
    return feature


def infer_roi(
    spaths: list[str],
    n_process: int,
    outdir: str,
    threshold: float,
    poly_simplify_tolerance: float,
    color: str,
    model: Any,
    device: torch.device,
    batch_to_gpu: int,
    width: int,
    stain: str,
) -> None:
    """
    Infer nuclei on regions of interest (ROIs) in parallel and save the results.

    This function performs nuclei detection on the specified regions of interest using a pre-trained model,
    processes the results, and saves them to disk.

    Parameters:
    spaths (list of str): List of file paths to the ROIs.
    n_process (int): The number of processes to use for multiprocessing.
    outdir (str): Output directory.
    threshold (float): Minimum size threshold for nuclei area.
    poly_simplify_tolerance (float): Tolerance for simplifying polygons.
    color (str): Color for drawing contours.
    model (torch.nn.Module): Pre-trained model for nuclei detection.
    device (torch.device): Device to perform computation on (GPU or CPU).
    batch_to_gpu (int): Target batch size for GPU.
    width (int): Width of the contour lines.
    """

    regions = np.array(multiproc_roi(load_roi, spaths, n_process))
    names: list[str] = [os.path.basename(spath).rpartition(".")[0] for spath in spaths]
    arg_list = processing_roi(regions, names, model, device, batch_to_gpu, stain)
    multiproc_roi(
        partial(
            post_processing_roi,
            outdir=outdir,
            threshold=threshold,
            poly_simplify_tolerance=poly_simplify_tolerance,
            color=color,
            width=width,
        ),
        arg_list,
        n_process,
        output=False,
    )


def main_roi(args: argparse.Namespace) -> None:
    """
    Main function to perform nuclei detection on regions of interest (ROIs).

    This function sets up the necessary parameters, loads the model, and performs nuclei detection
    on the specified ROIs, saving the results to disk.

    Parameters:
    args (argparse.Namespace): Parsed command-line arguments.
    """

    # get args

    slide_dirs = args.slide_folder
    outdir = args.outdir
    n_process = args.n_process
    model_path = args.model_path
    poly_simplify_tolerance = args.poly_simplify
    threshold = args.size_threshold
    batch_mem = args.batch_mem
    width = args.width
    batch_to_gpu = args.batch_gpu
    color = args.color
    stain = args.stain

    if n_process is None:
        n_process = os.cpu_count() or 1

    ensure_dirs(outdir, ["label_mask", "overlay", "json"])

    logger = setup_logger(outdir)

    # load model
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model: Any = load_model(model_path, device)

    # get input files
    if len(slide_dirs) == 1:
        # input is a glob pattern
        slide_dirs = glob.glob(slide_dirs[0])

    if not slide_dirs:
        error_message = "No tiles detected in the specified directory."
        logger.error(error_message)
        raise ValueError(error_message)

    start = time.time()
    for spaths in tqdm(
        divide_batch(slide_dirs, batch_mem), desc="outer", leave=False, total=math.ceil(len(slide_dirs) / batch_mem)
    ):
        try:
            infer_roi(
                spaths,
                n_process,
                outdir,
                threshold,
                poly_simplify_tolerance,
                color,
                model,
                device,
                batch_to_gpu,
                width,
                stain,
            )
        except (OSError, RuntimeError):
            logger.exception(f"Batch {slide_dirs} failed")
    end = time.time() - start
    print(f"running time: {end}, #patches {len(slide_dirs)}")
