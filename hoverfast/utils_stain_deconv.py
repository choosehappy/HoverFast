import torch
import numpy as np
from scipy import linalg

#Colorspace conversion matrices
rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                         [0.07, 0.99, 0.11],
                         [0.27, 0.57, 0.78]])
hed_from_rgb = linalg.inv(rgb_from_hed)


def extract_h_channel_and_stack(hed_batch_image_tensor):
    """
    Extract the H channel from HED images and stack with zero arrays for the other channels.
    
    Parameters:
    hed_batch_image_tensor (torch.Tensor): Batch of images in HED color space.
    device (torch.device): Device to perform computation on (GPU or CPU).

    Returns:
    torch.Tensor: Batch of images with only the H channel and zero arrays for the other channels.
    """
    h_channel = hed_batch_image_tensor[:, :, :, 0]
    zero_arr = torch.zeros_like(h_channel, dtype=torch.float16, device=h_channel.device)
    regions_filter = torch.stack((h_channel, zero_arr, zero_arr), dim=-1)
    
    return regions_filter


def rgb_to_hed_torch(batch_image_tensor, device):
    """
    Perform RGB to HED transformation using PyTorch for batches.
    
    Parameters:
    batch_image_tensor (torch.tensor): Batch of images in RGB color space.
    device (torch.device): Device to perform computation on (GPU or CPU).

    Returns:
    torch.Tensor: Batch of images in HED color space.
    """
    
    log_adjust = torch.log(torch.tensor(1e-6, dtype=torch.float16, device=device))
    hed_from_rgb_tensor = torch.tensor(hed_from_rgb, dtype=torch.float16, device=device)

    batch_image_tensor_max = torch.maximum(batch_image_tensor, torch.tensor(1e-6, dtype=torch.float16, device=device))
    log_batch_image_tensor = torch.log(batch_image_tensor_max) / log_adjust
    hed_batch_image_tensor = log_batch_image_tensor @ hed_from_rgb_tensor

    hed_batch_image_tensor = torch.maximum(hed_batch_image_tensor, torch.tensor(0, dtype=torch.float16, device=device))
    return hed_batch_image_tensor


def hed_to_rgb_torch(hed_batch_image_tensor, device):
    """
    Perform HED to RGB transformation using PyTorch for batches.
    
    Parameters:
    hed_batch_image_tensor (torch.Tensor): Batch of images in HED color space.
    device (torch.device): Device to perform computation on (GPU or CPU).

    Returns:
    torch.Tensor: Batch of images in RGB color space.
    """
    
    rgb_from_hed_tensor = torch.tensor(rgb_from_hed, dtype=torch.float16, device=device)

    log_adjust = torch.log(torch.tensor(1e-6, dtype=torch.float16, device=device))
    rgb_batch_image_tensor = torch.exp(hed_batch_image_tensor @ rgb_from_hed_tensor * log_adjust)
    rgb_batch_image_tensor = torch.clamp(rgb_batch_image_tensor, 0, 1)

    return rgb_batch_image_tensor
