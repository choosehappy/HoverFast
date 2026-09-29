#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch
from scipy import linalg

# Colorspace conversion matrices
_rgb_from_hed: np.ndarray = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
_hed_from_rgb: np.ndarray = linalg.inv(_rgb_from_hed)

_device_cache: tuple[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None


def _get_cached_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _device_cache
    if _device_cache is not None and _device_cache[0] == device:
        return _device_cache[1]

    hed_from_rgb_t = torch.tensor(_hed_from_rgb, dtype=torch.float16, device=device)
    rgb_from_hed_t = torch.tensor(_rgb_from_hed, dtype=torch.float16, device=device)
    log_adjust = torch.log(torch.tensor(1e-6, dtype=torch.float16, device=device))
    _device_cache = (device, (hed_from_rgb_t, rgb_from_hed_t, log_adjust))
    return hed_from_rgb_t, rgb_from_hed_t, log_adjust


def extract_h_channel_and_stack(hed_batch_image_tensor: torch.Tensor) -> torch.Tensor:
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


def rgb_to_hed_torch(batch_image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Perform RGB to HED transformation using PyTorch for batches.

    Parameters:
    batch_image_tensor (torch.tensor): Batch of images in RGB color space.
    device (torch.device): Device to perform computation on (GPU or CPU).

    Returns:
    torch.Tensor: Batch of images in HED color space.
    """
    hed_from_rgb_t, _, log_adjust = _get_cached_tensors(device)
    eps = torch.tensor(1e-6, dtype=torch.float16, device=device)
    zero = torch.tensor(0, dtype=torch.float16, device=device)

    batch_image_tensor_max = torch.maximum(batch_image_tensor, eps)
    log_batch_image_tensor = torch.log(batch_image_tensor_max) / log_adjust
    hed_batch_image_tensor = log_batch_image_tensor @ hed_from_rgb_t

    return torch.maximum(hed_batch_image_tensor, zero)


def hed_to_rgb_torch(hed_batch_image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Perform HED to RGB transformation using PyTorch for batches.

    Parameters:
    hed_batch_image_tensor (torch.Tensor): Batch of images in HED color space.
    device (torch.device): Device to perform computation on (GPU or CPU).

    Returns:
    torch.Tensor: Batch of images in RGB color space.
    """
    _, rgb_from_hed_t, log_adjust = _get_cached_tensors(device)

    rgb_batch_image_tensor = torch.exp(hed_batch_image_tensor @ rgb_from_hed_t * log_adjust)
    return torch.clamp(rgb_batch_image_tensor, 0, 1)
