#!/usr/bin/env python3

from __future__ import annotations

import os
from importlib.util import find_spec
from typing import Any

import numpy as np
import openslide
import torch
from torch.utils.data import Dataset


def _find_tensorrt_libs() -> tuple[str, str]:
    """Dynamically locate TensorRT runtime libraries.

    Kept here (rather than in :mod:`hoverfast.models.trt_engine`) for backwards
    compatibility: existing code and tests import it from this module.
    """
    spec = find_spec("torch_tensorrt")
    if spec and spec.origin:
        torch_trt_dir = os.path.dirname(spec.origin)  # site-packages/torch_tensorrt
        site_packages = os.path.dirname(torch_trt_dir)
        trt_lib = os.path.join(torch_trt_dir, "lib", "libtorchtrt_runtime.so")
        nvinfer_base = os.path.join(site_packages, "tensorrt_libs", "libnvinfer_plugin.so.11")
    else:
        # C3 fix: Use sys.prefix instead of hardcoded /opt/conda path
        import sys

        site_packages = os.path.join(
            sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"
        )
        trt_lib = os.path.join(site_packages, "torch_tensorrt", "lib", "libtorchtrt_runtime.so")
        nvinfer_base = os.path.join(site_packages, "tensorrt_libs", "libnvinfer_plugin.so.11")
    return nvinfer_base, trt_lib


def load_model(
    model_path: str,
    device: torch.device,
    engine_path: str | None = None,
    allow_eager_fallback: bool = True,
) -> Any:
    """Load a runnable HoverFast model.

    Prefers a compiled TensorRT engine and transparently falls back to eager
    PyTorch (with an explicit build hint) when the engine is missing or was
    compiled for a different machine. See :func:`hoverfast.models.trt_engine.resolve_model`.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    from .trt_engine import resolve_model

    return resolve_model(
        model_path,
        device,
        engine_path=engine_path,
        allow_eager_fallback=allow_eager_fallback,
    )


class WSIPatchDataset(Dataset):
    """PyTorch Dataset for lazy loading of WSI patches."""

    coords: np.ndarray | list[list[int]]
    slide_data: dict[str, Any]
    slide: openslide.OpenSlide | None

    def __init__(self, coords: np.ndarray | list[list[int]], slide_data: dict[str, Any]) -> None:
        self.coords = coords
        self.slide_data = slide_data
        self.slide = None

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.slide is None:
            self.slide = openslide.OpenSlide(
                os.path.join(self.slide_data["fpath"], self.slide_data["sname"] + f".{self.slide_data['format']}")
            )

        coord = self.coords[idx]
        region = self.slide.read_region(
            (self.slide_data["xb"] + coord[0], self.slide_data["yb"] + coord[1]),
            self.slide_data["level"],
            (int(self.slide_data["region_size"] * (self.slide_data["downfactor"] / self.slide_data["working_d"])),) * 2,
        )

        if self.slide_data["working_d"] != self.slide_data["downfactor"]:
            region = region.resize((self.slide_data["region_size"],) * 2)

        from ..wsi.image_utils import rgba2rgb

        img = rgba2rgb(region)
        img_np = np.array(img)
        tensor = (torch.from_numpy(img_np).to(torch.float16) / 255.0).permute(2, 0, 1)

        return tensor, torch.tensor([int(coord[0]), int(coord[1])], dtype=torch.int64)


def predict_ihc_batch(regions_gpu: torch.Tensor, model: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform nuclei detection with stain deconvolution on a batch of regions."""
    from ..common.stain_deconv import extract_h_channel_and_stack, hed_to_rgb_torch, rgb_to_hed_torch

    # C6 fix: Ensure float16 dtype for stain deconv (cached tensors are float16)
    if regions_gpu.dtype != torch.float16:
        regions_gpu = regions_gpu.half()

    hed_batch = rgb_to_hed_torch(regions_gpu, device)
    regions_hematoxylin = extract_h_channel_and_stack(hed_batch)
    reconstructed_rgb_batch = hed_to_rgb_torch(regions_hematoxylin, device)
    regions_gpu = reconstructed_rgb_batch.permute(0, 3, 1, 2)

    output, maps = model(regions_gpu)
    output_processed = output.argmax(axis=1).type(torch.bool)

    # Quantize maps to float8 for faster GPU-to-CPU transfer.
    maps_out = maps.to(torch.float8_e4m3fn) if device.type == "cuda" else maps

    return output_processed, maps_out


def predict_batch(regions_gpu: torch.Tensor, model: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform nuclei detection on a batch of regions."""
    output, maps = model(regions_gpu)
    output_processed = output.argmax(axis=1).type(torch.bool)

    # Quantize maps to float8 for faster GPU-to-CPU transfer (halves bandwidth from ~20MB to ~10MB per batch).
    # Post-processing converts back to float32; mean absolute error on roundtrip is ~0.018, negligible for watershed.
    maps_out = maps.to(torch.float8_e4m3fn) if regions_gpu.device.type == "cuda" else maps

    return output_processed, maps_out
