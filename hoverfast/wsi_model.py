#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from importlib.util import find_spec
from typing import Any

import numpy as np
import openslide
import safetensors.torch
import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from .hoverfast import HoverFast


def _find_tensorrt_libs() -> tuple[str, str]:
    """Dynamically locate TensorRT runtime libraries."""
    spec = find_spec("torch_tensorrt")
    if spec and spec.origin:
        torch_trt_dir = os.path.dirname(spec.origin)  # site-packages/torch_tensorrt
        site_packages = os.path.dirname(torch_trt_dir)
        trt_lib = os.path.join(torch_trt_dir, "lib", "libtorchtrt_runtime.so")
        nvinfer_base = os.path.join(site_packages, "tensorrt_libs", "libnvinfer_plugin.so.11")
    else:
        site_packages = "/opt/conda/lib/python3.11/site-packages"
        trt_lib = os.path.join(site_packages, "torch_tensorrt", "lib", "libtorchtrt_runtime.so")
        nvinfer_base = os.path.join(site_packages, "tensorrt_libs", "libnvinfer_plugin.so.11")
    return nvinfer_base, trt_lib


def load_model(model_path: str, device: torch.device) -> Any:
    """Load the pre-trained model from the given path."""
    if not os.path.exists("unet_trt.ts"):
        print("not compiled - building")
        with safe_open(model_path, framework="pt") as f:
            metadata = f.metadata()
            config = json.loads(metadata["config"])

        model = HoverFast(**config).to(device, memory_format=torch.channels_last)  # type: ignore[call-overload]
        safetensors.torch.load_model(model, model_path)
        model = model.half()
        model.eval()

        import torch_tensorrt

        batch = torch.export.Dim("batch", min=1, max=16)

        example_input = torch.randn(7, 3, 1024, 1024, device="cuda", dtype=torch.float16)
        dynamic_shapes = {"x": {0: batch}}

        exp_program = torch.export.export(
            model,
            (example_input,),
            dynamic_shapes=dynamic_shapes,
        )

        trt_model = torch_tensorrt.dynamo.compile(
            exp_program,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, 3, 1024, 1024),
                    opt_shape=(7, 3, 1024, 1024),
                    max_shape=(16, 3, 1024, 1024),
                    dtype=torch.half,
                )
            ],
            enabled_precisions={torch.half},
            optimization_level=5,
            workspace_size=8 << 30,
            use_python_runtime=False,
        )

        torch_tensorrt.save(trt_model, "unet_trt.ts", inputs=[example_input], output_format="torchscript")

        del example_input
        torch.cuda.empty_cache()
    else:
        print("loading compiled..")
        import ctypes

        nvinfer_path, trt_runtime_path = _find_tensorrt_libs()
        ctypes.CDLL(nvinfer_path, mode=ctypes.RTLD_GLOBAL)
        torch.ops.load_library(trt_runtime_path)  # type: ignore[no-untyped-call]
        trt_model = torch.jit.load("unet_trt.ts")  # type: ignore[no-untyped-call]

    print("returning model")
    return trt_model


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

        from .wsi_image_utils import rgba2rgb

        img = rgba2rgb(region)
        img_np = np.array(img)
        tensor = (torch.from_numpy(img_np).to(torch.float16) / 255.0).permute(2, 0, 1)

        return tensor, torch.tensor([int(coord[0]), int(coord[1])], dtype=torch.int64)


def predict_ihc_batch(regions_gpu: torch.Tensor, model: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform nuclei detection with stain deconvolution on a batch of regions."""
    from .utils_stain_deconv import extract_h_channel_and_stack, hed_to_rgb_torch, rgb_to_hed_torch

    hed_batch = rgb_to_hed_torch(regions_gpu, device)
    regions_hematoxylin = extract_h_channel_and_stack(hed_batch)
    reconstructed_rgb_batch = hed_to_rgb_torch(regions_hematoxylin, device)
    regions_gpu = reconstructed_rgb_batch.permute(0, 3, 1, 2)

    output, maps = model(regions_gpu)
    output_processed = output.argmax(axis=1).type(torch.bool)

    # Quantize maps to float8 for faster GPU-to-CPU transfer.
    if device.type == "cuda":
        maps_out = maps.to(torch.float8_e4m3fn)
    else:
        maps_out = maps

    return output_processed, maps_out


def predict_batch(regions_gpu: torch.Tensor, model: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform nuclei detection on a batch of regions."""
    output, maps = model(regions_gpu)
    output_processed = output.argmax(axis=1).type(torch.bool)

    # Quantize maps to float8 for faster GPU-to-CPU transfer (halves bandwidth from ~20MB to ~10MB per batch).
    # Post-processing converts back to float32; mean absolute error on roundtrip is ~0.018, negligible for watershed.
    if regions_gpu.device.type == "cuda":
        maps_out = maps.to(torch.float8_e4m3fn)
    else:
        maps_out = maps

    return output_processed, maps_out
