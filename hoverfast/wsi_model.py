#!/usr/bin/env python3

import json
import os

import numpy as np
import openslide
import safetensors.torch
import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from .hoverfast import HoverFast


def load_model(model_path, device):
    """Load the pre-trained model from the given path."""
    if not os.path.exists("unet_trt.ts"):
        print("not compiled - building")
        with safe_open(model_path, framework="pt") as f:
            metadata = f.metadata()
            config = json.loads(metadata["config"])

        model = HoverFast(**config).to(device, memory_format=torch.channels_last)
        safetensors.torch.load_model(model, model_path)
        model = model.half()
        model.eval()

        import torch_tensorrt
        batch = torch.export.Dim("batch", min=1, max=7)

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
                    max_shape=(7, 3, 1024, 1024),
                    dtype=torch.half,
                )
            ],
            enabled_precisions={torch.half},
            optimization_level=5,
            workspace_size=8 << 30,
            use_python_runtime=False,
        )

        torch_tensorrt.save(trt_model, "unet_trt.ts", inputs=[example_input],
                           output_format="torchscript")
    else:
        print("loading compiled..")
        import ctypes
        ctypes.CDLL(
            "/opt/conda/lib/python3.11/site-packages/tensorrt_libs/libnvinfer_plugin.so.11",
            mode=ctypes.RTLD_GLOBAL,
        )
        torch.ops.load_library(
            "/opt/conda/lib/python3.11/site-packages/torch_tensorrt/lib/libtorchtrt_runtime.so"
        )
        trt_model = torch.jit.load("unet_trt.ts")

    print("returning model")
    return trt_model


class WSIPatchDataset(Dataset):
    """PyTorch Dataset for lazy loading of WSI patches."""

    def __init__(self, coords, slide_data):
        self.coords = coords
        self.slide_data = slide_data
        self.slide = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self.slide is None:
            self.slide = openslide.OpenSlide(
                os.path.join(self.slide_data['fpath'],
                             self.slide_data['sname'] + f".{self.slide_data['format']}")
            )

        coord = self.coords[idx]
        region = self.slide.read_region(
            (self.slide_data['xb'] + coord[0], self.slide_data['yb'] + coord[1]),
            self.slide_data['level'],
            (int(self.slide_data['region_size'] * (self.slide_data['downfactor'] / self.slide_data['working_d'])),) * 2
        )

        if self.slide_data['working_d'] != self.slide_data['downfactor']:
            region = region.resize((self.slide_data['region_size'],) * 2)

        from .wsi_image_utils import rgba2rgb
        img = rgba2rgb(region)
        img_np = np.array(img)
        tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        return tensor, coord


def predict_ihc_batch(regions_gpu, model, device):
    """Perform nuclei detection with stain deconvolution on a batch of regions."""
    from .utils_stain_deconv import extract_h_channel_and_stack, hed_to_rgb_torch, rgb_to_hed_torch

    hed_batch = rgb_to_hed_torch(regions_gpu, device)
    regions_hematoxylin = extract_h_channel_and_stack(hed_batch)
    reconstructed_rgb_batch = hed_to_rgb_torch(regions_hematoxylin, device)
    regions_gpu = reconstructed_rgb_batch.permute(0, 3, 1, 2)

    output, maps = model(regions_gpu)
    output_processed = output.argmax(axis=1).type(torch.bool)

    return output_processed, maps


def predict_batch(regions_gpu, model):
    """Perform nuclei detection on a batch of regions."""
    output, maps = model(regions_gpu)
    output_processed = output.argmax(axis=1).type(torch.bool)
    maps_fp8 = maps.to(torch.float8_e4m3fn)

    return output_processed, maps_fp8
