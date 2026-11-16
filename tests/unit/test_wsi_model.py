#!/usr/bin/env python3
"""Unit tests for WSI model loading (hoverfast/models/wsi_model.py).

Covers CRITICAL ISSUES:
  C2 — load_model should fail gracefully when file doesn't exist
  T2  — test_load_model_missing_file
  I4  — hardcoded site-packages path
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# T2 / C2: Missing model file handling
# ---------------------------------------------------------------------------

class TestLoadModelMissingFile:

    def test_missing_safetensors_raises_file_not_found(self):
        """load_model() must raise FileNotFoundError when the .safetensors file is absent."""
        from hoverfast.models.wsi_model import load_model

        with pytest.raises(FileNotFoundError, match="not found|does not exist"):
            load_model("/nonexistent/path/to/model.safetensors", torch.device("cpu"))


# ---------------------------------------------------------------------------
# WSIPatchDataset basic sanity
# ---------------------------------------------------------------------------

class TestWSIPatchDataset:

    def test_len_matches_coords(self):
        from hoverfast.models.wsi_model import WSIPatchDataset

        coords = [[0, 0], [100, 100], [200, 200]]
        slide_data = {
            "fpath": "/tmp",
            "sname": "test",
            "format": "svs",
            "xb": 0,
            "yb": 0,
            "level": 0,
            "region_size": 256,
            "downfactor": 1.0,
            "working_d": 1.0,
        }

        ds = WSIPatchDataset(coords, slide_data)
        assert len(ds) == 3

    def test_len_empty_coords(self):
        from hoverfast.models.wsi_model import WSIPatchDataset

        ds = WSIPatchDataset([], {"fpath": "/tmp", "sname": "x", "format": "svs"})
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# predict_batch / predict_ihc_batch CPU path sanity
# ---------------------------------------------------------------------------

class TestPredictBatchCpuPath:

    def _make_dummy_model(self):
        """Return a mock model that returns predictable NCHW outputs."""
        mock = MagicMock()

        def fake_call(x):
            # x is NCHW format: (batch, channels, height, width)
            b, _, h, w = x.shape
            output = torch.zeros(b, 2, h, w)
            maps = torch.rand(b, 2, h, w) * 0.5
            return output, maps

        # model(regions_gpu) calls __call__, not .forward() directly
        mock.side_effect = fake_call
        return mock

    def test_predict_batch_cpu_no_crash(self):
        """predict_batch should work on CPU without CUDA-specific code paths."""
        from hoverfast.models.wsi_model import predict_batch

        # NCHW format: (batch, channels, height, width)
        regions_gpu = torch.rand(2, 3, 256, 256) / 255.0
        model = self._make_dummy_model()

        output_mask, maps_out = predict_batch(regions_gpu, model)

        assert isinstance(output_mask, torch.Tensor)
        assert isinstance(maps_out, torch.Tensor)
        assert output_mask.shape[0] == 2

    def test_predict_ihc_batch_cpu_no_crash(self):
        """predict_ihc_batch should work on CPU without CUDA-specific code paths."""
        from hoverfast.models.wsi_model import predict_ihc_batch

        # NCHW in [0, 1], as WSIPatchDataset yields and the pipeline passes it
        regions_gpu = torch.rand(2, 3, 256, 256)
        model = self._make_dummy_model()
        device = torch.device("cpu")

        output_mask, maps_out = predict_ihc_batch(regions_gpu, model, device)

        assert isinstance(output_mask, torch.Tensor)
        assert isinstance(maps_out, torch.Tensor)
        assert output_mask.shape == (2, 256, 256)

    def test_predict_ihc_batch_matches_v1_preprocessing(self):
        """The model must see the same hematoxylin-only image as HoverFast 1.x produced.

        1.x took (N, H, W, 3) uint8 regions, divided by 255, ran the stain deconvolution on the
        last axis and permuted to (N, 3, H, W). The pipeline now passes (N, 3, H, W) in [0, 1].
        """
        from hoverfast.common.stain_deconv import extract_h_channel_and_stack, hed_to_rgb_torch, rgb_to_hed_torch
        from hoverfast.models.wsi_model import predict_ihc_batch

        device = torch.device("cpu")
        rng = np.random.default_rng(0)
        regions_u8 = rng.integers(0, 256, size=(2, 64, 48, 3), dtype=np.uint8)  # H != W catches axis mix-ups

        # 1.x reference
        ref = torch.from_numpy(regions_u8).half() / 255
        ref = hed_to_rgb_torch(extract_h_channel_and_stack(rgb_to_hed_torch(ref, device)), device).permute(0, 3, 1, 2)

        seen = {}

        def capture(x):
            seen["x"] = x
            b, _, h, w = x.shape
            return torch.zeros(b, 2, h, w), torch.zeros(b, 2, h, w)

        model = MagicMock(side_effect=capture)
        nchw = (torch.from_numpy(regions_u8).to(torch.float16) / 255.0).permute(0, 3, 1, 2)
        output_mask, _ = predict_ihc_batch(nchw, model, device)

        assert seen["x"].shape == (2, 3, 64, 48)
        assert torch.equal(seen["x"], ref)
        assert output_mask.shape == (2, 64, 48)


# ---------------------------------------------------------------------------
# I4: Hardcoded site-packages path in _find_tensorrt_libs
# ---------------------------------------------------------------------------

class TestFindTensorRtLibs:

    def test_returns_paths_when_torch_tensorrt_missing(self):
        """_find_tensorrt_libs should not crash when torch_tensorrt is absent."""
        from hoverfast.models.wsi_model import _find_tensorrt_libs

        with patch("hoverfast.models.wsi_model.find_spec", return_value=None):
            nvinfer_path, trt_lib = _find_tensorrt_libs()
            assert isinstance(nvinfer_path, str)
            assert isinstance(trt_lib, str)
            assert "libnvinfer" in nvinfer_path or "nvinfer" in os.path.basename(
                nvinfer_path
            ).lower()

    def test_returns_paths_when_torch_tensorrt_present(self):
        from hoverfast.models.wsi_model import _find_tensorrt_libs

        fake_spec = MagicMock(origin="/some/path/torch_tensorrt/__init__.py")
        with patch("hoverfast.models.wsi_model.find_spec", return_value=fake_spec):
            nvinfer_path, trt_lib = _find_tensorrt_libs()
            assert isinstance(nvinfer_path, str)
            assert isinstance(trt_lib, str)

    @pytest.mark.parametrize("major", ["10", "11"])
    def test_nvinfer_plugin_major_is_discovered(self, tmp_path, major):
        """The plugin soname follows the installed TensorRT major version, not a hard-coded one."""
        from hoverfast.models.wsi_model import _find_nvinfer_plugin

        (tmp_path / f"libnvinfer_plugin.so.{major}").touch()
        (tmp_path / f"libnvinfer_plugin.so.{major}.1.0").touch()  # full-version file is ignored
        assert _find_nvinfer_plugin(str(tmp_path)) == str(tmp_path / f"libnvinfer_plugin.so.{major}")

    def test_nvinfer_plugin_highest_major_wins(self, tmp_path):
        from hoverfast.models.wsi_model import _find_nvinfer_plugin

        (tmp_path / "libnvinfer_plugin.so.10").touch()
        (tmp_path / "libnvinfer_plugin.so.11").touch()
        assert _find_nvinfer_plugin(str(tmp_path)).endswith("libnvinfer_plugin.so.11")

    def test_libraries_exist_when_tensorrt_installed(self):
        """With TensorRT installed, the paths handed to ctypes/torch must exist.

        Guards against the runtime falling back to eager PyTorch because the
        library lookup and the installed TensorRT version disagree.
        """
        from hoverfast.models.trt_engine import tensorrt_available
        from hoverfast.models.wsi_model import _find_tensorrt_libs

        if not tensorrt_available():
            pytest.skip("TensorRT not installed")
        nvinfer_path, trt_lib = _find_tensorrt_libs()
        assert os.path.isfile(nvinfer_path), nvinfer_path
        assert os.path.isfile(trt_lib), trt_lib
