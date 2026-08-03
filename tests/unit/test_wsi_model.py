#!/usr/bin/env python3
"""Unit tests for WSI model loading (hoverfast/wsi_model.py).

Covers CRITICAL ISSUES:
  C2 — load_model should fail gracefully when file doesn't exist
  T2  — test_load_model_missing_file
  I4  — hardcoded site-packages path
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# T2 / C2: Missing model file handling
# ---------------------------------------------------------------------------

class TestLoadModelMissingFile:

    @pytest.mark.xfail(reason="C2: load_model silently falls back to torch.jit.load on missing file")
    def test_missing_safetensors_raises_file_not_found(self):
        """load_model() must raise FileNotFoundError when the .safetensors file is absent."""
        from hoverfast.wsi_model import load_model

        with pytest.raises(FileNotFoundError, match="not found|does not exist"):
            load_model("/nonexistent/path/to/model.safetensors", torch.device("cpu"))


# ---------------------------------------------------------------------------
# WSIPatchDataset basic sanity
# ---------------------------------------------------------------------------

class TestWSIPatchDataset:

    def test_len_matches_coords(self):
        from hoverfast.wsi_model import WSIPatchDataset

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
        from hoverfast.wsi_model import WSIPatchDataset

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
        from hoverfast.wsi_model import predict_batch

        # NCHW format: (batch, channels, height, width)
        regions_gpu = torch.rand(2, 3, 256, 256) / 255.0
        model = self._make_dummy_model()

        output_mask, maps_out = predict_batch(regions_gpu, model)

        assert isinstance(output_mask, torch.Tensor)
        assert isinstance(maps_out, torch.Tensor)
        assert output_mask.shape[0] == 2

    @pytest.mark.xfail(reason="Stain deconv dtype mismatch (float vs Half) — pre-existing bug")
    def test_predict_ihc_batch_cpu_no_crash(self):
        """predict_ihc_batch should work on CPU without CUDA-specific code paths."""
        from hoverfast.wsi_model import predict_ihc_batch

        # NHWC format: (batch, height, width, channels) — what stain deconv expects
        regions_gpu = torch.rand(2, 256, 256, 3) / 255.0
        model = self._make_dummy_model()
        device = torch.device("cpu")

        output_mask, maps_out = predict_ihc_batch(regions_gpu, model, device)

        assert isinstance(output_mask, torch.Tensor)
        assert isinstance(maps_out, torch.Tensor)


# ---------------------------------------------------------------------------
# I4: Hardcoded site-packages path in _find_tensorrt_libs
# ---------------------------------------------------------------------------

class TestFindTensorRtLibs:

    def test_returns_paths_when_torch_tensorrt_missing(self):
        """_find_tensorrt_libs should not crash when torch_tensorrt is absent."""
        from hoverfast.wsi_model import _find_tensorrt_libs

        with patch("hoverfast.wsi_model.find_spec", return_value=None):
            nvinfer_path, trt_lib = _find_tensorrt_libs()
            assert isinstance(nvinfer_path, str)
            assert isinstance(trt_lib, str)
            assert "libnvinfer" in nvinfer_path or "nvinfer" in os.path.basename(
                nvinfer_path
            ).lower()

    def test_returns_paths_when_torch_tensorrt_present(self):
        from hoverfast.wsi_model import _find_tensorrt_libs

        fake_spec = MagicMock(origin="/some/path/torch_tensorrt/__init__.py")
        with patch("hoverfast.wsi_model.find_spec", return_value=fake_spec):
            nvinfer_path, trt_lib = _find_tensorrt_libs()
            assert isinstance(nvinfer_path, str)
            assert isinstance(trt_lib, str)
