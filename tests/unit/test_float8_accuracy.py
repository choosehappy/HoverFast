#!/usr/bin/env python3
"""Unit tests for float8 quantization round-trip accuracy.

Covers CRITICAL ISSUES:
  T11 — predict_batch float8 roundtrip accuracy

The codebase uses torch.float8_e4m3fn to reduce GPU→CPU transfer bandwidth.
This test verifies that the quantization error is within acceptable bounds
for watershed post-processing (MAE < 0.02).
"""

import pytest
import torch


class TestFloat8RoundTripAccuracy:

    def test_float8_mae_within_tolerance(self):
        """T11 — float8 round-trip MAE should be < 0.02 for typical map values."""
        # Simulate the quantization path used in predict_batch / predict_ihc_batch
        maps = torch.rand(4, 2, 256, 256, dtype=torch.float32)

        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("torch.float8_e4m3fn not available in this PyTorch version")

        maps_f8 = maps.to(torch.float8_e4m3fn)
        maps_roundtrip = maps_f8.float()

        mae = (maps - maps_roundtrip).abs().mean().item()
        assert mae < 0.02, f"float8 round-trip MAE={mae:.4f} exceeds threshold of 0.02"

    def test_float8_preserves_sign(self):
        """float8 should not flip the sign of map values."""
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("torch.float8_e4m3fn not available")

        maps = torch.rand(2, 2, 128, 128, dtype=torch.float32) * 2 - 1  # range [-1, 1]
        maps_f8 = maps.to(torch.float8_e4m3fn)
        maps_roundtrip = maps_f8.float()

        # Sign flips should be rare (< 0.1% of values)
        sign_flip_ratio = ((maps * maps_roundtrip) < 0).float().mean().item()
        assert sign_flip_ratio < 0.001, f"Sign flip ratio too high: {sign_flip_ratio}"

    def test_float8_zero_preservation(self):
        """Near-zero values should round-trip to near-zero."""
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("torch.float8_e4m3fn not available")

        maps = torch.zeros(2, 2, 64, 64, dtype=torch.float32)
        maps_f8 = maps.to(torch.float8_e4m3fn)
        maps_roundtrip = maps_f8.float()

        assert maps_roundtrip.abs().max() < 0.01

    def test_float8_max_value_preservation(self):
        """Peak values should be within acceptable range after round-trip."""
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("torch.float8_e4m3fn not available")

        maps = torch.ones(2, 2, 64, 64, dtype=torch.float32)
        maps_f8 = maps.to(torch.float8_e4m3fn)
        maps_roundtrip = maps_f8.float()

        # float8 can represent values up to ~448; 1.0 should round-trip well
        assert (maps_roundtrip - 1.0).abs().max() < 0.1


# ---------------------------------------------------------------------------
# predict_batch / predict_ihc_batch CPU path sanity
# ---------------------------------------------------------------------------

class TestPredictBatchCpuPath:

    def _make_dummy_model(self):
        """Return a mock model that returns predictable NCHW outputs."""
        from unittest.mock import MagicMock

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
# load_model missing file (T2 / C2)
# ---------------------------------------------------------------------------

class TestLoadModelMissingFile:

    @pytest.mark.xfail(reason="C2: load_model silently falls back to torch.jit.load on missing file")
    def test_missing_safetensors_raises_file_not_found(self):
        """load_model() must raise FileNotFoundError when the .safetensors file is absent."""
        from hoverfast.wsi_model import load_model

        with pytest.raises(FileNotFoundError, match="not found|does not exist"):
            load_model("/nonexistent/path/to/model.safetensors", torch.device("cpu"))


# ---------------------------------------------------------------------------
# _find_tensorrt_libs (I4)
# ---------------------------------------------------------------------------

class TestFindTensorRtLibs:

    def test_returns_paths_when_torch_tensorrt_missing(self):
        """_find_tensorrt_libs should not crash when torch_tensorrt is absent."""
        from hoverfast.wsi_model import _find_tensorrt_libs
        from unittest.mock import MagicMock, patch

        with patch("hoverfast.wsi_model.find_spec", return_value=None):
            nvinfer_path, trt_lib = _find_tensorrt_libs()
            assert isinstance(nvinfer_path, str)
            assert isinstance(trt_lib, str)

    def test_returns_paths_when_torch_tensorrt_present(self):
        from hoverfast.wsi_model import _find_tensorrt_libs
        from unittest.mock import MagicMock, patch

        fake_spec = MagicMock(origin="/some/path/torch_tensorrt/__init__.py")
        with patch("hoverfast.wsi_model.find_spec", return_value=fake_spec):
            nvinfer_path, trt_lib = _find_tensorrt_libs()
            assert isinstance(nvinfer_path, str)
            assert isinstance(trt_lib, str)
