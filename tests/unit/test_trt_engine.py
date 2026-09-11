#!/usr/bin/env python3
"""Unit tests for TensorRT engine lifecycle (hoverfast/models/trt_engine.py).

Covers the graceful-degradation contract: inference must never crash when the
compiled engine is missing or was built for a different machine; instead it
falls back to eager PyTorch and prints a build hint.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from hoverfast.models import trt_engine


class TestTensorrtAvailable:
    def test_false_when_torch_tensorrt_missing(self):
        with patch("hoverfast.models.trt_engine.find_spec", return_value=None):
            assert trt_engine.tensorrt_available() is False

    def test_false_when_import_fails(self):
        with (
            patch("hoverfast.models.trt_engine.find_spec", return_value=MagicMock()),
            patch.dict("sys.modules", {"torch_tensorrt": None, "tensorrt": None}),
        ):
            # find_spec is mocked to return a spec, but importing raises.
            assert trt_engine.tensorrt_available() is False


class TestEnginePathEnv:
    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("HOVERFAST_TRT_ENGINE", "/tmp/custom.ts")
        assert trt_engine._engine_path_from_env("unet_trt.ts") == "/tmp/custom.ts"

    def test_default_used_without_env(self, monkeypatch):
        monkeypatch.delenv("HOVERFAST_TRT_ENGINE", raising=False)
        assert trt_engine._engine_path_from_env("unet_trt.ts") == "unet_trt.ts"


class TestResolveModelFallback:
    def _dummy_model(self):
        return MagicMock(spec=torch.nn.Module)

    def test_missing_engine_falls_back_to_eager(self, tmp_path, capsys):
        engine = str(tmp_path / "missing.ts")
        dummy = self._dummy_model()
        with patch("hoverfast.models.trt_engine.load_eager_model", return_value=dummy) as mock_eager:
            model = trt_engine.resolve_model("model.safetensors", torch.device("cpu"), engine_path=engine)
        assert model is dummy
        mock_eager.assert_called_once()
        out = capsys.readouterr().out
        assert "eager PyTorch" in out
        assert "HoverFast build" in out

    def test_incompatible_engine_falls_back_to_eager(self, tmp_path, capsys):
        engine = tmp_path / "unet_trt.ts"
        engine.write_bytes(b"not a real engine")
        dummy = self._dummy_model()
        with (
            patch("hoverfast.models.trt_engine.load_engine", side_effect=RuntimeError("wrong arch")),
            patch("hoverfast.models.trt_engine.load_eager_model", return_value=dummy) as mock_eager,
        ):
            model = trt_engine.resolve_model("model.safetensors", torch.device("cpu"), engine_path=str(engine))
        assert model is dummy
        mock_eager.assert_called_once()
        out = capsys.readouterr().out
        assert "could not be loaded" in out
        assert "HoverFast build" in out

    def test_missing_engine_strict_raises(self, tmp_path):
        engine = str(tmp_path / "missing.ts")
        with pytest.raises(RuntimeError, match="No TensorRT engine"):
            trt_engine.resolve_model(
                "model.safetensors", torch.device("cpu"), engine_path=engine, allow_eager_fallback=False
            )

    def test_incompatible_engine_strict_raises(self, tmp_path):
        engine = tmp_path / "unet_trt.ts"
        engine.write_bytes(b"not a real engine")
        with (
            patch("hoverfast.models.trt_engine.load_engine", side_effect=OSError("missing lib")),
            pytest.raises(RuntimeError, match="could not be loaded"),
        ):
            trt_engine.resolve_model(
                "model.safetensors", torch.device("cpu"), engine_path=str(engine), allow_eager_fallback=False
            )

    def test_valid_engine_is_returned(self, tmp_path):
        engine = tmp_path / "unet_trt.ts"
        engine.write_bytes(b"engine")
        dummy = self._dummy_model()
        with patch("hoverfast.models.trt_engine.load_engine", return_value=dummy):
            model = trt_engine.resolve_model("model.safetensors", torch.device("cpu"), engine_path=str(engine))
        assert model is dummy


class TestBuildEngine:
    def test_build_without_tensorrt_raises(self):
        with (
            patch("hoverfast.models.trt_engine.tensorrt_available", return_value=False),
            pytest.raises(RuntimeError, match="TensorRT is not available"),
        ):
            trt_engine.build_engine("model.safetensors", engine_path="/tmp/x.ts")
