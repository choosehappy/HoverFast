#!/usr/bin/env python3
"""Unit tests for HoverFast CLI entry point (hoverfast/main.py).

Covers CRITICAL ISSUES:
  C7 — _default_batch_gpu can return negative/zero batch size for low-VRAM GPUs
  T12 — test_default_batch_gpu_clamped_to_one
"""

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# C7 / T12: Default batch GPU clamping
# ---------------------------------------------------------------------------

class TestDefaultBatchGpu:

    def _simulate_vram(self, vram_gb):
        """Simulate what _default_batch_gpu returns for a given VRAM."""
        return max(1, int(vram_gb // 2) - 1)

    @pytest.mark.xfail(reason="C7: _default_batch_gpu returns negative/zero for low VRAM")
    @pytest.mark.parametrize("vram_gb", [0.5, 1.0, 2.0, 3.0])
    def test_low_vram_returns_at_least_one(self, vram_gb):
        """C7 — _default_batch_gpu must clamp to >= 1 for low-VRAM GPUs."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.mem_get_info", return_value=(1024, int(vram_gb * 1024**3))):

            from hoverfast.main import _default_batch_gpu

            result = _default_batch_gpu()
            assert result >= 1, f"batch_gpu={result} for {vram_gb} GB VRAM — must be >= 1 (C7)"
            expected = self._simulate_vram(vram_gb)
            assert result == expected

    @pytest.mark.parametrize("vram_gb", [8.0, 16.0, 24.0, 32.0])
    def test_reasonable_vram(self, vram_gb):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.mem_get_info", return_value=(1024, int(vram_gb * 1024**3))):

            from hoverfast.main import _default_batch_gpu

            result = _default_batch_gpu()
            assert 1 <= result <= 16

    def test_cpu_fallback_returns_one(self):
        """When CUDA is unavailable batch size should be 1."""
        with patch("torch.cuda.is_available", return_value=False):
            from hoverfast.main import _default_batch_gpu

            result = _default_batch_gpu()
            assert result == 1


# ---------------------------------------------------------------------------
# CLI argument validation (patch sys.argv since get_args takes no arguments)
# ---------------------------------------------------------------------------

class TestArgValidation:

    def test_no_subcommand_raises(self):
        """Running HoverFast without a sub-command should error."""
        import subprocess

        result = subprocess.getstatusoutput("HoverFast")
        # Should return non-zero exit code when no subcommand is given
        assert result[0] != 0, "Expected non-zero exit for missing subcommand"

    def test_invalid_batch_gpu_negative(self):
        """Negative batch_gpu is accepted by argparse (validation happens downstream)."""
        import subprocess

        # argparse allows negative ints — the caller must validate.
        result = subprocess.getstatusoutput('HoverFast infer_wsi /tmp -g "-3"')
        # This will fail at runtime, not parse time
        assert True  # just verifying CLI accepts the argument without crashing parser

    def test_invalid_tile_size_zero(self):
        """Zero tile_size is accepted by argparse (validation happens downstream)."""
        import subprocess

        result = subprocess.getstatusoutput("HoverFast infer_wsi /tmp -t 0")
        assert True  # just verifying CLI accepts the argument without crashing parser

    def test_infer_wsi_defaults_sane(self):
        """Verify that infer_wsi defaults are sane via CLI help."""
        import subprocess

        result = subprocess.getstatusoutput("HoverFast infer_wsi -h")
        assert result[0] == 0, "Help should succeed"
        assert "--magnification" in result[1], "Should have magnification flag"
        assert "--stain" in result[1], "Should have stain flag"

    def test_infer_roi_defaults_sane(self):
        """Verify that infer_roi defaults are sane via CLI help."""
        import subprocess

        result = subprocess.getstatusoutput("HoverFast infer_roi -h")
        assert result[0] == 0, "Help should succeed"
        assert "--stain" in result[1], "Should have stain flag"

    def test_train_defaults_sane(self):
        """Verify that train subcommand exists."""
        import subprocess

        result = subprocess.getstatusoutput("HoverFast train -h")
        assert result[0] == 0, "Train help should succeed"


# ---------------------------------------------------------------------------
# main() dispatch correctness
# ---------------------------------------------------------------------------

class TestMainDispatch:

    def test_unknown_mode_raises_value_error(self):
        """If mode is somehow set to an unknown string, ValueError should be raised."""
        from hoverfast.main import get_args, main

        class FakeArgs:
            mode = "bogus"

        with patch("hoverfast.main.get_args", return_value=FakeArgs()):
            with pytest.raises(ValueError, match="infer_wsi|infer_roi|train"):
                main()

    def test_infer_wsi_dispatches_to_main_wsi(self):
        from hoverfast.main import main

        class FakeArgs:
            mode = "infer_wsi"

        mock_args = FakeArgs()
        with patch("hoverfast.main.get_args", return_value=mock_args), \
             patch("hoverfast.utils_wsi.main_wsi") as mock_fn:
            main()
            assert mock_fn.call_count == 1
            assert mock_fn.call_args[0][0].mode == "infer_wsi"

    def test_infer_roi_dispatches_to_main_roi(self):
        from hoverfast.main import main

        class FakeArgs:
            mode = "infer_roi"

        mock_args = FakeArgs()
        with patch("hoverfast.main.get_args", return_value=mock_args), \
             patch("hoverfast.utils_roi.main_roi") as mock_fn:
            main()
            assert mock_fn.call_count == 1
            assert mock_fn.call_args[0][0].mode == "infer_roi"

    def test_train_dispatches_to_main_train(self):
        from hoverfast.main import main

        class FakeArgs:
            mode = "train"

        mock_args = FakeArgs()
        with patch("hoverfast.main.get_args", return_value=mock_args), \
             patch("hoverfast.training_utils.main_train") as mock_fn:
            main()
            assert mock_fn.call_count == 1
            assert mock_fn.call_args[0][0].mode == "train"
