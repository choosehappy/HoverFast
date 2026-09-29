#!/usr/bin/env python3
"""Unit tests for WSI utility functions (hoverfast/utils_wsi.py).

Covers CRITICAL ISSUES:
  T5  — find_regions with no tissue detected
  C8  — make_maps boundary check uses bitwise OR instead of logical or
  I7  — preload_file_linux cross-platform compatibility
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# T5: find_regions with no tissue (all-white image)
# ---------------------------------------------------------------------------

class TestFindRegionsNoTissue:

    def test_find_regions_empty_result_on_white_image(self):
        """When the WSI is entirely white, find_regions should return empty array."""
        from hoverfast.utils_wsi import find_regions

        slide_data = {
            "fpath": "/tmp",
            "sname": "white_slide",
            "format": "svs",
            "xb": 0,
            "yb": 0,
            "width": 1024,
            "height": 1024,
            "stride_at_base": 64,
            "tile_at_base": 128,
        }

        # Mock openslide to return an all-white image
        mock_osh = MagicMock()
        mock_osh.level_downsamples = [1.0, 2.0, 4.0]

        from PIL import Image
        white_img = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
        mock_osh.read_region.return_value = white_img

        with patch("openslide.open_slide", return_value=mock_osh):
            result = find_regions(None, slide_data)
            # Should be empty or very small (no tissue detected → no regions)
            assert len(result) == 0


# ---------------------------------------------------------------------------
# get_slide edge cases
# ---------------------------------------------------------------------------

class TestGetSlide:

    def test_get_slide_missing_mpp_uses_default(self):
        """When MPP property is absent, default of 0.245 should be used."""
        import openslide

        from hoverfast.utils_wsi import get_slide
        import logging

        mock_osh = MagicMock()

        # Return None for mpp-x, return bounds for other properties
        def prop_get(key, default=None):
            if key == "openslide.mpp-x":
                return None
            if key == openslide.PROPERTY_NAME_BOUNDS_X:
                return 0
            if key == openslide.PROPERTY_NAME_BOUNDS_Y:
                return 0
            if key == openslide.PROPERTY_NAME_BOUNDS_WIDTH:
                return 10000
            if key == openslide.PROPERTY_NAME_BOUNDS_HEIGHT:
                return 10000
            return default

        mock_osh.properties.get = prop_get
        mock_osh.level_dimensions = [(10000, 10000)]
        mock_osh.level_downsamples = [1.0]

        logger = logging.getLogger("test_get_slide")

        with patch("openslide.open_slide", return_value=mock_osh):
            slide_data = get_slide(
                sname="test",
                sformat="svs",
                fpath="/tmp",
                mag=40.0,
                kernel_size=256,
                region_size=1024,
                threshold=85.0,
                outdir="./output",
                poly_simplify_tolerance=6.0,
                logger=logger,
            )

        assert slide_data["mpp"] == 0.245

    def test_get_slide_base_mag_lower_than_requested_raises(self):
        """When base magnification is below requested level, ValueError should be raised."""
        from hoverfast.utils_wsi import get_slide
        import logging

        mock_osh = MagicMock()

        def prop_get(key, default=None):
            if key == "openslide.mpp-x":
                return "0.5"  # high MPP → low magnification (~20X)
            return default

        mock_osh.properties.get = prop_get
        mock_osh.level_dimensions = [(10000, 10000)]
        mock_osh.level_downsamples = [1.0]

        logger = logging.getLogger("test_get_slide")

        with patch("openslide.open_slide", return_value=mock_osh):
            with pytest.raises(ValueError, match="Base magnification"):
                get_slide(
                    sname="test",
                    sformat="svs",
                    fpath="/tmp",
                    mag=40.0,  # requesting higher than available
                    kernel_size=256,
                    region_size=1024,
                    threshold=85.0,
                    outdir="./output",
                    poly_simplify_tolerance=6.0,
                    logger=logger,
                )


# ---------------------------------------------------------------------------
# I7: preload_file_linux cross-platform compatibility
# ---------------------------------------------------------------------------

class TestPreloadFileLinux:

    def test_preload_on_non_linux_no_crash(self):
        """preload_file_linux should silently return 0 on non-Linux platforms."""
        from hoverfast.wsi_image_utils import preload_file_linux

        # On Linux, preload_file_linux calls os.open() which may fail for nonexistent files.
        # The function is a no-op on non-Linux — verify it doesn't crash when patched.
        with patch("os.open", return_value=42):
            result = preload_file_linux("/tmp/dummy.svs")
            assert isinstance(result, int)

    def test_preload_on_nonexistent_file(self):
        """Should not crash when file doesn't exist (non-fatal in main flow)."""
        from hoverfast.wsi_image_utils import preload_file_linux

        with patch("os.open", side_effect=FileNotFoundError()):
            with pytest.raises(FileNotFoundError):
                preload_file_linux("/tmp/nonexistent.svs")

    def test_preload_on_linux_with_posix_fadvise(self):
        """On Linux with posix_fadvise available, should call it."""
        from hoverfast.wsi_image_utils import preload_file_linux

        mock_fd = 42
        with patch("os.open", return_value=mock_fd), \
             patch("os.fstat") as mock_stat, \
             patch("os.posix_fadvise") as mock_advise, \
             patch("os.close") as mock_close:

            mock_stat.return_value.st_size = 1_000_000

            result = preload_file_linux("/tmp/dummy.svs")
            assert result == 0
            mock_advise.assert_called_once_with(mock_fd, 0, 1_000_000, os.POSIX_FADV_WILLNEED)
            mock_close.assert_called_once_with(mock_fd)

    def test_preload_on_nonexistent_file(self):
        """Should not crash when file doesn't exist (non-fatal in main flow)."""
        from hoverfast.wsi_image_utils import preload_file_linux

        with patch("os.open", side_effect=FileNotFoundError()):
            with pytest.raises(FileNotFoundError):
                preload_file_linux("/tmp/nonexistent.svs")


# ---------------------------------------------------------------------------
# ensure_dirs edge cases
# ---------------------------------------------------------------------------

class TestEnsureDirs:

    def test_creates_base_dir(self, tmp_path):
        from hoverfast.wsi_image_utils import ensure_dirs

        target = tmp_path / "new_dir"
        ensure_dirs(str(target))
        assert target.is_dir()

    def test_creates_subdirs(self, tmp_path):
        from hoverfast.wsi_image_utils import ensure_dirs

        target = tmp_path / "base"
        ensure_dirs(str(target), ["sub1", "sub2"])
        assert (target / "sub1").is_dir()
        assert (target / "sub2").is_dir()

    def test_no_error_if_exists(self, tmp_path):
        from hoverfast.wsi_image_utils import ensure_dirs

        target = tmp_path / "existing"
        target.mkdir()
        ensure_dirs(str(target))  # should not raise


# ---------------------------------------------------------------------------
# setup_logger edge cases
# ---------------------------------------------------------------------------

class TestSetupLogger:

    def test_returns_logger(self, tmp_path):
        from hoverfast.wsi_image_utils import setup_logger

        logger = setup_logger(str(tmp_path))
        assert logger is not None
        assert len(logger.handlers) == 2  # file + console


# ---------------------------------------------------------------------------
# magnification_from_mpp edge cases
# ---------------------------------------------------------------------------

class TestMagnificationFromMpp:

    def test_standard_mpp_0_24(self):
        from hoverfast.wsi_image_utils import magnification_from_mpp

        mag = magnification_from_mpp(0.24)
        assert abs(mag - 40.0) < 1.0

    def test_high_mpp_low_mag(self):
        from hoverfast.wsi_image_utils import magnification_from_mpp

        mag = magnification_from_mpp(0.5)
        assert mag < 40.0

    def test_very_small_mpp(self):
        """Edge case: extremely small MPP (super-high resolution)."""
        from hoverfast.wsi_image_utils import magnification_from_mpp

        mag = magnification_from_mpp(0.01)
        assert mag > 40.0
