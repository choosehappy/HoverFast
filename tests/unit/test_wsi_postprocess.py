#!/usr/bin/env python3
"""Unit tests for WSI post-processing (hoverfast/wsi_postprocess.py)."""

import numpy as np
from hoverfast.wsi_postprocess import pre_watershed


class TestPreWatershed:
    def test_all_zero_mask_returns_none(self):
        output_mask = np.zeros((64, 64), dtype=np.uint8)
        maps = [np.random.rand(64, 64).astype(np.float32)] * 2
        dist, marker, opening = pre_watershed(output_mask, maps)
        assert dist is None
        assert marker is None
        assert opening is None

    def test_nonzero_mask_returns_valid(self):
        output_mask = np.zeros((64, 64), dtype=np.uint8)
        output_mask[20:40, 20:40] = 1
        maps = [np.random.rand(64, 64).astype(np.float32)] * 2
        dist, marker, opening = pre_watershed(output_mask, maps)
        assert dist is not None
        assert marker is not None
        assert opening is not None

    def test_output_shapes(self):
        output_mask = np.zeros((128, 128), dtype=np.uint8)
        output_mask[30:50, 30:50] = 1
        maps = [np.random.rand(128, 128).astype(np.float32)] * 2
        dist, marker, opening = pre_watershed(output_mask, maps)
        assert dist.shape == (128, 128)
        assert marker.shape == (128, 128)
        assert opening.shape == (128, 128)

    def test_marker_has_labels(self):
        output_mask = np.zeros((64, 64), dtype=np.uint8)
        output_mask[10:30, 10:30] = 1
        maps = [np.random.rand(64, 64).astype(np.float32)] * 2
        _, marker, _ = pre_watershed(output_mask, maps)
        unique_labels = np.unique(marker)
        assert len(unique_labels) >= 2

    def test_dist_range(self):
        output_mask = np.zeros((64, 64), dtype=np.uint8)
        output_mask[15:35, 15:35] = 1
        maps = [np.random.rand(64, 64).astype(np.float32)] * 2
        dist, _, _ = pre_watershed(output_mask, maps)
        assert dist.min() <= 0
