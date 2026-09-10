#!/usr/bin/env python3
"""Unit tests for WSI post-processing (hoverfast/wsi_postprocess.py).

Covers CRITICAL ISSUES:
  T3  — pre_watershed all-zero mask returns None
  T4  — post_processing_batch_task empty batch handling
"""

import numpy as np
import torch
from hoverfast.wsi_postprocess import (
    _contour_centroid,
    _repair_polygon,
    _simplify_contour,
    centroid_in_valid_region,
    post_processing_batch_task,
    pre_watershed,
)

# ---------------------------------------------------------------------------
# Existing tests (T3 already covered)
# ---------------------------------------------------------------------------


class TestPreWatershed:
    def test_all_zero_mask_returns_none(self):
        """T3 — pre_watershed returns (None, None, None) for all-zero mask."""
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


# ---------------------------------------------------------------------------
# T4: post_processing_batch_task edge cases
# ---------------------------------------------------------------------------


class TestPostProcessingBatchTask:
    def _make_slide_data(self):
        return {
            "region_size": 256,
            "stride": 128,
            "downfactor": 1.0,
            "threshold": 50.0,
            "poly_simplification": 6.0,
        }

    def test_empty_batch_returns_zero(self):
        """T4 — empty tensor batch should return 0 and not crash."""
        from multiprocessing import Queue

        slide_data = self._make_slide_data()

        # Create zero-sized tensors (batch_size=0)
        output_tensor = torch.zeros((0, 256, 256), dtype=torch.bool)
        maps_tensor = torch.zeros((0, 2, 256, 256), dtype=torch.float32)
        coords_tensor = torch.zeros((0, 2), dtype=torch.int64)

        queue = Queue()
        result = post_processing_batch_task(output_tensor, maps_tensor, coords_tensor, slide_data, queue, None)
        assert result == 0

    def test_all_zero_masks_returns_zero(self):
        """When every patch has no nuclei detected, result should be 0."""
        from multiprocessing import Queue

        slide_data = self._make_slide_data()

        output_tensor = torch.zeros((3, 256, 256), dtype=torch.bool)
        maps_tensor = torch.rand(3, 2, 256, 256, dtype=torch.float32)
        coords_tensor = torch.tensor([[0, 0], [100, 100], [200, 200]], dtype=torch.int64)

        queue = Queue()
        result = post_processing_batch_task(output_tensor, maps_tensor, coords_tensor, slide_data, queue, None)
        assert result == 0

    def test_single_patch_with_nuclei(self):
        """A single patch with a clear nuclei blob should produce at least one feature."""
        from multiprocessing import Queue

        slide_data = self._make_slide_data()

        # Create a mask with one clear circular region
        output_tensor = torch.zeros((1, 256, 256), dtype=torch.bool)
        yy, xx = np.ogrid[:256, :256]
        mask = (yy - 128) ** 2 + (xx - 128) ** 2 < 30**2
        output_tensor[0] = torch.from_numpy(mask.astype(bool))

        maps_tensor = torch.rand(1, 2, 256, 256, dtype=torch.float32)
        coords_tensor = torch.tensor([[0, 0]], dtype=torch.int64)

        queue = Queue()
        result = post_processing_batch_task(output_tensor, maps_tensor, coords_tensor, slide_data, queue, None)
        assert result >= 1

    def test_multiple_patches_mixed_results(self):
        """Some patches with nuclei, some without — total count should be correct."""
        from multiprocessing import Queue

        slide_data = self._make_slide_data()

        # Patch 0: has a blob; Patch 1: all zeros
        output_tensor = torch.zeros((2, 256, 256), dtype=torch.bool)
        yy, xx = np.ogrid[:256, :256]
        mask = (yy - 128) ** 2 + (xx - 128) ** 2 < 30**2
        output_tensor[0] = torch.from_numpy(mask.astype(bool))
        # Patch 1 stays all zeros

        maps_tensor = torch.rand(2, 2, 256, 256, dtype=torch.float32)
        coords_tensor = torch.tensor([[0, 0], [256, 256]], dtype=torch.int64)

        queue = Queue()
        result = post_processing_batch_task(output_tensor, maps_tensor, coords_tensor, slide_data, queue, None)
        assert result >= 1  # at least the nuclei from patch 0


# ---------------------------------------------------------------------------
# pre_watershed edge cases
# ---------------------------------------------------------------------------


class TestPreWatershedEdgeCases:
    def test_single_pixel_nucleus(self):
        """A single-pixel 'nucleus' should still produce valid output."""
        output_mask = np.zeros((32, 32), dtype=np.uint8)
        output_mask[16, 16] = 1
        maps = [np.random.rand(32, 32).astype(np.float32)] * 2
        dist, marker, opening = pre_watershed(output_mask, maps)
        assert dist is not None

    def test_full_mask(self):
        """Entire image is tissue — should still work."""
        output_mask = np.ones((64, 64), dtype=np.uint8)
        maps = [np.random.rand(64, 64).astype(np.float32)] * 2
        dist, marker, opening = pre_watershed(output_mask, maps)
        assert dist is not None
        assert marker is not None

    def test_maps_different_dtypes(self):
        """Maps passed as list of arrays with different dtypes should normalize."""
        output_mask = np.zeros((64, 64), dtype=np.uint8)
        output_mask[20:30, 20:30] = 1
        maps = [np.random.rand(64, 64).astype(np.float64)] * 2
        dist, marker, opening = pre_watershed(output_mask, maps)
        assert dist is not None


# ---------------------------------------------------------------------------
# Tile-margin partition (over-segmentation / double-count regression)
# ---------------------------------------------------------------------------


class TestValidRegionPartition:
    """The valid region must be half-open so adjacent tiles never both claim a
    cell sitting exactly on the shared margin (the old closed interval did)."""

    REGION = 256
    STRIDE = 128

    def test_interior_centroid_is_valid(self):
        assert centroid_in_valid_region(128, 128, self.REGION, self.STRIDE)

    def test_lower_bound_is_inclusive(self):
        assert centroid_in_valid_region(64, 64, self.REGION, self.STRIDE)

    def test_upper_bound_is_exclusive(self):
        # x = 192 is the first pixel owned by the next tile.
        assert not centroid_in_valid_region(192, 128, self.REGION, self.STRIDE)
        assert not centroid_in_valid_region(128, 192, self.REGION, self.STRIDE)

    def test_boundary_claimed_by_exactly_one_tile(self):
        step = self.REGION - self.STRIDE  # 128
        # A physical x on the margin between tile 0 and tile 1.
        physical_x = 192
        claims = [
            centroid_in_valid_region(physical_x - tile * step, 128, self.REGION, self.STRIDE) for tile in range(3)
        ]
        assert sum(claims) == 1, f"expected exactly one claim, got {claims}"

    def test_every_coordinate_claimed_at_most_once(self):
        step = self.REGION - self.STRIDE
        for physical_x in range(0, 1024):
            claims = sum(
                centroid_in_valid_region(physical_x - tile * step, 128, self.REGION, self.STRIDE) for tile in range(9)
            )
            assert claims <= 1, f"coordinate {physical_x} claimed by {claims} tiles"

    def test_partition_has_no_gap_inside_coverage(self):
        step = self.REGION - self.STRIDE
        # Every integer strictly inside the covered span must be claimed once.
        for physical_x in range(64, 960):
            claims = sum(
                centroid_in_valid_region(physical_x - tile * step, 128, self.REGION, self.STRIDE) for tile in range(9)
            )
            assert claims == 1, f"coordinate {physical_x} claimed by {claims} tiles"


# ---------------------------------------------------------------------------
# Contour helpers (extracted from watershed_object)
# ---------------------------------------------------------------------------


class TestContourHelpers:
    def test_simplify_zero_tolerance_is_noop(self):
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
        out = _simplify_contour(contour, 0)
        assert np.array_equal(out, contour)

    def test_repair_valid_polygon_is_noop(self):
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
        out = _repair_polygon(contour)
        assert out is not None
        assert np.array_equal(out, contour)

    def test_contour_centroid_of_square(self):
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
        assert _contour_centroid(contour) == (5, 5)

    def test_contour_centroid_zero_area_returns_none(self):
        contour = np.array([[[0, 0]], [[0, 0]], [[0, 0]]], dtype=np.int32)
        assert _contour_centroid(contour) is None
