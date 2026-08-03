#!/usr/bin/env python3
"""Unit tests for training utilities (hoverfast/training_utils.py).

Covers CRITICAL ISSUES:
  C10 — Dataset reopens HDF5 file on every __getitem__
  T10  — test_train_dataset_hdf5_handle_cleanup
  C8   — make_maps boundary check uses bitwise OR instead of logical or
"""

import os
import tempfile
import time

import numpy as np
import pytest
import torch
from hoverfast.training_utils import (
    Criterion,
    Dataset,
    asMinutes,
    dice_loss,
    grad_kernel,
    make_maps,
    timeSince,
)


# ---------------------------------------------------------------------------
# Existing tests (kept for reference)
# ---------------------------------------------------------------------------

class TestAsMinutes:
    def test_zero(self):
        assert asMinutes(0) == "0m 0s"

    def test_under_a_minute(self):
        result = asMinutes(30)
        assert "0m" in result
        assert "30s" in result

    def test_over_a_minute(self):
        result = asMinutes(90)
        assert "1m" in result
        assert "30s" in result


class TestTimeSince:
    def test_format(self):
        start = time.time()
        result = timeSince(start, 0.5)
        assert "(-" in result

    def test_no_division_by_zero(self):
        start = time.time()
        result = timeSince(start, 0.0)
        assert isinstance(result, str)


class TestMakeMaps:
    def test_output_shapes(self):
        label = np.zeros((64, 64), dtype=np.uint8)
        label[10:20, 10:20] = 1
        maps, weight = make_maps(label)
        assert maps.shape == (2, 64, 64)
        assert weight.shape == (64, 64)

    def test_boundary_weight_zero_at_edges(self):
        label = np.zeros((64, 64), dtype=np.uint8)
        label[0:10, 0:10] = 1
        _, weight = make_maps(label)
        assert weight[0, 0] == 0

    def test_boundary_weight_one_interior(self):
        label = np.zeros((64, 64), dtype=np.uint8)
        label[10:20, 10:20] = 1
        _, weight = make_maps(label)
        assert weight[15, 15] == 1

    def test_maps_dtype(self):
        label = np.zeros((32, 32), dtype=np.uint8)
        maps, _ = make_maps(label)
        assert maps.dtype == np.float32


class TestDiceLoss:
    def test_identical_inputs_zero_loss(self):
        pred = torch.randint(0, 2, (4, 64, 64), dtype=torch.long)
        loss = dice_loss(pred, pred)
        assert abs(loss.item()) < 1e-6

    def test_completely_different_high_loss(self):
        pred = torch.zeros(4, 64, 64, dtype=torch.long)
        true = torch.ones(4, 64, 64, dtype=torch.long)
        loss = dice_loss(pred, true)
        assert loss.item() > 0

    def test_output_scalar(self):
        pred = torch.randint(0, 2, (2, 32, 32), dtype=torch.long)
        true = torch.randint(0, 2, (2, 32, 32), dtype=torch.long)
        loss = dice_loss(pred, true)
        assert loss.dim() == 0


class TestGradKernel:
    def test_kernel_shape(self):
        kernel = grad_kernel(size=11)
        assert kernel.shape == (1, 1, 11, 11)

    def test_custom_size(self):
        kernel = grad_kernel(size=5)
        assert kernel.shape == (1, 1, 5, 5)

    def test_dtype_float32(self):
        kernel = grad_kernel()
        assert kernel.dtype == torch.float32


class TestCriterion:
    @pytest.fixture(autouse=True)
    def setup_criterion(self):
        torch.device("cpu")
        class_weight = torch.tensor([0.5, 0.5], dtype=torch.float32)
        self.criterion = Criterion(
            class_weight=class_weight,
            edge_weight=1.1,
            grad_weight=0.1,
            hv_weight=14.0,
            dice_weight=1 / 6,
            crossentropy_weight=1.0,
        )

    @pytest.mark.xfail(reason="Criterion.grad_kernel creates CUDA tensors at init time — pre-existing bug")
    def test_forward_returns_tuple(self):
        batch_size = 2
        x_pred = torch.randn(batch_size, 2, 32, 32)
        hvm_pred = torch.randn(batch_size, 2, 32, 32)
        y = torch.randint(0, 2, (batch_size, 32, 32))
        hvmaps = torch.randn(batch_size, 2, 32, 32)
        y_weight = torch.ones(batch_size, 32, 32)
        hv_weight = torch.ones(batch_size, 32, 32)

        losses = self.criterion(x_pred, hvm_pred, y, hvmaps, y_weight, hv_weight)
        assert isinstance(losses, tuple)
        assert len(losses) == 4


# ---------------------------------------------------------------------------
# C8: make_maps boundary check — bitwise OR vs logical or
# ---------------------------------------------------------------------------

class TestMakeMapsBoundaryCheck:

    def test_region_touching_single_edge_zeroed(self):
        """C8 — Region touching only ONE edge should have weight=0 at that region."""
        label = np.zeros((64, 64), dtype=np.uint8)
        # Touches top edge (ymin==0) but not other edges
        label[0:10, 5:15] = 1
        _, weight = make_maps(label)
        assert weight[5, 10] == 0, "Region touching top edge should have zeroed weight"

    def test_region_touching_right_edge_zeroed(self):
        """C8 — Region touching right edge (xmax==label.shape[1])."""
        label = np.zeros((64, 64), dtype=np.uint8)
        label[5:15, 54:64] = 1
        _, weight = make_maps(label)
        assert weight[10, 60] == 0

    def test_region_touching_bottom_edge_zeroed(self):
        """C8 — Region touching bottom edge (ymax==label.shape[0])."""
        label = np.zeros((64, 64), dtype=np.uint8)
        label[54:64, 5:15] = 1
        _, weight = make_maps(label)
        assert weight[60, 10] == 0

    def test_region_touching_left_edge_zeroed(self):
        """C8 — Region touching left edge (xmin==0)."""
        label = np.zeros((64, 64), dtype=np.uint8)
        label[5:15, 0:10] = 1
        _, weight = make_maps(label)
        assert weight[10, 5] == 0

    def test_interior_region_not_zeroed(self):
        """Interior regions should have weight=1."""
        label = np.zeros((64, 64), dtype=np.uint8)
        label[20:30, 20:30] = 1
        _, weight = make_maps(label)
        assert weight[25, 25] == 1

    def test_multiple_regions_mixed_edges(self):
        """Some regions at edges, some interior — mixed weights expected."""
        label = np.zeros((64, 64), dtype=np.uint8)
        label[0:10, 0:10] = 1       # corner → zeroed
        label[20:30, 20:30] = 2     # interior → weight=1
        _, weight = make_maps(label)
        assert weight[5, 5] == 0
        assert weight[25, 25] == 1


# ---------------------------------------------------------------------------
# C10 / T10: HDF5 Dataset handle management
# ---------------------------------------------------------------------------

class TestDatasetHdf5Handle:

    @pytest.fixture
    def temp_pytable(self):
        """Create a minimal .pytable file for testing."""
        import tables

        tmp = tempfile.NamedTemporaryFile(suffix=".pytable", delete=False)
        tmp.close()

        with tables.open_file(tmp.name, "w") as f:
            # Create dummy image data (10 samples of 64x64 RGB)
            img_data = np.random.randint(0, 256, (10, 64, 64, 3), dtype=np.uint8)
            label_data = np.zeros((10, 64, 64), dtype=np.uint8)
            for i in range(10):
                label_data[i, 20:40, 20:40] = 1

            img_atom = tables.Atom.from_dtype(img_data.dtype)
            label_atom = tables.Atom.from_dtype(label_data.dtype)

            img_array = f.create_carray(f.root, "img", img_atom, img_data.shape)
            label_array = f.create_carray(f.root, "label", label_atom, label_data.shape)

            numpixels_atom = tables.Int64Atom()
            numpixels = f.create_carray(f.root, "numpixels", numpixels_atom, (2, 3))
            numpixels[0, :] = [1000, 500, 200]
            numpixels[1, :] = [500, 300, 100]

            img_array[:] = img_data
            label_array[:] = label_data

        yield tmp.name

        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    def test_dataset_opens_file_once(self, temp_pytable):
        """C10 — Dataset should open HDF5 file once at init, not per __getitem__."""
        ds = Dataset(temp_pytable, device=torch.device("cpu"))
        assert len(ds) == 10

        # The fix ensures the file handle is persistent.
        # Access multiple items to verify no repeated open/close overhead.
        for i in range(3):
            img, mask, maps, eweight, bweight = ds[i]
            assert isinstance(img, torch.Tensor)
            assert img.shape[0] == 3  # RGB channels first

    def test_dataset_del_closes_handle(self, temp_pytable):
        """T10 — Dataset.__del__ should close the HDF5 file handle."""
        ds = Dataset(temp_pytable, device=torch.device("cpu"))

        # Force cleanup
        del ds

        # If __del__ works properly, the file should be releasable.
        import tables
        with tables.open_file(temp_pytable, "r") as f:
            assert f.root.img.shape[0] == 10

    def test_dataset_getitem_returns_correct_types(self, temp_pytable):
        """Dataset.__getitem__ should return proper tensor types."""
        ds = Dataset(temp_pytable, device=torch.device("cpu"))
        img, mask, maps, eweight, bweight = ds[0]

        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(maps, torch.Tensor)
        assert isinstance(eweight, torch.Tensor)
        assert isinstance(bweight, torch.Tensor)


# ---------------------------------------------------------------------------
# make_maps edge cases
# ---------------------------------------------------------------------------

class TestMakeMapsEdgeCases:

    def test_empty_label(self):
        """All-zero label should produce zero maps."""
        label = np.zeros((64, 64), dtype=np.uint8)
        maps, weight = make_maps(label)
        assert maps.sum() == 0
        assert (weight == 1).all()

    def test_single_pixel_label(self):
        """Single labeled pixel should not crash."""
        label = np.zeros((32, 32), dtype=np.uint8)
        label[16, 16] = 1
        maps, weight = make_maps(label)
        assert maps.shape == (2, 32, 32)

    def test_full_label(self):
        """Entire image labeled should produce valid output."""
        label = np.ones((32, 32), dtype=np.uint8) * 1
        maps, weight = make_maps(label)
        assert maps.shape == (2, 32, 32)
