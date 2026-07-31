#!/usr/bin/env python3
"""Unit tests for training utilities (hoverfast/training_utils.py)."""

import time

import numpy as np
import pytest
import torch
from hoverfast.training_utils import (
    Criterion,
    asMinutes,
    dice_loss,
    grad_kernel,
    make_maps,
    timeSince,
)


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
