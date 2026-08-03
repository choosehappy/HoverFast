#!/usr/bin/env python3
"""Unit tests for stain deconvolution utilities (hoverfast/utils_stain_deconv.py)."""

import pytest
import torch
from hoverfast.utils_stain_deconv import (
    extract_h_channel_and_stack,
    hed_to_rgb_torch,
    rgb_to_hed_torch,
)


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


class TestRgbToHedTorch:
    def test_output_shape(self, cpu_device):
        batch = torch.rand(2, 64, 64, 3, dtype=torch.float16)
        hed = rgb_to_hed_torch(batch, cpu_device)
        assert hed.shape == (2, 64, 64, 3)

    def test_non_negative(self, cpu_device):
        batch = torch.rand(1, 32, 32, 3, dtype=torch.float16)
        hed = rgb_to_hed_torch(batch, cpu_device)
        assert (hed >= 0).all()

    def test_white_input_near_zero(self, cpu_device):
        white = torch.ones(1, 16, 16, 3, dtype=torch.float16)
        hed = rgb_to_hed_torch(white, cpu_device)
        assert hed.abs().max() < 0.5

    def test_dtype_preserved(self, cpu_device):
        batch = torch.rand(1, 32, 32, 3, dtype=torch.float16)
        hed = rgb_to_hed_torch(batch, cpu_device)
        assert hed.dtype == torch.float16


class TestHedToRgbTorch:
    def test_output_shape(self, cpu_device):
        batch = torch.rand(2, 64, 64, 3, dtype=torch.float16)
        rgb = hed_to_rgb_torch(batch, cpu_device)
        assert rgb.shape == (2, 64, 64, 3)

    def test_output_clamped(self, cpu_device):
        batch = torch.rand(1, 32, 32, 3, dtype=torch.float16) * 5
        rgb = hed_to_rgb_torch(batch, cpu_device)
        assert (rgb >= 0).all()
        assert (rgb <= 1).all()

    def test_dtype_preserved(self, cpu_device):
        batch = torch.rand(1, 32, 32, 3, dtype=torch.float16)
        rgb = hed_to_rgb_torch(batch, cpu_device)
        assert rgb.dtype == torch.float16


class TestExtractHChannelAndStack:
    def test_output_shape(self):
        batch = torch.rand(2, 64, 64, 3, dtype=torch.float16)
        result = extract_h_channel_and_stack(batch)
        assert result.shape == (2, 64, 64, 3)

    def test_only_h_channel_nonzero(self):
        batch = torch.rand(1, 32, 32, 3, dtype=torch.float16)
        result = extract_h_channel_and_stack(batch)
        assert (result[:, :, :, 0] != 0).any()
        assert (result[:, :, :, 1] == 0).all()
        assert (result[:, :, :, 2] == 0).all()

    def test_h_values_match_input(self):
        batch = torch.rand(1, 32, 32, 3, dtype=torch.float16)
        result = extract_h_channel_and_stack(batch)
        assert torch.allclose(result[:, :, :, 0], batch[:, :, :, 0])


class TestRoundTrip:
    def test_rgb_hed_rgb_approximate(self, cpu_device):
        rgb = torch.rand(1, 32, 32, 3, dtype=torch.float16)
        rgb[rgb < 1e-6] = 1e-6
        hed = rgb_to_hed_torch(rgb, cpu_device)
        recovered = hed_to_rgb_torch(hed, cpu_device)
        assert recovered.shape == rgb.shape
