#!/usr/bin/env python3
"""Unit tests for augmentation (hoverfast/augment.py)."""

import numpy as np
import pytest
from hoverfast.augment import HEDJitterAugmentation, randaugment


class TestHEDJitterAugmentationInit:
    def test_single_number_alpha_beta(self):
        aug = HEDJitterAugmentation(0.4, 0.01)
        assert isinstance(aug.alpha[0], float)
        assert aug.alpha[0] < 0 < aug.alpha[1]

    def test_tuple_alpha_beta(self):
        aug = HEDJitterAugmentation((-0.2, 0.3), (-0.01, 0.02))
        assert aug.alpha == (-0.2, 0.3)
        assert aug.beta == (-0.01, 0.02)

    def test_invalid_alpha_tuple(self):
        with pytest.raises(ValueError, match="Alpha range"):
            HEDJitterAugmentation((0.5, -0.5), 0.01)

    def test_invalid_beta_type(self):
        with pytest.raises(TypeError, match="Beta must be"):
            HEDJitterAugmentation(0.4, "invalid")

    def test_cap_attribute_exists(self):
        aug = HEDJitterAugmentation(0.4, 0.01)
        assert hasattr(aug, "cap")
        assert len(aug.cap) == 3


class TestHEDJitterAugmentationApply:
    def test_adjust_hed_output_shape(self):
        aug = HEDJitterAugmentation(0.4, 0.01)
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = aug.adjust_HED(img)
        assert result.shape == (64, 64, 3)

    def test_adjust_hed_output_dtype(self):
        aug = HEDJitterAugmentation(0.4, 0.01)
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = aug.adjust_HED(img)
        assert result.dtype == np.uint8

    def test_adjust_hed_value_range(self):
        aug = HEDJitterAugmentation(0.4, 0.01)
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = aug.adjust_HED(img)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_apply_method(self):
        aug = HEDJitterAugmentation(0.4, 0.01)
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = aug.apply(img)
        assert result.shape == (64, 64, 3)


class TestRandaugment:
    def test_returns_compose(self):
        from albumentations.core.composition import Compose

        transforms = randaugment()
        assert isinstance(transforms, Compose)

    def test_transforms_produces_valid_output(self):
        transforms = randaugment()
        img = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        result = transforms(image=img, mask=mask)
        assert "image" in result
        assert "mask" in result
