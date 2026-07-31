#!/usr/bin/env python3
from __future__ import annotations

import numbers
from typing import Any

import numpy as np
from albumentations import (
    Blur,
    Compose,
    GaussNoise,
    HorizontalFlip,
    ISONoise,
    MotionBlur,
    MultiplicativeNoise,
    RandomBrightnessContrast,
    RandomGamma,
    RandomResizedCrop,
    RandomSizedCrop,
    Rotate,
    VerticalFlip,
)
from albumentations.core.transforms_interface import ImageOnlyTransform
from skimage.color import hed2rgb, rgb2hed


class HEDJitterAugmentation(ImageOnlyTransform):  # type: ignore[misc]
    """
    Custom color augmentation using HED jittering for stain normalization.

    This augmentation randomly adjusts the Hematoxylin-Eosin-DAB (HED) color space of an image
    to simulate variations in staining.

    Parameters:
    alpha (float or tuple): Range of alpha values for scaling the HED channels. If a single number, the range is (-alpha, alpha).
    beta (float or tuple): Range of beta values for shifting the HED channels. If a single number, the range is (-beta, beta).
    always_apply (bool, optional): Whether to always apply this transformation. Default is False.
    p (float, optional): Probability of applying the transformation. Default is 0.5.
    """

    def __init__(
        self,
        alpha: float | tuple[float, float],
        beta: float | tuple[float, float],
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(always_apply, p)
        if isinstance(alpha, numbers.Number):
            self.alpha = (-alpha, alpha)
        elif isinstance(alpha, tuple):
            if alpha[0] <= alpha[1]:
                self.alpha = alpha
            else:
                raise ValueError("Alpha range must be in the form (min, max).")
        else:
            raise TypeError("Alpha must be a number or a tuple.")

        if isinstance(beta, numbers.Number):
            self.beta = (-beta, beta)
        elif isinstance(beta, tuple):
            if beta[0] <= beta[1]:
                self.beta = beta
            else:
                raise ValueError("Beta range must be in the form (min, max).")
        else:
            raise TypeError("Beta must be a number or a tuple.")

        self.cap = np.array([1.87798274, 1.13473037, 1.57358807])

    alpha: tuple[float, float]
    beta: tuple[float, float]

    def adjust_HED(self, img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Adjust the HED color space of the image.

        Parameters:
        img (numpy.ndarray): Input image in RGB format.

        Returns:
        numpy.ndarray: Augmented image with adjusted HED channels.
        """
        img = np.array(img)

        alpha_val = np.random.uniform(1 + self.alpha[0], 1 + self.alpha[1], (1, 3))
        betti = np.random.uniform(self.beta[0], self.beta[1], (1, 3))

        alpha_val[0, 2] = min(alpha_val[0, 2], 2 * alpha_val[0, :2].prod() / alpha_val[0, :2].sum())

        s = rgb2hed(img) / self.cap
        s = alpha_val * (s + betti)
        nimg = hed2rgb(s * self.cap)

        return (255 * nimg).clip(0, 255).astype(np.uint8)  # type: ignore[no-any-return]

    def apply(self, image: np.ndarray[Any, Any], **params: Any) -> np.ndarray[Any, Any]:
        """
        Apply the custom HED jittering augmentation to the image.

        Parameters:
        image (numpy.ndarray): Input image in RGB format.

        Returns:
        numpy.ndarray: Augmented image.
        """
        augmented_image = self.adjust_HED(image)

        return augmented_image


def randaugment() -> Any:
    """
    Generate a random augmentation pipeline for image data.

    This function creates a set of random augmentations that includes
    HED jittering, flips, rotations, crops, brightness/contrast adjustments,
    gamma adjustments, blurs, and noise additions.

    Returns:
    albumentations.core.composition.Compose: A composition of random augmentations.
    """

    p = 0.8

    aug_always = [
        HEDJitterAugmentation((-0.4, 0.4), (-0.005, 0.01), p=p),
        VerticalFlip(p=p),
        HorizontalFlip(p=p),
        RandomResizedCrop(1024, 1024, scale=(0.95, 1.05), always_apply=True, p=p),
        Rotate(p=1, value=(255, 255, 255), crop_border=True),
        RandomSizedCrop((512, 512), 512, 512),
    ]

    aug_intensity = [
        RandomBrightnessContrast(p=p, brightness_limit=0.2, contrast_limit=0.2),
        RandomGamma(p=p, gamma_limit=(65, 140), eps=1e-7),
    ]

    blur = [Blur(blur_limit=5, p=0.3), MotionBlur(p=0.3, blur_limit=(3, 5))]

    aug_noise = [
        GaussNoise(p=p, var_limit=(120, 600)),
        ISONoise(p=p, intensity=(0.1, 0.4), color_shift=(0.05, 0.2)),
        MultiplicativeNoise(p=p, multiplier=(0.75, 1.25), elementwise=True),
    ]

    noise_ops = np.random.choice(aug_noise, 1, replace=False).tolist()
    intensity_ops = np.random.choice(aug_intensity, 1, replace=False).tolist()
    blur_ops = np.random.choice(blur, 1, replace=False).tolist()
    transforms = Compose(aug_always + intensity_ops + blur_ops + noise_ops)

    return transforms
