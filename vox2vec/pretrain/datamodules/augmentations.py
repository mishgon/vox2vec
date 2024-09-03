from dataclasses import dataclass
from typing import Tuple, Union, Sequence
import numpy as np
import random
from scipy.ndimage import gaussian_filter1d

from vox2vec.utils.misc import normalize_axis_list


@dataclass
class ColorAugmentations:
    blur_or_sharpen_p: float = 0.8
    blur_sigma_range: Tuple[float, float] = (0.0, 1.5)
    sharpen_sigma_range: Tuple[float, float] = (0.0, 1.5)
    sharpen_alpha_range: Tuple[float, float] = (0.0, 2.0)
    noise_p: float = 0.8
    noise_sigma_range: float = (0.0, 0.1)
    invert_p: float = 0.0
    brightness_p: float = 0.8
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_p: float = 0.8
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    gamma_p: float = 0.8
    gamma_range: Tuple[float, float] = (0.8, 1.25)


def augment_color(
        image: np.ndarray,
        voxel_spacing: np.ndarray,
        color_augmentations: ColorAugmentations
) -> np.ndarray:
    if random.uniform(0, 1) < color_augmentations.blur_or_sharpen_p:
        if random.uniform(0, 1) < 0.5:
            # random gaussian blur in axial plane
            sigma = random.uniform(*color_augmentations.blur_sigma_range) / voxel_spacing[:2]
            image = gaussian_filter(image, sigma, axis=(0, 1))
        else:
            sigma = random.uniform(*color_augmentations.sharpen_sigma_range) / voxel_spacing[:2]
            alpha = random.uniform(*color_augmentations.sharpen_alpha_range)
            image = gaussian_sharpen(image, sigma, alpha, axis=(0, 1))

    if random.uniform(0, 1) < color_augmentations.noise_p:
        # gaussian noise
        noise_sigma = random.uniform(*color_augmentations.noise_sigma_range)
        image = image + np.random.normal(0, noise_sigma, size=image.shape).astype('float32')

    if random.uniform(0, 1) < color_augmentations.invert_p:
        # invert
        image = 1.0 - image

    if random.uniform(0, 1) < color_augmentations.brightness_p:
        # adjust brightness
        brightness_factor = random.uniform(*color_augmentations.brightness_range)
        image = np.clip(image * brightness_factor, 0.0, 1.0)

    if random.uniform(0, 1) < color_augmentations.contrast_p:
        # adjust contrast
        contrast_factor = random.uniform(*color_augmentations.contrast_range)
        mean = image.mean()
        image = np.clip((image - mean) * contrast_factor + mean, 0.0, 1.0)

    if random.uniform(0, 1) < color_augmentations.gamma_p:
        image = np.clip(image, 0.0, 1.0)
        gamma = random.uniform(*color_augmentations.gamma_range)
        image = np.power(image, gamma)

    return image


def gaussian_filter(
        x: np.ndarray,
        sigma: Union[float, Sequence[float]],
        axis: Union[int, Sequence[int]]
) -> np.ndarray:
    axis = normalize_axis_list(axis, x.ndim)
    sigma = np.broadcast_to(sigma, len(axis))
    for sgm, ax in zip(sigma, axis):
        x = gaussian_filter1d(x, sgm, ax)
    return x


def gaussian_sharpen(
        x: np.ndarray,
        sigma: Union[float, Sequence[float]],
        alpha: float,
        axis: Union[int, Sequence[int]]
) -> np.ndarray:
    return x + alpha * (x - gaussian_filter(x, sigma, axis))
