from typing import Sequence, Union
import numpy as np
from scipy.ndimage import gaussian_filter1d

from vox2vec.utils.misc import normalize_axis_list


def rescale_hu_piecewise(
        image: np.ndarray,
        hu_pivots: Sequence[float] = (-1000., -200., 200., 1500.),
        rescaled_pivots: Sequence[float] = (0.0, 0.4, 0.8, 1.0)
) -> np.ndarray:
    """Proposed in https://arxiv.org/abs/2102.01897.
    """
    rescaled_image = np.zeros_like(image)
    rescaled_image[image < hu_pivots[0]] = rescaled_pivots[0]
    for hu1, hu2, p1, p2 in zip(hu_pivots, hu_pivots[1:], rescaled_pivots, rescaled_pivots[1:]):
        mask = (image >= hu1) & (image < hu2)
        rescaled_image[mask] = (image[mask] - hu1) / (hu2 - hu1) * (p2 - p1) + p1
    rescaled_image[image >= hu_pivots[-1]] = rescaled_pivots[-1]
    return rescaled_image


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
