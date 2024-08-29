from typing import Sequence, Union
import numpy as np
from scipy.ndimage import gaussian_filter1d

from vox2vec.utils.misc import normalize_axis_list


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
