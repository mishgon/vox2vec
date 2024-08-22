from typing import Sequence
import numpy as np

from imops import pad

from vox2vec.utils.misc import normalize_axis_list


def restore_crop(
        x: np.ndarray,
        box: np.ndarray,
        size_before_crop: Sequence[int],
        axis: Sequence[int],
        padding_values=0
) -> np.ndarray:
    axis = normalize_axis_list(axis, x.ndim)

    assert box.shape[1] == len(size_before_crop) == len(axis)

    padding = np.array([box[0], size_before_crop - box[1]], dtype=int).T
    return pad(x, padding, axis, padding_values, num_threads=-1, backend='Scipy')
