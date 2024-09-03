from typing import Sequence, Optional
import itertools
import numpy as np


def get_random_box(
        image_size: Sequence[int],
        box_size: Sequence[int],
        pins: Optional[np.ndarray] = None
) -> np.ndarray:
    image_size = np.array(image_size)
    box_size = np.array(box_size)
    if not np.all(image_size >= box_size):
        raise ValueError(f'Can\'t sample patch of size {box_size} from image of size {image_size}')

    min_start = 0
    max_start = image_size - box_size

    if pins is not None:
        assert pins.ndim == 2
        assert pins.shape[1] == 3

        min_start = np.maximum(min_start, np.max(pins, axis=0) - box_size + 1)
        max_start = np.minimum(max_start, np.min(pins, axis=0))

        assert np.all(min_start <= max_start)

    start = np.random.randint(min_start, max_start + 1)

    return np.array([start, start + box_size])


def mask_to_bbox(mask: np.ndarray) -> np.ndarray:
    """
    Find the smallest box that contains all true values of the ``mask``.
    """
    if not mask.any():
        raise ValueError('The mask is empty.')

    start, stop = [], []
    for ax in itertools.combinations(range(mask.ndim), mask.ndim - 1):
        nonzero = np.any(mask, axis=ax)
        if np.any(nonzero):
            left, right = np.where(nonzero)[0][[0, -1]]
        else:
            left, right = 0, 0
        start.insert(0, left)
        stop.insert(0, right + 1)

    return np.array([start, stop])


def get_overlap_box(*boxes: np.ndarray) -> np.ndarray:
    start = np.max([box[0] for box in boxes], axis=0)
    stop = np.min([box[1] for box in boxes], axis=0)
    if not np.all(start < stop):
        return
    return np.array([start, stop])
