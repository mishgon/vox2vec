from typing import *
import itertools
import numpy as np

from scipy.optimize import linear_sum_assignment


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


def slices_to_box(slices: Sequence[slice]) -> np.ndarray:
    return np.array([[s.start for s in slices], [s.stop for s in slices]])


def zoom_box(box: np.ndarray, scale_factor: Union[float, Sequence[float]]) -> np.ndarray:
    start, stop = box * scale_factor
    start = np.floor(start).astype(int)
    stop = np.ceil(stop).astype(int)
    return np.array([start, stop])


def compute_boxes_ious(boxes_1: Sequence[np.ndarray], boxes_2: Sequence[np.ndarray]) -> np.ndarray:
    """Find pairwise IOUs between two sequences of boxes.
    """
    assert len(boxes_1) and len(boxes_2)

    boxes_1 = np.stack(boxes_1)[:, None]
    boxes_2 = np.stack(boxes_2)[None]

    starts = np.maximum(boxes_1[:, :, 0], boxes_2[:, :, 0])
    stops = np.minimum(boxes_1[:, :, 1], boxes_2[:, :, 1])
    intersections_volumes = np.prod(np.maximum(stops - starts, 0), -1)

    boxes_gt_volumes = np.prod(boxes_1[:, :, 1] - boxes_1[:, :, 0], -1)
    boxes_pred_volumes = np.prod(boxes_2[:, :, 1] - boxes_2[:, :, 0], -1)

    return intersections_volumes / (boxes_gt_volumes + boxes_pred_volumes - intersections_volumes)


def match_boxes(
        boxes_1: Sequence[np.ndarray],
        boxes_2: Sequence[np.ndarray],
        min_iou: float
) -> Tuple[np.ndarray, np.ndarray]:
    if not len(boxes_1) or not len(boxes_2):
        return np.array([], dtype=int), np.array([], dtype=int)

    ious = compute_boxes_ious(boxes_1, boxes_2)
    matches = ious >= min_iou
    indices1, indices2 = linear_sum_assignment(matches.astype(int), maximize=True)
    matched = matches[indices1, indices2]
    return indices1[matched], indices2[matched]


def limit_box(box: np.ndarray, limit: Union[int, Sequence[int]]) -> np.ndarray:
    start, stop = box
    start = np.maximum(start, 0)
    stop = np.minimum(stop, limit)
    return np.array([start, stop])


def get_union_box(*boxes: np.ndarray) -> np.ndarray:
    start = np.min([box[0] for box in boxes], axis=0)
    stop = np.max([box[1] for box in boxes], axis=0)
    return np.array([start, stop])


def resize_box(box: np.ndarray, min_size: np.ndarray, max_stop: np.ndarray) -> np.ndarray:
    assert np.all(box[1] <= max_stop)
    assert np.all(min_size <= max_stop)

    old_start, old_stop = box
    old_size = old_stop - old_start

    new_size = np.maximum(old_size, min_size)
    new_start = np.maximum(old_start - (new_size - old_size) // 2, 0)
    new_stop = np.minimum(new_start + new_size, max_stop)
    new_start = new_stop - new_size
    assert np.all(new_start >= 0), (new_start, new_stop)
    return np.array([new_start, new_stop])
