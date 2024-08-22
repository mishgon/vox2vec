from typing import Callable, Sequence, Union, Any
import itertools
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm

import numpy as np
from numpy.core.numeric import normalize_axis_tuple


def normalize_axis_list(axis, ndim):
    return list(normalize_axis_tuple(axis, ndim))


def is_diagonal(matrix: np.ndarray) -> bool:
    return np.allclose(matrix, np.diag(np.diag(matrix)))


def apply_along_axis(
        func: Callable,
        x: np.ndarray,
        axis: Union[int, Sequence[int]],
        n_jobs: int = -1,
        *args: Any,
        **kwargs: Any
) -> np.ndarray:
    """
    Apply ``func`` to slices from ``x`` taken along ``axis``.
    ``args`` and ``kwargs`` are passed as additional arguments.

    Notes
    -----
    ``func`` must return an array of the same shape as it received.
    """
    axis = normalize_axis_tuple(axis, x.ndim)
    if len(axis) == x.ndim:
        return func(x)

    last = tuple(range(-len(axis), 0))
    x = np.moveaxis(x, axis, last)

    with parallel_backend('threading', n_jobs=n_jobs):
        y = Parallel()(
            delayed(func)(slc, *args, **kwargs)
            for slc in x.reshape(-1, *x.shape[-len(axis):])
        )

    y = np.stack(y)
    return np.moveaxis(y.reshape(*x.shape), last, axis)


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


class ProgressParallel(Parallel):
    def __init__(self, *args, total=None, **kwargs):
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(total=self._total) as self._pbar:
            return super().__call__(*args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
