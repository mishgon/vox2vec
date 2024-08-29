import numpy as np
from typing import Sequence, Callable, Union
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
import matplotlib.patches as patches

from ipywidgets import interact, IntSlider


def _get_rows_cols(max_cols, data):
    columns = min(len(data), max_cols or len(data))
    return (len(data) - 1) // columns + 1, columns


def _simple_slice(arr, inds, axis):
    # this does the same as np.take() except only supports simple slicing, not
    # advanced indexing, and thus is much faster
    sl = [slice(None)] * arr.ndim
    sl[axis] = inds
    return arr[tuple(sl)]


def _slice_base(
        data: [np.ndarray],
        axis: int = -1,
        scale: int = 5,
        max_columns: int = None,
        colorbar: bool = False,
        show_axes: bool = False,
        cmap: Union[Colormap, str, Sequence[Colormap], Sequence[str]] = 'gray',
        vlim: Union[float, Sequence[float]] = None,
        callback: Callable = None,
        sliders: dict = None,
        titles: list = None
):
    cmap = np.broadcast_to(cmap, len(data)).tolist()
    vlim = np.broadcast_to(vlim, [len(data), 2]).tolist()
    rows, columns = _get_rows_cols(max_columns, data)
    sliders = sliders or {}
    if titles is None:
        titles = [None] * len(data)

    assert len(titles) == len(data)
    if 'idx' in sliders:
        raise ValueError('Overriding "idx" is not allowed.')

    def update(idx, **kwargs):
        fig, axes = plt.subplots(rows, columns, figsize=(scale * columns, scale * rows))
        axes = np.array(axes).flatten()
        ax: Axes
        # hide unneeded axes
        for ax in axes[len(data):]:
            ax.set_visible(False)
        for ax, x, cmap_, (vmin, vmax), title in zip(axes, data, cmap, vlim, titles):
            im = ax.imshow(_simple_slice(x, idx, axis), cmap=cmap_, vmin=vmin, vmax=vmax)
            if colorbar:
                fig.colorbar(im, ax=ax, orientation='horizontal')
            if not show_axes:
                ax.set_axis_off()
            if title is not None:
                ax.set_title(title)

        if callback is not None:
            callback(axes, idx=idx, **kwargs)

        plt.tight_layout()
        plt.show()

    interact(update, idx=IntSlider(min=0, max=data[0].shape[axis] - 1, continuous_update=False), **sliders)


def slice3d(
        *data: np.ndarray,
        axis: int = -1,
        scale: int = 5,
        max_columns: int = None,
        colorbar: bool = False,
        show_axes: bool = False,
        cmap: Union[Colormap, str] = 'gray',
        vlim: Union[float, Sequence[float]] = None,
        titles: Sequence[Union[str, None]] = None
):
    _slice_base(data, axis, scale, max_columns, colorbar, show_axes, cmap, vlim, titles=titles)


def draw_boxes_2d(image: np.ndarray, boxes: Sequence[np.ndarray], figsize=(5, 5), lw=1, show_axes=False):
    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap='gray')

    for box in boxes:
        start = box[0] - .5
        height, width = box[1] - box[0]

        rect = patches.Rectangle(start[::-1], width, height, linewidth=lw, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    if not show_axes:
        ax.set_axis_off()

    plt.show()


def draw_boxes_3d(image: np.ndarray, boxes: Sequence[np.ndarray], figsize=(5, 5), lw=1, show_axes=False):
    def update(idx):
        boxes_in = np.array([b for b in boxes if b[0, 2] <= idx < b[1, 2]])
        draw_boxes_2d(image.take(idx, axis=-1), boxes_in[..., :-1], figsize=figsize, lw=lw, show_axes=show_axes)

    interact(update, idx=IntSlider(min=0, max=image[0].shape[-1] - 1, continuous_update=False))
