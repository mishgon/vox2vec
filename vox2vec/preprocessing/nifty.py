from typing import Tuple, Optional
import numpy as np

from vox2vec.utils.misc import is_diagonal


def affine_to_voxel_spacing(affine: np.ndarray) -> Tuple[float, float, float]:
    return tuple(np.abs(np.diag(affine[:3, :3])))


def to_canonical_orientation(
        image: np.ndarray,
        voxel_spacing: Optional[Tuple[float, float, float]],
        affine: np.ndarray
) -> np.ndarray:
    assert is_diagonal(affine[:3, :3])

    flip_axis = tuple(np.where(np.diag(affine[:3, :3]) < 0)[0])
    image = np.flip(image, axis=flip_axis)

    # from nifty-canonical to dicom-canonical
    image = image.transpose((1, 0, 2))
    image = np.flip(image, axis=(0, 1, 2))

    image = image.copy()

    return image, voxel_spacing
