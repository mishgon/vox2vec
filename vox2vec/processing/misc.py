from typing import Tuple
import numpy as np
from skimage.segmentation import flood
from skimage.measure import block_reduce
from imops import pad

from vox2vec.utils.misc import mask_to_bbox


BODY_THRESHOLD_HU = -500


def get_body_box(image: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> np.ndarray:
    block_size = tuple(np.int64(np.ceil(5.0 / np.array(voxel_spacing))))
    mask = block_reduce(image >= BODY_THRESHOLD_HU, block_size=block_size, func=np.min)
    bbox = mask_to_bbox(mask)
    return bbox * block_size


def get_body_mask(image: np.ndarray) -> np.ndarray:
    air_mask = image < BODY_THRESHOLD_HU
    air_mask = pad(air_mask, padding=1, axis=(0, 1), padding_values=True, num_threads=-1, backend='Scipy')
    body_mask = ~flood(air_mask, seed_point=(0, 0, 0))
    body_mask = body_mask[1:-1, 1:-1]
    return body_mask
