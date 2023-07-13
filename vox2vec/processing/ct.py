from typing import *
import numpy as np
import math
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import flood
from imops import pad

from vox2vec.utils.misc import apply_along_axes


BODY_THRESHOLD_HU = -500
DENSE_LUNGS_THRESHOLD_HU = -750


def get_body_mask(image: np.ndarray) -> np.ndarray:
    air = image < BODY_THRESHOLD_HU
    return ~flood(pad(air, padding=1, axis=(0, 1), padding_values=True), seed_point=(0, 0, 0))[1:-1, 1:-1]


def get_lungs_mask(image: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
    from ira.lungs_segm import predict_lungs

    pixel_spacing = spacing[:2]
    slice_locations = np.arange(image.shape[-1]) * spacing[-1]
    return predict_lungs(image, pixel_spacing, slice_locations, legacy=True).binary


def dilate_lungs(lungs, pixel_spacing):
    max_distance_from_lungs_mm = 3
    footprint = disk(math.ceil(max_distance_from_lungs_mm / pixel_spacing[0]))
    return apply_along_axes(binary_dilation, lungs, axis=(0, 1), footprint=footprint)
