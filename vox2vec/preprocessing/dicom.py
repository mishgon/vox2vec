from typing import Sequence, Tuple, Optional
from enum import Enum
import itertools
import numpy as np

from pydicom import Dataset


Instance = Dataset
Series = Sequence[Dataset]


class Plane(Enum):
    Sagittal, Coronal, Axial = 0, 1, 2


def get_series_uid(series: Series) -> str:
    series_uid, = {i.SeriesInstanceUID for i in series}
    return series_uid


def get_instance_image_position_patient(instance: Instance) -> np.ndarray:
    return np.array(instance.ImagePositionPatient, dtype='float32')


def get_series_image_position_patient(series: Series) -> np.ndarray:
    return np.stack([get_instance_image_position_patient(i) for i in series])


def get_instance_image_orientation_patient(instance: Instance) -> np.ndarray:
    return np.array(instance.ImageOrientationPatient, dtype='float32')


def get_instance_orientation_matrix(instance: Instance) -> np.ndarray:
    row, col = get_instance_image_orientation_patient(instance).reshape(2, 3)
    return np.stack([row, col, np.cross(row, col)])


def get_series_orientation_matrix(series: Series) -> np.ndarray:
    """
    Returns a 3 x 3 orthogonal transition matrix from the image-based basis to the patient-based basis.
    Rows are coordinates of image-based basis vectors in the patient-based basis.
    Columns are coordinates of patient-based basis vectors in the image-based basis vectors.

    See https://dicom.innolitics.com/ciods/rt-dose/image-plane/00200037 for details.
    """
    om = get_instance_orientation_matrix(series[0])
    if not np.all([np.allclose(om, get_instance_orientation_matrix(i)) for i in series]):
        raise ValueError('Orientation matrix varies across slices.')
    return om


def orientation_matrix_to_image_planes(om: np.ndarray) -> Tuple[Plane]:
    return tuple(Plane(i) for i in np.argmax(np.abs(om), axis=1))


def orientation_matrix_to_slice_plane(om: np.ndarray) -> Plane:
    return orientation_matrix_to_image_planes(om)[2]


def get_instance_slice_plane(instance: Instance) -> Plane:
    return orientation_matrix_to_slice_plane(get_instance_orientation_matrix(instance))


def get_series_slice_plane(series: Series) -> Plane:
    return orientation_matrix_to_slice_plane(get_series_orientation_matrix(series))


def to_canonical_orientation(
        image: np.ndarray,
        voxel_spacing: Optional[Tuple[int, int, int]],
        om: np.ndarray
) -> np.ndarray:
    planes = orientation_matrix_to_image_planes(om)

    if set(planes) != {Plane.Sagittal, Plane.Coronal, Plane.Axial}:
        raise ValueError('Main image planes cannot be treated as saggital, coronal and axial.')

    if planes[2] != Plane.Axial:
        raise NotImplementedError('We do not know what is normal orientation for non-axial slice.')
 
    if planes[0] == Plane.Coronal:
        image = image.transpose((1, 0, 2))
        if voxel_spacing is not None:
            voxel_spacing = tuple(voxel_spacing[i] for i in [1, 0, 2])
        om = om[[1, 0, 2]]

    if om[1, 1] < 0:
        image = np.flip(image, axis=0)
    if om[0, 0] < 0:
        image = np.flip(image, axis=1)

    image = image.copy()

    return image, voxel_spacing


def order_series(series: Series) -> Series:
    index = get_series_slice_plane(series).value
    return sorted(series, key=lambda i: get_instance_image_position_patient(i)[index], reverse=True)


def get_series_slice_locations(series: Series) -> np.ndarray:
    """
    Computes slices location from ImagePositionPatient. 
    NOTE: the order of slice locations can be both increasing or decreasing for ordered series 
    (see order_series).
    """
    om = get_series_orientation_matrix(series)
    return np.array([get_instance_image_position_patient(i) @ om[-1] for i in series])


def locations_to_spacing(locations: Sequence[float]) -> float:
    spacings = np.round(np.abs(np.diff(locations)), 2)
    unique_spacings, counts = np.unique(spacings, return_counts=True)
    if counts.max() < len(spacings) * 0.95:
        raise ValueError('Non-uniform locations.')
    return float(unique_spacings[counts.argmax()])


def get_series_slice_spacing(series: Series) -> float:
    locations = get_series_slice_locations(series)
    return locations_to_spacing(sorted(locations))


def get_series_pixel_spacing(series: Series) -> Tuple[float, float]:
    pixel_spacings = np.stack([i.PixelSpacing for i in series])
    if (pixel_spacings.max(axis=0) - pixel_spacings.min(axis=0)).max() > 0.01:
        raise ValueError('Pixel spacing varies across slices.')
    return tuple(pixel_spacings[0])


def get_series_voxel_spacing(series: Series) -> Tuple[float, float, float]:
    pixel_spacing = get_series_pixel_spacing(series)
    slice_spacing = get_series_slice_spacing(series)
    return (*pixel_spacing, slice_spacing)


def get_instance_sop_uid(instance: Instance) -> str:
    return instance.SOPInstanceUID


def drop_duplicated_instances(series: Series) -> Series:
    series = sorted(series, key=get_instance_sop_uid)
    unique_instances = []
    for _, duplicates in itertools.groupby(series, key=get_instance_sop_uid):
        unique_instances.append(list(duplicates)[0])

    return unique_instances


def drop_duplicated_slices(series: Series, tolerance_hu=1) -> Series:
    series = drop_duplicated_instances(series)

    indices = list(range(len(series)))
    slice_locations = get_series_slice_locations(series)
    try:
        instance_numbers = [i.InstanceNumber for i in series]
        indices = sorted(indices, key=lambda i: (slice_locations[i], instance_numbers[i]))
    except AttributeError:
        indices = sorted(indices, key=lambda i: slice_locations[i])

    new_indices = []
    for _, duplicates in itertools.groupby(indices, key=lambda i: slice_locations[i]):
        duplicates = list(duplicates)
        image = get_instance_image(series[duplicates[0]])
        if not all(np.allclose(get_instance_image(series[i]), image, atol=tolerance_hu) for i in duplicates):
            raise ValueError('Slices have same locations, but different pixel arrays.')
        new_indices.append(duplicates[0])

    return [series[i] for i in sorted(new_indices)]


def get_instance_image(instance: Dataset):
    def _to_int(x):
        # this little trick helps to avoid unneeded type casting
        if x == int(x):
            x = int(x)
        return x

    image = instance.pixel_array

    slope, intercept = instance.get('RescaleSlope'), instance.get('RescaleIntercept')
    if slope is not None and slope != 1:
        image = image * _to_int(slope)
    if intercept is not None and intercept != 0:
        image = image + _to_int(intercept)

    return image


def get_series_image(series: Series) -> np.ndarray:
    return np.stack(list(map(get_instance_image, series)), axis=-1)
