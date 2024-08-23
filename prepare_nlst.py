from typing import List, Iterable
from pathlib import Path
from omegaconf import DictConfig
import hydra
import warnings
import pydicom
import numpy as np
from imops import crop_to_box, zoom
from skimage.exposure import equalize_adapthist

from vox2vec.utils.dicom import (
    get_series_uid, Plane, get_series_slice_plane, drop_duplicated_slices, order_series,
    get_series_image, get_series_voxel_spacing, get_series_orientation_matrix,
    to_canonical_orientation
)
from vox2vec.processing import get_body_box, get_body_mask, rescale_hu_piecewise
from vox2vec.utils.misc import mask_to_bbox, ProgressParallel
from vox2vec.utils.io import save_numpy, save_json


def iterate_series_dirpaths(patient_dirpath: Path) -> Iterable[Path]:
    for study_dirpath in patient_dirpath.iterdir():
        study_dirpath, = study_dirpath.iterdir()
        for path in study_dirpath.iterdir():
            if path.is_dir():
                yield path


def estimate_series_length(series_dirpath: Path) -> int:
    return len(list(series_dirpath.glob('*.dcm')))


def load_series(series_dirpath: Path) -> List[pydicom.FileDataset]:
    return [pydicom.dcmread(filepath) for filepath in series_dirpath.glob('*.dcm')]


def prepare_patient(patient_dirpath: Path, config: DictConfig) -> None:
    series_dirpath = max(iterate_series_dirpaths(patient_dirpath), key=estimate_series_length)

    # drop series if it is still too short
    if estimate_series_length(series_dirpath) < config.min_series_length:
        return

    # read series
    series = load_series(series_dirpath)

    # extract image, voxel spacing and orientation matrix from dicoms
    # drop non-axial series and series with invalid tags
    try:
        if get_series_slice_plane(series) != Plane.Axial:
            raise ValueError('Series is not axial')

        series = drop_duplicated_slices(series)
        series = order_series(series)

        series_uid = get_series_uid(series)
        image = get_series_image(series)
        voxel_spacing = get_series_voxel_spacing(series)
        om = get_series_orientation_matrix(series)
    except (AttributeError, ValueError) as e:
        warnings.warn(f'Series at {str(series_dirpath)} fails with {e.__class__.__name__}: {str(e)}')
        return

    # to canonical orientation
    image, voxel_spacing, om = to_canonical_orientation(image, voxel_spacing, om)

    # crop to body
    box = get_body_box(image, voxel_spacing)
    image = crop_to_box(image, box, num_threads=-1, backend='Scipy')

    # zoom to config.voxel_spacing
    image = image.astype('float32')
    scale_factor = tuple(voxel_spacing[i] / config.voxel_spacing[i] for i in range(3))
    image = zoom(image, scale_factor, fill_value=np.min, num_threads=-1, backend='Scipy')
    voxel_spacing = tuple(config.voxel_spacing)

    # zoom may pad image with zeros
    box = mask_to_bbox(image > image.min())
    image = crop_to_box(image, box, num_threads=-1, backend='Scipy')

    # get body binary mask
    body_mask = get_body_mask(image)

    # rescale Hounsfield Units (HU) using piecewise-linear func
    image = rescale_hu_piecewise(image, config.hu_pivots, config.rescaled_pivots)

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    if config.clahe:
        image = equalize_adapthist(image, clip_limit=config.clahe_clip_limit)

    save_dirpath = Path(config.paths.prep_nlst_dirpath) / series_uid
    save_dirpath.mkdir()
    save_numpy(image.astype('float16'), save_dirpath / 'image.npy.gz', compression=1, timestamp=0)
    save_json(voxel_spacing, save_dirpath / 'voxel_spacing.json')
    save_numpy(body_mask.astype('bool'), save_dirpath / 'body_mask.npy.gz', compression=1, timestamp=0)


@hydra.main(version_base=None, config_path='configs', config_name='prepare_data')
def main(config: DictConfig):
    nlst_dirpath = Path(config.paths.nlst_dirpath)
    patient_dirpaths = list(nlst_dirpath.glob('NLST/*'))
    prep_nlst_dirpath = Path(config.paths.prep_nlst_dirpath)
    prep_nlst_dirpath.mkdir(parents=True, exist_ok=True)

    ProgressParallel(n_jobs=config.num_workers, backend='loky', total=len(patient_dirpaths))(
        (prepare_patient, [patient_dirpath, config], {}) for patient_dirpath in patient_dirpaths
    )


if __name__ == '__main__':
    main()
