from typing import List, Iterable
from pathlib import Path
from omegaconf import DictConfig
import hydra
import warnings
import pydicom

from vox2vec.preprocessing.dicom import (
    get_series_uid, Plane, get_series_slice_plane, drop_duplicated_slices, order_series,
    get_series_image, get_series_voxel_spacing, get_series_orientation_matrix,
    to_canonical_orientation
)
from vox2vec.preprocessing.common import preprocess, Data
from vox2vec.utils.misc import ProgressParallel
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
    return list(map(pydicom.dcmread, series_dirpath.glob('*.dcm')))


def prepare_patient(patient_dirpath: Path, config: DictConfig) -> None:
    series_dirpath = max(iterate_series_dirpaths(patient_dirpath), key=estimate_series_length)

    # drop too short series
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
    except (AttributeError, ValueError, NotImplementedError) as e:
        warnings.warn(f'Series at {str(series_dirpath)} fails with {e.__class__.__name__}: {str(e)}')
        return

    # to canonical orientation
    image, voxel_spacing = to_canonical_orientation(image, voxel_spacing, om)

    # drop series with too large voxel spacing
    if any(voxel_spacing[i] > config.max_voxel_spacing[i] for i in range(3)):
        return

    # preprocessing
    data = Data(image, voxel_spacing)
    data = preprocess(data, config.preprocessing)
    image, voxel_spacing, _, body_mask = data

    # drop images "without body"
    if not body_mask.any():
        return

    # drop too small images
    if any(image.shape[i] < config.min_image_size[i] for i in range(3)):
        return

    save_dirpath = Path(config.paths.prepared_data_dirs.nlst) / series_uid
    save_dirpath.mkdir(parents=True)
    save_numpy(image.astype('float16'), save_dirpath / 'image.npy.gz', compression=1, timestamp=0)
    save_json(voxel_spacing, save_dirpath / 'voxel_spacing.json')
    save_numpy(body_mask, save_dirpath / 'body_mask.npy.gz', compression=1, timestamp=0)


@hydra.main(version_base=None, config_path='../configs', config_name='prepare_data')
def main(config: DictConfig): 
    patient_dirpaths = list(Path(config.paths.source_data_dirs.nlst).glob('NLST/*'))

    ProgressParallel(n_jobs=config.num_workers, backend='loky', total=len(patient_dirpaths), desc='Preparing NLST')(
        (prepare_patient, [patient_dirpath, config], {}) for patient_dirpath in patient_dirpaths
    )


if __name__ == '__main__':
    main()
