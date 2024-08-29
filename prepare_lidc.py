from pathlib import Path
from omegaconf import DictConfig
import hydra
import warnings
import numpy as np

import pylidc as pl
from pylidc.utils import consensus

from vox2vec.preprocessing.dicom import (
    get_series_uid, Plane, get_series_slice_plane, drop_duplicated_slices, order_series,
    get_series_image, get_series_voxel_spacing, get_series_orientation_matrix,
    to_canonical_orientation
)
from vox2vec.preprocessing.common import preprocess, Data, get_body_mask
from vox2vec.utils.misc import ProgressParallel
from vox2vec.utils.io import save_numpy, save_json


def prepare_scan(scan: pl.Scan, config: DictConfig):
    # read series
    series = scan.load_all_dicom_images(verbose=False)

    # drop too short series
    if len(series) < config.min_series_length:
        return

    # extract image, voxel spacing and orientation matrix from dicoms
    # drop non-axial series and series with invalid tags
    series_uid = get_series_uid(series)
    try:
        if get_series_slice_plane(series) != Plane.Axial:
            raise ValueError('Series is not axial')

        series = drop_duplicated_slices(series)
        series = order_series(series)

        image = get_series_image(series)
        voxel_spacing = get_series_voxel_spacing(series)
        om = get_series_orientation_matrix(series)
    except (AttributeError, ValueError, NotImplementedError) as e:
        warnings.warn(f'Series {series_uid} fails with {e.__class__.__name__}: {str(e)}')
        return

    # create mask using pylidc
    mask = np.zeros(image.shape, dtype=bool)
    for anns in scan.cluster_annotations():
        cmask, cbbox, _ = consensus(anns)
        mask[cbbox] = cmask
    # pylidc stacks slices in the other order than us
    mask = np.flip(mask, -1)

    # to canonical orientation
    image, voxel_spacing = to_canonical_orientation(image, voxel_spacing, om)
    mask, _ = to_canonical_orientation(mask, None, om)

    # preprocessing
    data = Data(image, voxel_spacing, mask)
    data = preprocess(data, config.preprocessing)
    image, voxel_spacing, mask, body_mask = data

    # drop images "without body"
    if not body_mask.any():
        return

    # drop too small images
    if any(image.shape[i] < config.min_image_size[i] for i in range(3)):
        return

    save_dirpath = Path(config.paths.prep_lidc_dirpath) / series_uid
    save_dirpath.mkdir(parents=True)
    save_numpy(image.astype('float16'), save_dirpath / 'image.npy.gz', compression=1, timestamp=0)
    save_json(voxel_spacing, save_dirpath / 'voxel_spacing.json')
    save_numpy(mask, save_dirpath / 'mask.npy.gz', compression=1, timestamp=0)
    save_numpy(body_mask, save_dirpath / 'body_mask.npy.gz', compression=1, timestamp=0)


@hydra.main(version_base=None, config_path='configs', config_name='prepare_data')
def main(config: DictConfig):
    scans = pl.query(pl.Scan).all()

    ProgressParallel(n_jobs=config.num_workers, backend='threading', total=len(scans), desc='Preparing LIDC')(
        (prepare_scan, [scan, config], {}) for scan in scans
    )


if __name__ == '__main__':
    main()
