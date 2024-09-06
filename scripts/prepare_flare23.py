from typing import Literal
from pathlib import Path
from omegaconf import DictConfig
import hydra
import zipfile
import gzip
import nibabel
import numpy as np

from vox2vec.preprocessing.nifty import affine_to_voxel_spacing, to_canonical_orientation
from vox2vec.preprocessing.common import preprocess, Data
from vox2vec.utils.misc import ProgressParallel, is_diagonal
from vox2vec.utils.io import save_numpy, save_json


def get_ids(config: DictConfig):
    src_dirpath = Path(config.paths.source_data_dirs.flare23)
    labeled_train_ids = [
        filepath.name[:-len('_0000.nii.gz')]
        for filepath in src_dirpath.glob('imagesTr2200/*_0000.nii.gz')
    ]
    unlabeled_train_ids = [
        filepath.name[:-len('_0000.nii.gz')]
        for filepath in src_dirpath.glob('unlabeledTr1800/*_0000.nii.gz')
    ]
    labeled_val_ids = [
        filepath.name[:-len('.nii.gz')]
        for filepath in src_dirpath.glob('val-gt-50cases-for-sanity-check/*.nii.gz')
    ]
    return labeled_train_ids, unlabeled_train_ids, labeled_val_ids


def prepare_id(i: str, config: DictConfig, subset: Literal['labeled_train', 'unlabeled_train', 'labeled_val']) -> None:
    src_dirpath = Path(config.paths.source_data_dirs.flare23)
    match subset:
        case 'labeled_train':
            image_filepath = src_dirpath / f'imagesTr2200/{i}_0000.nii.gz'
            mask_filepath = src_dirpath / f'labelsTr2200/20230507-fix/{i}.nii.gz'
            if not mask_filepath.exists():
                mask_filepath = src_dirpath / f'labelsTr2200/{i}.nii.gz'
            save_dirpath = Path(config.paths.prepared_data_dirs.flare23_labeled_train) / i
        case 'unlabeled_train':
            image_filepath = src_dirpath / f'unlabeledTr1800/{i}_0000.nii.gz'
            mask_filepath = None
            save_dirpath = Path(config.paths.prepared_data_dirs.flare23_unlabeled_train) / i
        case 'labeled_val':
            image_filepath = src_dirpath / f'validation/{i}_0000.nii.gz'
            mask_filepath = src_dirpath / f'val-gt-50cases-for-sanity-check/{i}.nii.gz'
            save_dirpath = Path(config.paths.prepared_data_dirs.flare23_labeled_val) / i

    # load image and affine (drop if non-diagonal)
    image_nii = nibabel.load(image_filepath)
    image_nii = nibabel.as_closest_canonical(image_nii)
    affine = image_nii.affine
    if not is_diagonal(affine[:3, :3]):
        return
    image = image_nii.get_fdata()

    # load mask and mask affine (drop if non-diagonal)
    if mask_filepath is not None:
        mask_nii = nibabel.load(mask_filepath)
        mask_nii = nibabel.as_closest_canonical(mask_nii)
        mask_affine = mask_nii.affine
        if not is_diagonal(mask_affine[:3, :3]):
            return
        mask = mask_nii.get_fdata().astype('uint8')
    else:
        mask = None

    # compute voxel spacing
    voxel_spacing = affine_to_voxel_spacing(affine)

    # to canonical orientation
    image, voxel_spacing = to_canonical_orientation(image, voxel_spacing, affine)
    if mask is not None:
        mask, _ = to_canonical_orientation(mask, None, mask_affine)

    # drop series with too large voxel spacing
    if any(voxel_spacing[i] > config.max_voxel_spacing[i] for i in range(3)):
        return

    # preprocess
    data = Data(image, voxel_spacing, mask)
    data = preprocess(data, config.preprocessing)
    image, voxel_spacing, mask, body_mask = data

    # drop images "without body"
    if not body_mask.any():
        return

    # drop too small images
    if any(image.shape[i] < config.min_image_size[i] for i in range(3)):
        return

    save_dirpath.mkdir(parents=True)
    save_numpy(image.astype('float16'), save_dirpath / 'image.npy.gz', compression=1, timestamp=0)
    save_json(voxel_spacing, save_dirpath / 'voxel_spacing.json')
    if mask is not None:
        save_numpy(mask, save_dirpath / 'mask.npy.gz', compression=1, timestamp=0)
    save_numpy(body_mask, save_dirpath / 'body_mask.npy.gz', compression=1, timestamp=0)


@hydra.main(version_base=None, config_path='../configs', config_name='prepare_data')
def main(config: DictConfig):
    labeled_train_ids, unlabeled_train_ids, labeled_val_ids = get_ids(config)

    desc = 'Preparing FLARE23 labeled train subset'
    ProgressParallel(n_jobs=config.num_workers, backend='loky', total=len(labeled_train_ids), desc=desc)(
        (prepare_id, [i, config, 'labeled_train'], {}) for i in labeled_train_ids
    )

    desc = 'Preparing FLARE23 unlabeled train subset'
    ProgressParallel(n_jobs=config.num_workers, backend='loky', total=len(unlabeled_train_ids), desc=desc)(
        (prepare_id, [i, config, 'unlabeled_train'], {}) for i in unlabeled_train_ids
    )

    desc = 'Preparing FLARE23 labeled val subset'
    ProgressParallel(n_jobs=config.num_workers, backend='loky', total=len(labeled_val_ids), desc=desc)(
        (prepare_id, [i, config, 'labeled_val'], {}) for i in labeled_val_ids
    )


if __name__ == '__main__':
    main()
