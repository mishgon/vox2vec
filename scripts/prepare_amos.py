from typing import Literal
from pathlib import Path
from omegaconf import DictConfig
import hydra
import zipfile
import gzip
import nibabel

from vox2vec.preprocessing.nifty import affine_to_voxel_spacing, to_canonical_orientation
from vox2vec.preprocessing.common import preprocess, Data
from vox2vec.utils.misc import ProgressParallel
from vox2vec.utils.io import save_numpy, save_json


# ids that cause some runtime errors
DROP_IDS = (
    '5397', '5073', '5120', '5173', '5414', '5437', '5514', '5547', '5588',
    '5640', '5654', '5677', '5687', '5721', '5871', '5950', '6016'
)


def get_ids(config: DictConfig):
    labeled_train_ids, val_ids, test_ids = [], [], []
    with zipfile.ZipFile(Path(config.paths.source_data_dirs.amos) / 'amos22.zip') as zf:
        for name in zf.namelist():
            if not name.endswith('.nii.gz'):
                continue

            if name.startswith('amos22/imagesTr/amos_'):
                labeled_train_ids.append(name[len('amos22/imagesTr/amos_'):-len('.nii.gz')])
            elif name.startswith('amos22/imagesVa/amos_'):
                val_ids.append(name[len('amos22/imagesVa/amos_'):-len('.nii.gz')])
            elif name.startswith('amos22/imagesTs/amos_'):
                test_ids.append(name[len('amos22/imagesTs/amos_'):-len('.nii.gz')])

    # int(i) <= 500 - CT ids
    ct_labeled_train_ids = [i for i in labeled_train_ids if int(i) <= 500 and i not in DROP_IDS]
    ct_val_ids = [i for i in val_ids if int(i) <= 500 and i not in DROP_IDS]
    ct_test_ids = [i for i in test_ids if int(i) <= 500 and i not in DROP_IDS]

    ct_unlabeled_train_ids = []
    for filename in [
            'amos22_unlabeled_ct_5000_5399.zip',
            'amos22_unlabeled_ct_5400_5899.zip',
            'amos22_unlabeled_ct_5900_6199.zip',
            'amos22_unlabeled_ct_6200_6899.zip',
    ]:
        with zipfile.ZipFile(Path(config.paths.source_data_dirs.amos) / filename) as zf:
            for name in zf.namelist():
                if not name.endswith('.nii.gz'):
                    continue

                ct_unlabeled_train_ids.append(name.split('/')[1][len('amos_'):-len('.nii.gz')])

    ct_unlabeled_train_ids = [i for i in ct_unlabeled_train_ids if i not in DROP_IDS]

    return ct_labeled_train_ids, ct_val_ids, ct_test_ids, ct_unlabeled_train_ids


def prepare_id(i: str, config: DictConfig, subset: Literal['labeled_train', 'val', 'unlabeled_train']) -> None:
    match subset:
        case 'labeled_train':
            zip_filename = 'amos22.zip'
            image_filepath = f'amos22/imagesTr/amos_{i}.nii.gz'
            mask_filepath = f'amos22/labelsTr/amos_{i}.nii.gz'
            save_dirpath = Path(config.paths.prepared_data_dirs.amos_ct_labeled_train) / i
        case 'val':
            zip_filename = 'amos22.zip'
            image_filepath = f'amos22/imagesVa/amos_{i}.nii.gz'
            mask_filepath = f'amos22/labelsVa/amos_{i}.nii.gz'
            save_dirpath = Path(config.paths.prepared_data_dirs.amos_ct_val) / i
        case 'unlabeled_train':
            if 5000 <= int(i) < 5400:
                zip_filename = 'amos22_unlabeled_ct_5000_5399.zip'
                image_filepath = f'amos_unlabeled_ct_5000_5399/amos_{i}.nii.gz'
            elif 5400 <= int(i) < 5900:
                zip_filename = 'amos22_unlabeled_ct_5400_5899.zip'
                image_filepath = f'amos_unlabeled_ct_5400_5899/amos_{i}.nii.gz'
            elif 5900 <= int(i) < 6200:
                zip_filename = 'amos22_unlabeled_ct_5900_6199.zip'
                image_filepath = f'amos22_unlabeled_ct_5900_6199/amos_{i}.nii.gz'
            else:
                zip_filename = 'amos22_unlabeled_ct_6200_6899.zip'
                image_filepath = f'amos22_unlabeled_6200_6899/amos_{i}.nii.gz'
            mask_filepath = None
            save_dirpath = Path(config.paths.prepared_data_dirs.amos_ct_unlabeled_train) / i

    # read image and affine
    with zipfile.Path(Path(config.paths.source_data_dirs.amos) / zip_filename, image_filepath).open('rb') as gz_file:
        with gzip.GzipFile(fileobj=gz_file) as nii_file:
            fh = nibabel.FileHolder(fileobj=nii_file)
            image_nii = nibabel.Nifti1Image.from_file_map({'header': fh, 'image': fh})
            image = image_nii.get_fdata()
            affine = image_nii.affine

    # read mask
    if mask_filepath is not None:
        with zipfile.Path(Path(config.paths.source_data_dirs.amos) / zip_filename, mask_filepath).open('rb') as gz_file:
            with gzip.GzipFile(fileobj=gz_file) as nii_file:
                fh = nibabel.FileHolder(fileobj=nii_file)
                mask_nii = nibabel.Nifti1Image.from_file_map({'header': fh, 'image': fh})
                mask = mask_nii.get_fdata().astype('uint8')
                mask_affine = mask_nii.affine
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
    # parse ids from the archive files
    ct_labeled_train_ids, ct_val_ids, ct_test_ids, ct_unlabeled_train_ids = get_ids(config)

    desc = 'Preparing AMOS CT labeled train subset'
    ProgressParallel(n_jobs=config.num_workers, backend='loky', total=len(ct_labeled_train_ids), desc=desc)(
        (prepare_id, [i, config, 'labeled_train'], {}) for i in ct_labeled_train_ids
    )

    desc = 'Preparing AMOS CT val subset'
    ProgressParallel(n_jobs=config.num_workers, backend='loky', total=len(ct_val_ids), desc=desc)(
        (prepare_id, [i, config, 'val'], {}) for i in ct_val_ids
    )

    desc = 'Preparing AMOS CT unlabeled train subset'
    ProgressParallel(n_jobs=config.num_workers, backend='loky', total=len(ct_unlabeled_train_ids), desc=desc)(
        (prepare_id, [i, config, 'unlabeled_train'], {}) for i in ct_unlabeled_train_ids
    )


if __name__ == '__main__':
    main()
