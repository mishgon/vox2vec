from pathlib import Path
from omegaconf import DictConfig
import hydra
import gzip
import nibabel

from vox2vec.preprocessing.nifty import affine_to_voxel_spacing, to_canonical_orientation
from vox2vec.preprocessing.common import preprocess, Data
from vox2vec.utils.io import save_numpy, save_json
from vox2vec.utils.misc import ProgressParallel, is_diagonal


def prepare_id(i: str, config: DictConfig) -> None:
    src_dirpath = Path(config.paths.source_data_dirs.kits21)
    image_filepath, = src_dirpath.glob(f'kits21/data/{i}/imaging.nii.gz')
    try:
        mask_filepath, = src_dirpath.glob(f'kits21/data/{i}/aggregated_MAJ_seg.nii.gz')
    except ValueError:
        return

    with gzip.GzipFile(filename=image_filepath) as nii_file:
        fh = nibabel.FileHolder(fileobj=nii_file)
        image_nii = nibabel.Nifti1Image.from_file_map({'header': fh, 'image': fh})
        image_nii = nibabel.as_closest_canonical(image_nii)
        affine = image_nii.affine
        if not is_diagonal(affine[:3, :3]):
            return
        image = image_nii.get_fdata()

    with gzip.GzipFile(filename=mask_filepath) as nii_file:
        fh = nibabel.FileHolder(fileobj=nii_file)
        mask_nii = nibabel.Nifti1Image.from_file_map({'header': fh, 'image': fh})
        mask_nii = nibabel.as_closest_canonical(mask_nii)
        mask_affine = mask_nii.affine
        if not is_diagonal(mask_affine[:3, :3]):
            return
        mask = mask_nii.get_fdata().astype('uint8')

    # compute voxel spacing
    voxel_spacing = affine_to_voxel_spacing(affine)

    # to canonical orientation
    image, voxel_spacing = to_canonical_orientation(image, voxel_spacing, affine)
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

    save_dirpath = Path(config.paths.prepared_data_dirs.kits21) / i
    save_dirpath.mkdir(parents=True)
    save_numpy(image.astype('float16'), save_dirpath / 'image.npy.gz', compression=1, timestamp=0)
    save_json(voxel_spacing, save_dirpath / 'voxel_spacing.json')
    if mask is not None:
        save_numpy(mask, save_dirpath / 'mask.npy.gz', compression=1, timestamp=0)
    save_numpy(body_mask, save_dirpath / 'body_mask.npy.gz', compression=1, timestamp=0)


@hydra.main(version_base=None, config_path='../configs', config_name='prepare_data')
def main(config: DictConfig):
    ids = [p.name for p in Path(config.paths.source_data_dirs.kits21).glob('kits21/data/*') if p.is_dir()]

    ProgressParallel(n_jobs=config.num_workers, backend='loky', total=len(ids), desc='Preparing KiTS21')(
        (prepare_id, [i, config], {}) for i in ids
    )


if __name__ == "__main__":
    main()
