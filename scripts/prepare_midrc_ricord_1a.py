from pathlib import Path
from omegaconf import DictConfig
import hydra
from tqdm.auto import tqdm
import pydicom
import mdai
import numpy as np
import pandas as pd
from skimage.draw import polygon
import warnings

from vox2vec.preprocessing.dicom import (
    get_series_uid, Plane, get_series_slice_plane, drop_duplicated_slices, order_series,
    get_series_image, get_series_voxel_spacing, get_series_orientation_matrix,
    to_canonical_orientation
)
from vox2vec.preprocessing.common import preprocess, Data
from vox2vec.utils.io import save_numpy, save_json


LABELS = [
    'Atelectasis',
    'Infectious cavity',
    'Infectious opacity',
    'Infectious TIB/micronodules',
    'Noninfectious nodule/mass',
    'Other noninfectious opacity'
]


@hydra.main(version_base=None, config_path='../configs', config_name='prepare_data')
def main(config: DictConfig):
    src_dirpath = Path(config.paths.midrc_ricord_1a_src_dirpath)

    anns: pd.DataFrame = mdai.common_utils.json_to_dataframe(
        src_dirpath / 'MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json'
    )['annotations']
    anns = anns[anns['scope'] == 'INSTANCE']  # drop study-level annotations, i.e. labels

    for series_uid, series_anns in tqdm(anns.groupby('SeriesInstanceUID'),
                                               desc='Preparing MIDRC-RICORD-1a'):
        # load series from DICOMs
        series = list(map(pydicom.dcmread, src_dirpath.glob(f'**/{series_uid}/*.dcm')))

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

            sop_instance_uids = [i.SOPInstanceUID for i in series]
        except (AttributeError, ValueError, NotImplementedError) as e:
            warnings.warn(f'Series {series_uid} fails with {e.__class__.__name__}: {str(e)}')
            continue
        
        # to canonical orientation
        image, voxel_spacing = to_canonical_orientation(image, voxel_spacing, om)

        # drop series with too large voxel spacing
        if any(voxel_spacing[i] > config.max_voxel_spacing[i] for i in range(3)):
            continue

        # create multiclass mask from annotations
        mask = np.zeros(image.shape, dtype='uint8')
        for label, label_anns in series_anns.groupby('labelName'):
            for _, ann in label_anns.iterrows():
                slice_index = sop_instance_uids.index(ann['SOPInstanceUID'])
                if ann['data'] is None:
                    warnings.warn(f'{label} annotations for series {series_uid} contains None for slice {slice_index}.')
                    continue
                vertices = np.array(ann['data']['vertices'])
                mask[(*polygon(vertices[:, 1], vertices[:, 0], image.shape[:2]), slice_index)] = LABELS.index(label) + 1

        # to canonical orientation
        mask, _ = to_canonical_orientation(mask, None, om)

        # preprocessing
        data = Data(image, voxel_spacing, mask)
        data = preprocess(data, config.preprocessing)
        image, voxel_spacing, mask, body_mask = data

        # drop images "without body"
        if not body_mask.any():
            continue

        # drop too small images
        if any(image.shape[i] < config.min_image_size[i] for i in range(3)):
            continue

        save_dirpath = Path(config.paths.midrc_ricord_1a_dirpath) / series_uid
        save_dirpath.mkdir(parents=True)
        save_numpy(image.astype('float16'), save_dirpath / 'image.npy.gz', compression=1, timestamp=0)
        save_json(voxel_spacing, save_dirpath / 'voxel_spacing.json')
        save_numpy(mask, save_dirpath / 'mask.npy.gz', compression=1, timestamp=0)
        save_numpy(body_mask, save_dirpath / 'body_mask.npy.gz', compression=1, timestamp=0)


if __name__ == "__main__":
    main()
