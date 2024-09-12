from dataclasses import dataclass
from typing import Tuple, Optional, Union
from pathlib import Path
import random
import numpy as np
from sklearn.model_selection import train_test_split
from imops import crop_to_box, zoom_to_shape as resize

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from vox2vec.utils.io import load_numpy, load_json
from vox2vec.utils.misc import get_random_sample
from vox2vec.utils.box import get_random_box, get_overlap_box
from vox2vec.typing import PreparedDataDirs, PretrainDataFractions
from .augmentations import ColorAugmentations, augment_color


@dataclass
class MoCoSpatialAugmentations:
    context_min_voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 2.0)
    context_max_voxel_spacing: Tuple[float, float, float] = (2.0, 2.0, 4.0)
    context_crop_size: Tuple[int, int, int] = (64, 64, 32)
    target_crop_size: Tuple[int, int, int] = (128, 128, 64)


class MoCoDataModule(pl.LightningDataModule):
    def __init__(
            self,
            prepared_data_dirs: PreparedDataDirs,
            nlst_val_size: int = 1000,
            pretrain_data_fractions: PretrainDataFractions = PretrainDataFractions(),
            spatial_augmentations: MoCoSpatialAugmentations = MoCoSpatialAugmentations(),
            color_augmentations: ColorAugmentations = ColorAugmentations(),
            num_voxels_per_crop: int = 512,
            batch_size: int = 8,  # num images per batch
            num_batches_per_epoch: int = 3000,
            num_workers: int = 0,
            prefetch_factor: Optional[int] = None,
            random_seed: int = 42,
    ) -> None:
        super().__init__()

        self.prepared_data_dirs = prepared_data_dirs
        self.nlst_val_size = nlst_val_size
        self.pretrain_data_fractions = pretrain_data_fractions
        self.spatial_augmentations = spatial_augmentations
        self.color_augmentations = color_augmentations
        self.num_voxels_per_crop = num_voxels_per_crop
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.random_seed = random_seed

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = 'fit') -> None:
        num_images_per_epoch = self.batch_size * self.num_batches_per_epoch
        self.train_dataset = _MoCoDataset(
            prepared_data_dirs=self.prepared_data_dirs,
            nlst_val_size=self.nlst_val_size,
            pretrain_data_fractions=self.pretrain_data_fractions,
            spatial_augmentations=self.spatial_augmentations,
            color_augmentations=self.color_augmentations,
            num_voxels_per_crop=self.num_voxels_per_crop,
            num_images_per_epoch=num_images_per_epoch,
            random_seed=self.random_seed
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor
        )

    def _collate_fn(self, batch):
        target_images, target_voxel_indices, context_images, context_voxel_indices = zip(*batch)

        return (
            torch.from_numpy(np.stack(target_images)),
            [torch.from_numpy(indices) for indices in target_voxel_indices],
            torch.from_numpy(np.stack(context_images)),
            [torch.from_numpy(indices) for indices in context_voxel_indices],
        )


class _MoCoDataset(Dataset):
    def __init__(
            self,
            prepared_data_dirs: PreparedDataDirs,
            nlst_val_size: int,
            pretrain_data_fractions: PretrainDataFractions,
            spatial_augmentations: MoCoSpatialAugmentations,
            color_augmentations: ColorAugmentations,
            num_voxels_per_crop: int,
            num_images_per_epoch: int,
            random_seed: int
    ) -> None:
        super().__init__()

        random.seed(random_seed)

        self.spatial_augmentations = spatial_augmentations
        self.color_augmentations = color_augmentations
        self.num_voxels_per_crop = num_voxels_per_crop
        self.num_images_per_epoch = num_images_per_epoch

        nlst_image_dirpaths, _ = train_test_split(list(Path(prepared_data_dirs.nlst).iterdir()),
                                                  test_size=nlst_val_size, random_state=random_seed)
        self.image_dirpaths = (
            get_random_sample(population=nlst_image_dirpaths,
                              size=pretrain_data_fractions.nlst)
            + get_random_sample(population=list(Path(prepared_data_dirs.amos_ct_labeled_train).iterdir()),
                                size=pretrain_data_fractions.amos_ct_labeled_train)
            + get_random_sample(population=list(Path(prepared_data_dirs.amos_ct_unlabeled_train).iterdir()),
                                size=pretrain_data_fractions.amos_ct_unlabeled_train)
            + get_random_sample(population=list(Path(prepared_data_dirs.abdomen_atlas).iterdir()),
                                size=pretrain_data_fractions.abdomen_atlas)
            + get_random_sample(population=list(Path(prepared_data_dirs.flare23_labeled_train).iterdir()),
                                size=pretrain_data_fractions.flare23_labeled_train)
            + get_random_sample(population=list(Path(prepared_data_dirs.flare23_unlabeled_train).iterdir()),
                                size=pretrain_data_fractions.flare23_unlabeled_train)
        )

    def __len__(self):
        return self.num_images_per_epoch

    def __getitem__(self, index: int):
        image_dirpath = random.choice(self.image_dirpaths)
        image = load_numpy(image_dirpath / 'image.npy.gz', decompress=True).astype('float32')
        voxel_spacing = load_json(image_dirpath / 'voxel_spacing.json')
        body_mask = load_numpy(image_dirpath / 'body_mask.npy.gz', decompress=True)

        return _get_targets_and_contexts(
            image=image,
            voxel_spacing=voxel_spacing,
            body_mask=body_mask,
            spatial_augmentations=self.spatial_augmentations,
            color_augmentations=self.color_augmentations,
            num_voxels_per_crop=self.num_voxels_per_crop,
        )


def _get_targets_and_contexts(
        image: np.ndarray,
        voxel_spacing: Tuple[float, float, float],
        body_mask: np.ndarray,
        spatial_augmentations: MoCoSpatialAugmentations,
        color_augmentations: ColorAugmentations,
        num_voxels_per_crop: int,
) -> Tuple:
    image_size = np.array(image.shape, dtype='int64')

    target_crop_size = np.array(spatial_augmentations.target_crop_size, dtype='int64')
    target_box = get_random_box(image_size, target_crop_size)
    target_image = crop_to_box(image, target_box)
    target_image = np.expand_dims(target_image, axis=0)

    context_crop_size = np.array(spatial_augmentations.context_crop_size, dtype='int64')
    context_min_voxel_spacing = np.array(spatial_augmentations.context_min_voxel_spacing, dtype='float32')
    context_max_voxel_spacing = np.array(spatial_augmentations.context_max_voxel_spacing, dtype='float32')
    context_max_voxel_spacing = np.minimum(context_max_voxel_spacing, voxel_spacing * image_size / context_crop_size)
    context_voxel_spacing = np.random.uniform(context_min_voxel_spacing, context_max_voxel_spacing)
    resize_factor = voxel_spacing / context_voxel_spacing
    context_crop_size_before_resize = np.int64(np.round(context_crop_size / resize_factor))

    voxel_index = np.random.randint(*target_box, size=(1, 3))
    context_box = get_random_box(image_size, context_crop_size_before_resize, voxel_index)
    overlap_box = get_overlap_box(context_box, target_box)
    voxel_indices = overlap_box[0] + np.argwhere(crop_to_box(body_mask, overlap_box))
    if len(voxel_indices) > num_voxels_per_crop:
        voxel_indices = voxel_indices[np.random.choice(len(voxel_indices), num_voxels_per_crop, replace=False)]

    target_voxel_indices = voxel_indices - target_box[0]

    context_image = crop_to_box(image, context_box)
    context_voxel_indices = voxel_indices - context_box[0]

    context_image = np.ascontiguousarray(context_image)
    context_image = resize(context_image, context_crop_size, backend='Scipy')
    context_voxel_indices = np.int64(np.floor(context_voxel_indices * resize_factor))

    # flips
    # for p, axis in zip(spatial_augmentations.context_flips_p, [-3, -2, -1], strict=True):
    #     if random.uniform(0, 1) < p:
    #         context_image = np.flip(context_image, axis)
    #         context_voxel_indices[:, axis] = context_image.shape[axis] - 1 - context_voxel_indices[:, axis]

    # fix numpy meta after flips/rots
    # context_image = context_image.copy()

    # augment colors
    context_image = augment_color(context_image, context_voxel_spacing, color_augmentations)

    # add channel dim
    context_image = np.expand_dims(context_image, axis=0)

    return target_image, target_voxel_indices, context_image, context_voxel_indices
