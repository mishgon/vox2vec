from dataclasses import dataclass
from typing import Tuple, Optional, Sequence
from pathlib import Path
import random
import numpy as np
from sklearn.model_selection import train_test_split
from imops import crop_to_box, zoom

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from vox2vec.utils.io import load_numpy, load_json
from vox2vec.utils.misc import get_random_sample
from vox2vec.utils.box import get_random_box, get_overlap_box
from vox2vec.typing import PreparedDataDirs, PretrainDataFractions
from .augmentations import ColorAugmentations, augment_color


@dataclass
class SimCLRSpatialAugmentations:
    min_voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.5)
    max_voxel_spacing: Tuple[float, float, float] = (2.0, 2.0, 3.0)
    crop_size: Tuple[int, int, int] = (96, 96, 64)


@dataclass
class SimCLRMasking:
    p: float = 0.0
    ratio: float = 0.6
    block_size: Tuple[int, int, int] = (24, 24, 16)


class SimCLRDataModule(pl.LightningDataModule):
    def __init__(
            self,
            prepared_data_dirs: PreparedDataDirs,
            nlst_val_size: int = 1000,
            pretrain_data_fractions: PretrainDataFractions = PretrainDataFractions(),
            spatial_augmentations: SimCLRSpatialAugmentations = SimCLRSpatialAugmentations(),
            color_augmentations: ColorAugmentations = ColorAugmentations(),
            masking: SimCLRMasking = SimCLRMasking(),
            num_voxels_per_crop: int = 1024,
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
        self.masking = masking
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
        self.train_dataset = _SimCLRDataset(
            prepared_data_dirs=self.prepared_data_dirs,
            nlst_val_size=self.nlst_val_size,
            pretrain_data_fractions=self.pretrain_data_fractions,
            spatial_augmentations=self.spatial_augmentations,
            color_augmentations=self.color_augmentations,
            masking=self.masking,
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
        (images_1, masks_1, voxel_indices_1, images_2, masks_2, voxel_indices_2) = zip(*batch)
        return (
            torch.from_numpy(np.stack(images_1)),
            torch.from_numpy(np.stack(masks_1)),
            [torch.from_numpy(indices) for indices in voxel_indices_1],
            torch.from_numpy(np.stack(images_2)),
            torch.from_numpy(np.stack(masks_2)),
            [torch.from_numpy(indices) for indices in voxel_indices_2],
        )


class _SimCLRDataset(Dataset):
    def __init__(
            self,
            prepared_data_dirs: PreparedDataDirs,
            nlst_val_size: int,
            pretrain_data_fractions: PretrainDataFractions,
            spatial_augmentations: SimCLRSpatialAugmentations,
            color_augmentations: ColorAugmentations,
            masking: SimCLRMasking,
            num_voxels_per_crop: int,
            num_images_per_epoch: int,
            random_seed: int
    ) -> None:
        super().__init__()

        random.seed(random_seed)

        self.spatial_augmentations = spatial_augmentations
        self.color_augmentations = color_augmentations
        self.masking = masking
        self.num_voxels_per_crop = num_voxels_per_crop
        self.num_images_per_epoch = num_images_per_epoch

        nlst_image_dirpaths, _ = train_test_split(sorted(Path(prepared_data_dirs.nlst).iterdir()),
                                                  test_size=nlst_val_size, random_state=random_seed)
        self.image_dirpaths = (
            get_random_sample(population=nlst_image_dirpaths,
                              size=pretrain_data_fractions.nlst)
            + get_random_sample(population=sorted(Path(prepared_data_dirs.amos_ct_labeled_train).iterdir()),
                                size=pretrain_data_fractions.amos_ct_labeled_train)
            + get_random_sample(population=sorted(Path(prepared_data_dirs.amos_ct_unlabeled_train).iterdir()),
                                size=pretrain_data_fractions.amos_ct_unlabeled_train)
            + get_random_sample(population=sorted(Path(prepared_data_dirs.abdomen_atlas).iterdir()),
                                size=pretrain_data_fractions.abdomen_atlas)
            # + get_random_sample(population=sorted(Path(prepared_data_dirs.flare23_labeled_train).iterdir()),
            #                     size=pretrain_data_fractions.flare23_labeled_train)
            # + get_random_sample(population=sorted(Path(prepared_data_dirs.flare23_unlabeled_train).iterdir()),
            #                     size=pretrain_data_fractions.flare23_unlabeled_train)
        )

    def __len__(self):
        return self.num_images_per_epoch

    def __getitem__(self, index: int):
        image_dirpath = random.choice(self.image_dirpaths)
        image = load_numpy(image_dirpath / 'image.npy.gz', decompress=True).astype('float32')
        voxel_spacing = load_json(image_dirpath / 'voxel_spacing.json')
        body_mask = load_numpy(image_dirpath / 'body_mask.npy.gz', decompress=True)

        return _get_augmented_crops(
            image=image,
            voxel_spacing=voxel_spacing,
            body_mask=body_mask,
            spatial_augmentations=self.spatial_augmentations,
            color_augmentations=self.color_augmentations,
            masking=self.masking,
            num_voxels_per_crop=self.num_voxels_per_crop,
        )


def _get_augmented_crops(
        image: np.ndarray,
        voxel_spacing: Tuple[float, float, float],
        body_mask: np.ndarray,
        spatial_augmentations: SimCLRSpatialAugmentations,
        color_augmentations: ColorAugmentations,
        masking: SimCLRMasking,
        num_voxels_per_crop: int,
) -> Tuple:
    image_size = np.array(image.shape, dtype='int64')
    crop_size = np.array(spatial_augmentations.crop_size, dtype='int64')
    min_voxel_spacing = np.array(spatial_augmentations.min_voxel_spacing, dtype='float32')
    max_voxel_spacing = np.array(spatial_augmentations.max_voxel_spacing, dtype='float32')
    max_voxel_spacing = np.minimum(max_voxel_spacing, voxel_spacing * image_size / crop_size)

    voxel_spacing_1 = np.random.uniform(min_voxel_spacing, max_voxel_spacing)
    voxel_spacing_2 = np.random.uniform(min_voxel_spacing, max_voxel_spacing)

    crop_size_before_resize_1 = np.int64(np.round(crop_size * voxel_spacing_1 / voxel_spacing))
    crop_size_before_resize_2 = np.int64(np.round(crop_size * voxel_spacing_2 / voxel_spacing))

    resize_factor_1 = crop_size / crop_size_before_resize_1
    resize_factor_2 = crop_size / crop_size_before_resize_2

    voxel_index = np.random.randint(0, image_size, size=(1, 3))
    crop_box_1 = get_random_box(image_size, crop_size_before_resize_1, voxel_index)
    crop_box_2 = get_random_box(image_size, crop_size_before_resize_2, voxel_index)
    overlap_box = get_overlap_box(crop_box_1, crop_box_2)
    voxel_indices = overlap_box[0] + np.argwhere(crop_to_box(body_mask, overlap_box))
    if len(voxel_indices) > num_voxels_per_crop:
        voxel_indices = voxel_indices[np.random.choice(len(voxel_indices), num_voxels_per_crop, replace=False)]

    image_1, masks_1, voxel_indices_1 = _get_augmented_crop(
        image, voxel_spacing, voxel_indices, crop_box_1, resize_factor_1, color_augmentations, masking
    )
    image_2, masks_2, voxel_indices_2 = _get_augmented_crop(
        image, voxel_spacing, voxel_indices, crop_box_2, resize_factor_2, color_augmentations, masking
    )
    return image_1, masks_1, voxel_indices_1, image_2, masks_2, voxel_indices_2


def _get_augmented_crop(
        image: np.ndarray,
        voxel_spacing: np.ndarray,
        voxel_indices: np.ndarray,
        crop_box: np.ndarray,
        resize_factor: np.ndarray,
        color_augmentations: ColorAugmentations,
        masking: SimCLRMasking,
) -> Tuple:
    image = crop_to_box(image, crop_box)
    voxel_indices = voxel_indices - crop_box[0]

    image = zoom(np.ascontiguousarray(image), resize_factor, backend='Scipy')
    voxel_indices = np.int64(np.floor(voxel_indices * resize_factor))
    voxel_spacing = voxel_spacing / resize_factor

    # flips
    # for p, axis in zip(spatial_augmentations.context_flips_p, [-3, -2, -1], strict=True):
    #     if random.uniform(0, 1) < p:
    #         context_image = np.flip(context_image, axis)
    #         context_voxel_indices[:, axis] = context_image.shape[axis] - 1 - context_voxel_indices[:, axis]

    # fix numpy meta after flips/rots
    # context_image = context_image.copy()

    # augment colors
    image = augment_color(image, voxel_spacing, color_augmentations)

    # sample mask
    mask = _get_random_mask(image.shape, masking)

    # add channel dim
    image = np.expand_dims(image, axis=0)

    return image, mask, voxel_indices


def _get_random_mask(size: Sequence[int], masking: SimCLRMasking) -> np.ndarray:
    if masking.ratio == 0.0 or random.uniform(0, 1) > masking.p:
        return np.ones(size, dtype='float32')

    size = np.array(size, dtype='int64')
    block_size = np.array(masking.block_size, dtype='int64')

    assert np.all(size % block_size == 0)

    mask = np.ones(size // block_size, dtype='float32')
    mask[np.unravel_index(np.random.permutation(mask.size)[:int(mask.size * masking.ratio)], mask.shape)] = 0.0
    assert (mask != 1.0).any()
    for axis, repeats in enumerate(block_size):
        mask = np.repeat(mask, repeats, axis)

    return mask
