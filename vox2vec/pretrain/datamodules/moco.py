from dataclasses import dataclass
from typing import Tuple, Optional, Union, Sequence
from pathlib import Path
import random
import numpy as np
from imops import crop_to_box, zoom_to_shape as resize

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from vox2vec.utils.io import load_numpy, load_json
from vox2vec.utils.misc import get_random_sample
from vox2vec.utils.box import get_random_box, get_overlap_box
from .augmentations import ColorAugmentationsConfig, augment_color


@dataclass
class MoCoDataPaths:
    nlst_dirpath: str
    amos_ct_labeled_train_dirpath: str
    amos_ct_unlabeled_train_dirpath: str
    abdomen_atlas_dirpath: str
    flare23_labeled_train_dirpath: str
    flare23_unlabeled_train_dirpath: str


@dataclass
class MoCoDatasetConfig:
    nlst_size: Union[float, int] = 1.0
    amos_ct_labeled_train_size: Union[float, int] = 1.0
    amos_ct_unlabeled_train_size: Union[float, int] = 1.0
    abdomen_atlas_size: Union[float, int] = 1.0
    flare23_labeled_train_size: Union[float, int] = 1.0
    flare23_unlabeled_train_size: Union[float, int] = 1.0


@dataclass
class MoCoSpatialAugmentationsConfig:
    context_min_voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 2.0)
    context_max_voxel_spacing: Tuple[float, float, float] = (2.0, 2.0, 4.0)
    context_crop_size: Tuple[int, int, int] = (128, 128, 64)
    target_crop_size: Tuple[int, int, int] = (256, 256, 128)


@dataclass
class MoCoMaskingConfig:
    p: float = 0.0
    ratio: float = 0.6
    block_size: Optional[Tuple[int, int, int]] = (16, 16, 8)


class MoCoDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_paths: MoCoDataPaths,
            dataset_config: MoCoDatasetConfig = MoCoDatasetConfig(),
            spatial_augmentations_config: MoCoSpatialAugmentationsConfig = MoCoSpatialAugmentationsConfig(),
            color_augmentations_config: ColorAugmentationsConfig = ColorAugmentationsConfig(),
            masking_config: MoCoMaskingConfig = MoCoMaskingConfig(),
            num_voxels_per_crop: int = 512,
            batch_size: int = 8,  # images per batch
            num_batches_per_epoch: int = 3000,
            num_workers: int = 0,
            prefetch_factor: Optional[int] = None,
            random_seed: int = 42,
    ) -> None:
        super().__init__()

        self.data_paths = data_paths
        self.dataset_config = dataset_config
        self.spatial_augmentations_config = spatial_augmentations_config
        self.color_augmentations_config = color_augmentations_config
        self.masking_config = masking_config
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
            data_paths=self.data_paths,
            dataset_config=self.dataset_config,
            spatial_augmentations_config=self.spatial_augmentations_config,
            color_augmentations_config=self.color_augmentations_config,
            masking_config=self.masking_config,
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
            prefetch_factor=self.prefetch_factor
        )

    def _collate_fn(self, batch):
        (target_images, target_voxel_indices,
         context_images, context_masks, context_voxel_indices) = zip(*batch)

        return (
            torch.from_numpy(np.stack(target_images)),
            [torch.from_numpy(indices) for indices in target_voxel_indices],
            torch.from_numpy(np.stack(context_images)),
            torch.from_numpy(np.stack(context_masks)),
            [torch.from_numpy(indices) for indices in context_voxel_indices],
        )


class _MoCoDataset(Dataset):
    def __init__(
            self,
            data_paths: MoCoDataPaths,
            dataset_config: MoCoDatasetConfig,
            spatial_augmentations_config: MoCoSpatialAugmentationsConfig,
            color_augmentations_config: ColorAugmentationsConfig,
            masking_config: MoCoMaskingConfig,
            num_voxels_per_crop: int,
            num_images_per_epoch: int,
            random_seed: int
    ) -> None:
        super().__init__()

        self.spatial_augmentations_config = spatial_augmentations_config
        self.color_augmentations_config = color_augmentations_config
        self.masking_config = masking_config
        self.num_voxels_per_crop = num_voxels_per_crop
        self.num_images_per_epoch = num_images_per_epoch

        random.seed(random_seed)

        self.image_dirpaths = (
            get_random_sample(population=list(Path(data_paths.nlst_dirpath).iterdir()),
                              size=dataset_config.nlst_size)
            + get_random_sample(population=list(Path(data_paths.amos_ct_labeled_train_dirpath).iterdir()),
                                size=dataset_config.amos_ct_labeled_train_size)
            + get_random_sample(population=list(Path(data_paths.amos_ct_unlabeled_train_dirpath).iterdir()),
                                size=dataset_config.amos_ct_unlabeled_train_size)
            + get_random_sample(population=list(Path(data_paths.abdomen_atlas_dirpath).iterdir()),
                                size=dataset_config.abdomen_atlas_size)
            + get_random_sample(population=list(Path(data_paths.flare23_labeled_train_dirpath).iterdir()),
                                size=dataset_config.flare23_labeled_train_size)
            + get_random_sample(population=list(Path(data_paths.flare23_unlabeled_train_dirpath).iterdir()),
                                size=dataset_config.flare23_unlabeled_train_size)
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
            spatial_augmentations_config=self.spatial_augmentations_config,
            color_augmentations_config=self.color_augmentations_config,
            masking_config=self.masking_config,
            num_voxels_per_crop=self.num_voxels_per_crop,
        )


def _get_targets_and_contexts(
        image: np.ndarray,
        voxel_spacing: Tuple[float, float, float],
        body_mask: np.ndarray,
        spatial_augmentations_config: MoCoSpatialAugmentationsConfig,
        color_augmentations_config: ColorAugmentationsConfig,
        masking_config: MoCoMaskingConfig,
        num_voxels_per_crop: int,
) -> Tuple:
    image_size = np.array(image.shape, dtype='int64')

    target_crop_size = np.array(spatial_augmentations_config.target_crop_size, dtype='int64')
    target_box = get_random_box(image_size, target_crop_size)
    target_image = crop_to_box(image, target_box)
    target_image = np.expand_dims(target_image, axis=0)

    context_crop_size = np.array(spatial_augmentations_config.context_crop_size, dtype='int64')
    context_min_voxel_spacing = np.array(spatial_augmentations_config.context_min_voxel_spacing, dtype='float32')
    context_max_voxel_spacing = np.array(spatial_augmentations_config.context_max_voxel_spacing, dtype='float32')
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
    # for p, axis in zip(spatial_augmentations_config.context_flips_p, [-3, -2, -1], strict=True):
    #     if random.uniform(0, 1) < p:
    #         context_image = np.flip(context_image, axis)
    #         context_voxel_indices[:, axis] = context_image.shape[axis] - 1 - context_voxel_indices[:, axis]

    # fix numpy meta after flips/rots
    # context_image = context_image.copy()

    # augment colors
    context_image = augment_color(context_image, context_voxel_spacing, color_augmentations_config)

    # sample mask
    context_mask = _get_context_mask(context_crop_size, masking_config)

    # add channel dim
    context_image = np.expand_dims(context_image, axis=0)

    return target_image, target_voxel_indices, context_image, context_mask, context_voxel_indices


def _get_context_mask(context_crop_size: np.ndarray, masking_config: MoCoMaskingConfig) -> np.ndarray:
    if masking_config.ratio == 0.0 or random.uniform(0, 1) > masking_config.p:
        return np.ones(context_crop_size, dtype='float32')
    
    block_size = np.array(masking_config.block_size, dtype='int64')

    assert np.all(context_crop_size % block_size == 0)

    mask = np.ones(context_crop_size // block_size, dtype='float32')
    mask[np.unravel_index(np.random.permutation(mask.size)[:int(mask.size * masking_config.ratio)], mask.shape)] = 0.0
    assert (mask != 1.0).any()
    for axis, repeats in enumerate(block_size):
        mask = np.repeat(mask, repeats, axis)

    return mask
