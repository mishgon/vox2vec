from dataclasses import dataclass
from typing import Tuple, Optional, Union
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
from .augmentations import ColorAugmentations, augment_color


@dataclass
class SimCLRDataPaths:
    nlst_dirpath: str
    amos_ct_labeled_train_dirpath: str
    amos_ct_unlabeled_train_dirpath: str
    abdomen_atlas_dirpath: str
    flare23_labeled_train_dirpath: str
    flare23_unlabeled_train_dirpath: str


@dataclass
class SimCLRDatasets:
    nlst: Union[float, int] = 1.0
    amos_ct_labeled_train: Union[float, int] = 1.0
    amos_ct_unlabeled_train: Union[float, int] = 1.0
    abdomen_atlas: Union[float, int] = 1.0
    flare23_labeled_train: Union[float, int] = 1.0
    flare23_unlabeled_train: Union[float, int] = 1.0


@dataclass
class SimCLRSpatialAugmentations:
    min_voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 2.0)
    max_voxel_spacing: Tuple[float, float, float] = (2.0, 2.0, 4.0)
    crop_size: Tuple[int, int, int] = (128, 128, 64)


class SimCLRDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_paths: SimCLRDataPaths,
            datasets: SimCLRDatasets = SimCLRDatasets(),
            spatial_augmentations: SimCLRSpatialAugmentations = SimCLRSpatialAugmentations(),
            color_augmentations: ColorAugmentations = ColorAugmentations(),
            num_voxels_per_crop: int = 512,
            batch_size: int = 8,  # images per batch
            num_batches_per_epoch: int = 3000,
            num_workers: int = 0,
            prefetch_factor: Optional[int] = None,
            nlst_val_size: int = 1000,
            random_seed: int = 42,
    ) -> None:
        super().__init__()

        self.data_paths = data_paths
        self.datasets = datasets
        self.spatial_augmentations = spatial_augmentations
        self.color_augmentations = color_augmentations
        self.num_voxels_per_crop = num_voxels_per_crop
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.nlst_val_size = nlst_val_size
        self.random_seed = random_seed

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = 'fit') -> None:
        num_images_per_epoch = self.batch_size * self.num_batches_per_epoch
        self.train_dataset = _SimCLRDataset(
            data_paths=self.data_paths,
            datasets=self.datasets,
            spatial_augmentations=self.spatial_augmentations,
            color_augmentations=self.color_augmentations,
            num_voxels_per_crop=self.num_voxels_per_crop,
            num_images_per_epoch=num_images_per_epoch,
            nlst_val_size=self.nlst_val_size,
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
        (images_1, voxel_indices_1, images_2, voxel_indices_2) = zip(*batch)
        return (
            torch.from_numpy(np.stack(images_1)),
            [torch.from_numpy(indices) for indices in voxel_indices_1],
            torch.from_numpy(np.stack(images_2)),
            [torch.from_numpy(indices) for indices in voxel_indices_2],
        )


class _SimCLRDataset(Dataset):
    def __init__(
            self,
            data_paths: SimCLRDataPaths,
            datasets: SimCLRDatasets,
            spatial_augmentations: SimCLRSpatialAugmentations,
            color_augmentations: ColorAugmentations,
            num_voxels_per_crop: int,
            num_images_per_epoch: int,
            nlst_val_size: int,
            random_seed: int
    ) -> None:
        super().__init__()

        self.spatial_augmentations = spatial_augmentations
        self.color_augmentations = color_augmentations
        self.num_voxels_per_crop = num_voxels_per_crop
        self.num_images_per_epoch = num_images_per_epoch

        nlst_image_dirpaths, _ = train_test_split(list(Path(data_paths.nlst_dirpath).iterdir()),
                                                  test_size=nlst_val_size, random_state=random_seed)
        random.seed(random_seed)
        self.image_dirpaths = (
            get_random_sample(population=nlst_image_dirpaths,
                              size=datasets.nlst)
            + get_random_sample(population=list(Path(data_paths.amos_ct_labeled_train_dirpath).iterdir()),
                                size=datasets.amos_ct_labeled_train)
            + get_random_sample(population=list(Path(data_paths.amos_ct_unlabeled_train_dirpath).iterdir()),
                                size=datasets.amos_ct_unlabeled_train)
            + get_random_sample(population=list(Path(data_paths.abdomen_atlas_dirpath).iterdir()),
                                size=datasets.abdomen_atlas)
            + get_random_sample(population=list(Path(data_paths.flare23_labeled_train_dirpath).iterdir()),
                                size=datasets.flare23_labeled_train)
            + get_random_sample(population=list(Path(data_paths.flare23_unlabeled_train_dirpath).iterdir()),
                                size=datasets.flare23_unlabeled_train)
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
            num_voxels_per_crop=self.num_voxels_per_crop,
        )


def _get_augmented_crops(
        image: np.ndarray,
        voxel_spacing: Tuple[float, float, float],
        body_mask: np.ndarray,
        spatial_augmentations: SimCLRSpatialAugmentations,
        color_augmentations: ColorAugmentations,
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

    image_1, voxel_indices_1 = _get_augmented_crop(image, voxel_spacing, voxel_indices,
                                                   crop_box_1, resize_factor_1, color_augmentations)
    image_2, voxel_indices_2 = _get_augmented_crop(image, voxel_spacing, voxel_indices,
                                                   crop_box_2, resize_factor_2, color_augmentations)
    return image_1, voxel_indices_1, image_2, voxel_indices_2


def _get_augmented_crop(
        image: np.ndarray,
        voxel_spacing: np.ndarray,
        voxel_indices: np.ndarray,
        crop_box: np.ndarray,
        resize_factor: np.ndarray,
        color_augmentations: ColorAugmentations
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

    # add channel dim
    image = np.expand_dims(image, axis=0)

    return image, voxel_indices
