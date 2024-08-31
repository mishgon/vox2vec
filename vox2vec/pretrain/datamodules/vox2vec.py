from dataclasses import dataclass
from typing import Tuple, Optional, Union, Sequence, Any, List, NamedTuple, Dict
from pathlib import Path
import math
import random
import numpy as np
from scipy.ndimage import gaussian_filter1d
from imops import crop_to_box, zoom, zoom_to_shape as resize

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from vox2vec.utils.io import load_numpy, load_json
from vox2vec.utils.misc import normalize_axis_list


@dataclass
class Vox2VecDataPaths:
    nlst_dirpath: str
    amos_ct_labeled_train_dirpath: str
    amos_ct_unlabeled_train_dirpath: str
    abdomen_atlas_dirpath: str
    flare23_labeled_train_dirpath: str
    flare23_unlabeled_train_dirpath: str


@dataclass
class Vox2VecDatasetConfig:
    nlst_size: Union[float, int] = 1.0
    amos_ct_labeled_train_size: Union[float, int] = 1.0
    amos_ct_unlabeled_train_size: Union[float, int] = 1.0
    abdomen_atlas_size: Union[float, int] = 1.0
    flare23_labeled_train_size: Union[float, int] = 1.0
    flare23_unlabeled_train_size: Union[float, int] = 1.0


@dataclass
class Vox2VecSpatialAugmentationsConfig:
    context_min_voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 2.0)
    context_max_voxel_spacing: Tuple[float, float, float] = (2.0, 2.0, 4.0)
    context_size: Tuple[int, int, int] = (128, 128, 64)
    target_size: Tuple[int, int, int] = (256, 256, 128)
    context_flips_p: float = (0.0, 0.0, 0.0)
    context_rot90: bool = False


@dataclass
class Vox2VecColorAugmentationsConfig:
    blur_or_sharpen_p: float = 0.8
    blur_sigma_range: Tuple[float, float] = (0.0, 1.5)
    sharpen_sigma_range: Tuple[float, float] = (0.0, 1.5)
    sharpen_alpha_range: Tuple[float, float] = (0.0, 2.0)
    noise_p: float = 0.8
    noise_sigma_range: float = (0.0, 0.1)
    invert_p: float = 0.0
    brightness_p: float = 0.8
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_p: float = 0.8
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    gamma_p: float = 0.8
    gamma_range: Tuple[float, float] = (0.8, 1.25)


@dataclass
class Vox2VecMaskingConfig:
    ratio_range: Tuple[float, float] = (0.6, 0.8)
    block_size: Optional[Tuple[int, int, int]] = (32, 32, 16)


class Vox2VecDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_paths: Vox2VecDataPaths,
            dataset_config: Vox2VecDatasetConfig = Vox2VecDatasetConfig(),
            spatial_augmentations_config: Vox2VecSpatialAugmentationsConfig = Vox2VecSpatialAugmentationsConfig(),
            color_augmentations_config: Vox2VecColorAugmentationsConfig = Vox2VecColorAugmentationsConfig(),
            masking_config: Vox2VecMaskingConfig = Vox2VecMaskingConfig(),
            context_crops_per_image: int = 8,
            voxels_per_crop: int = 512,
            batch_size: int = 8,  # images per batch
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
        self.context_crops_per_image = context_crops_per_image
        self.voxels_per_crop = voxels_per_crop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.random_seed = random_seed

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = 'fit') -> None:
        self.train_dataset = Vox2VecDataset(
            data_paths=self.data_paths,
            dataset_config=self.dataset_config,
            spatial_augmentations_config=self.spatial_augmentations_config,
            color_augmentations_config=self.color_augmentations_config,
            masking_config=self.masking_config,
            context_crops_per_image=self.context_crops_per_image,
            voxels_per_crop=self.voxels_per_crop,
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
        (images_1, masks_1, voxels_per_image_1, background_voxels_per_image_1,
         images_2, masks_2, voxels_per_image_2, background_voxels_per_image_2,
         positions_per_image, background_positions_per_image_1,
         background_positions_per_image_2) = zip(*batch)

        return (
            torch.from_numpy(np.stack(images_1)),
            torch.from_numpy(np.stack(masks_1)),
            [torch.from_numpy(voxels) for voxels in voxels_per_image_1],
            [torch.from_numpy(voxels) for voxels in background_voxels_per_image_1],
            torch.from_numpy(np.stack(images_2)),
            torch.from_numpy(np.stack(masks_2)),
            [torch.from_numpy(voxels) for voxels in voxels_per_image_2],
            [torch.from_numpy(voxels) for voxels in background_voxels_per_image_2],
            [torch.from_numpy(positions) for positions in positions_per_image],
            [torch.from_numpy(positions) for positions in background_positions_per_image_1],
            [torch.from_numpy(positions) for positions in background_positions_per_image_2]
        )


class Vox2VecDataset(Dataset):
    def __init__(
            self,
            data_paths: Vox2VecDataPaths,
            dataset_config: Vox2VecDatasetConfig = Vox2VecDatasetConfig(),
            spatial_augmentations_config: Vox2VecSpatialAugmentationsConfig = Vox2VecSpatialAugmentationsConfig(),
            color_augmentations_config: Vox2VecColorAugmentationsConfig = Vox2VecColorAugmentationsConfig(),
            masking_config: Vox2VecMaskingConfig = Vox2VecMaskingConfig(),
            context_crops_per_image: int = 4,
            voxels_per_crop: int = 512,
            random_seed: int = 42
    ) -> None:
        super().__init__()

        self.spatial_augmentations_config = spatial_augmentations_config
        self.color_augmentations_config = color_augmentations_config
        self.masking_config = masking_config
        self.context_crops_per_image = context_crops_per_image
        self.voxels_per_crop = voxels_per_crop

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
        return len(self.image_dirpaths)

    def __getitem__(self, index: int):
        image_dirpath = self.image_dirpaths[index]
        image = load_numpy(image_dirpath / 'image.npy.gz', decompress=True).astype('float32')
        voxel_spacing = load_json(image_dirpath / 'voxel_spacing.json')
        body_mask = load_numpy(image_dirpath / 'body_mask.npy.gz', decompress=True)

        return get_targets_and_contexts(
            image=image,
            voxel_spacing=voxel_spacing,
            body_mask=body_mask,
            spatial_augmentations_config=self.spatial_augmentations_config,
            color_augmentations_config=self.color_augmentations_config,
            masking_config=self.masking_config,
            context_crops_per_image=self.context_crops_per_image,
            voxels_per_crop=self.voxels_per_crop,
        )


def get_random_sample(population: Sequence[Any], size: Union[float, int]) -> List[Any]:
    if isinstance(size, float):
        assert 0.0 <= size <= 1.0
        return random.sample(population, int(len(population) * size))
    elif isinstance(size, int):
        assert 0 <= size <= len(population)
        return random.sample(population, size)
    else:
        raise TypeError(type(size))


def get_targets_and_contexts(
        image: np.ndarray,
        voxel_spacing: Tuple[float, float, float],
        body_mask: np.ndarray,
        spatial_augmentations_config: Vox2VecSpatialAugmentationsConfig,
        color_augmentations_config: Vox2VecColorAugmentationsConfig,
        masking_config: Vox2VecMaskingConfig,
        context_crops_per_image: int,
        voxels_per_crop: int,
) -> Tuple:
    image_size = np.array(image.shape, dtype='int64')
    target_box = get_random_box(image_size, spatial_augmentations_config.target_size)
    target_image = crop_to_box(image, target_box)
    target_image = np.expand_dims(target_image, axis=0)

    target_voxels, context_images, context_masks, context_voxels = [], [], [], []
    for _ in range(context_crops_per_image):
        context_size = np.array(spatial_augmentations_config.context_size, dtype='int64')
        context_rot90_k = random.choice([0, 1, 2, 3]) if spatial_augmentations_config.context_rot90 else 0
        context_size_before_rot = context_size[[1, 0, 2]] if context_rot90_k % 2 else context_size
        context_min_voxel_spacing = np.array(spatial_augmentations_config.context_min_voxel_spacing, dtype='float32')
        context_max_voxel_spacing = np.array(spatial_augmentations_config.context_max_voxel_spacing, dtype='float32')
        context_max_voxel_spacing = np.minimum(context_max_voxel_spacing, voxel_spacing * image_size / context_size_before_rot)
        context_voxel_spacing = np.random.uniform(context_min_voxel_spacing, context_max_voxel_spacing)
        resize_factor = voxel_spacing / context_voxel_spacing
        context_size_before_resize = np.int64(np.round(context_size_before_rot / resize_factor))

        voxel = np.random.randint(*target_box, size=(1, 3))
        context_box = get_random_box(image_size, context_size_before_resize, voxel)
        overlap_box = get_overlap_box(context_box, target_box)
        voxels = overlap_box[0] + np.argwhere(crop_to_box(body_mask, overlap_box))
        if len(voxels) > voxels_per_crop:
            voxels = voxels[np.random.choice(len(voxels), voxels_per_crop, replace=False)]

        target_voxels.append(voxels - target_box[0])

        context_image = crop_to_box(image, context_box)
        voxels = voxels - context_box[0]

        context_image = np.ascontiguousarray(context_image)
        context_image = resize(context_image, context_size_before_rot, backend='Scipy')
        voxels = np.int64(np.floor(voxels * resize_factor))

        # flips
        for p, axis in zip(spatial_augmentations_config.context_flips_p, [-3, -2, -1], strict=True):
            if random.uniform(0, 1) < p:
                context_image = np.flip(context_image, axis)
                voxels[:, axis] = context_image.shape[axis] - 1 - voxels[:, axis]

        # rot90
        if context_rot90_k > 0:
            voxels = rot90_voxels(voxels, k=context_rot90_k, image_size=context_image.shape)
            context_image = np.rot90(context_image, k=context_rot90_k, axes=(0, 1))

        # fix numpy meta after flips/rots
        context_image = context_image.copy()

        # augment colors
        context_image = augment_color(context_image, context_voxel_spacing, color_augmentations_config)

        # sample mask
        context_mask = sample_mask(context_image.shape, masking_config)

        # add channel dim
        context_image = np.expand_dims(context_image, axis=0)

        context_images.append(context_image)
        context_masks.append(context_mask)
        context_voxels.append(voxels)

    return target_image, target_voxels, context_images, context_masks, context_voxels


def augment_color(
        image: np.ndarray,
        voxel_spacing: np.ndarray,
        color_augmentations_config: Vox2VecColorAugmentationsConfig
) -> np.ndarray:
    if random.uniform(0, 1) < color_augmentations_config.blur_or_sharpen_p:
        if random.uniform(0, 1) < 0.5:
            # random gaussian blur in axial plane
            sigma = random.uniform(*color_augmentations_config.blur_sigma_range) / voxel_spacing[:2]
            image = gaussian_filter(image, sigma, axis=(0, 1))
        else:
            sigma = random.uniform(*color_augmentations_config.sharpen_sigma_range) / voxel_spacing[:2]
            alpha = random.uniform(*color_augmentations_config.sharpen_alpha_range)
            image = gaussian_sharpen(image, sigma, alpha, axis=(0, 1))

    if random.uniform(0, 1) < color_augmentations_config.noise_p:
        # gaussian noise
        noise_sigma = random.uniform(*color_augmentations_config.noise_sigma_range)
        image = image + np.random.normal(0, noise_sigma, size=image.shape).astype('float32')

    if random.uniform(0, 1) < color_augmentations_config.invert_p:
        # invert
        image = 1.0 - image

    if random.uniform(0, 1) < color_augmentations_config.brightness_p:
        # adjust brightness
        brightness_factor = random.uniform(*color_augmentations_config.brightness_range)
        image = np.clip(image * brightness_factor, 0.0, 1.0)

    if random.uniform(0, 1) < color_augmentations_config.contrast_p:
        # adjust contrast
        contrast_factor = random.uniform(*color_augmentations_config.contrast_range)
        mean = image.mean()
        image = np.clip((image - mean) * contrast_factor + mean, 0.0, 1.0)

    if random.uniform(0, 1) < color_augmentations_config.gamma_p:
        image = np.clip(image, 0.0, 1.0)
        gamma = random.uniform(*color_augmentations_config.gamma_range)
        image = np.power(image, gamma)

    return image


def sample_mask(size: Sequence[int], masking_config: Vox2VecMaskingConfig) -> np.ndarray:
    if masking_config.ratio_range == 0.0 or random.uniform(0, 1) > masking_config.p:
        return np.ones(size, dtype='float32')

    mask_block_size = np.minimum(masking_config.block_size, size)

    mask = np.ones(size // mask_block_size, dtype='float32')
    mask[np.unravel_index(np.random.permutation(mask.size)[:int(mask.size * masking_config.ratio_range)], mask.shape)] = 0.0
    assert (mask != 1.0).any()
    for axis, repeats in enumerate(mask_block_size):
        mask = np.repeat(mask, repeats, axis)

    padding = size % mask_block_size
    left_padding = np.random.randint(padding + 1)
    right_padding = padding - left_padding
    padding = tuple(zip(left_padding, right_padding))
    mask = np.pad(mask, padding, constant_values=1.0)

    return mask


def get_random_box(
        image_size: Sequence[int],
        box_size: np.ndarray,
        pins: Optional[np.ndarray] = None
) -> np.ndarray:
    image_size = np.array(image_size)

    if not np.all(image_size >= box_size):
        raise ValueError(f'Can\'t sample patch of size {box_size} from image of size {image_size}')

    min_start = 0
    max_start = image_size - box_size

    if pins is not None:
        assert pins.ndim == 2
        assert pins.shape[1] == 3

        min_start = np.maximum(min_start, np.max(pins, axis=0) - box_size + 1)
        max_start = np.minimum(max_start, np.min(pins, axis=0))

        assert np.all(min_start <= max_start)

    start = np.random.randint(min_start, max_start + 1)

    return np.array([start, start + box_size])


def get_overlap_box(*boxes: np.ndarray) -> np.ndarray:
    start = np.max([box[0] for box in boxes], axis=0)
    stop = np.min([box[1] for box in boxes], axis=0)
    if not np.all(start < stop):
        return
    return np.array([start, stop])


def rot90_voxels(voxels: np.ndarray, k: int, image_size: Sequence[int]) -> np.ndarray:
    voxels = voxels.copy()
    angle = math.radians(90 * k)
    image_size = np.array(image_size)

    voxels_xy = voxels[:, [0, 1]]
    image_size_xy = image_size[[0, 1]]
    voxels_xy = voxels_xy - image_size_xy / 2.0
    rot_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])
    voxels_xy = voxels_xy @ rot_matrix.T
    if k % 2:
        image_size_xy = image_size_xy[[1, 0]]
    voxels_xy = voxels_xy + image_size_xy / 2.0
    voxels_xy = np.clip(voxels_xy, 0, image_size_xy - 1)
    voxels[:, [0, 1]] = voxels_xy
    return voxels


def gaussian_filter(
        x: np.ndarray,
        sigma: Union[float, Sequence[float]],
        axis: Union[int, Sequence[int]]
) -> np.ndarray:
    axis = normalize_axis_list(axis, x.ndim)
    sigma = np.broadcast_to(sigma, len(axis))
    for sgm, ax in zip(sigma, axis):
        x = gaussian_filter1d(x, sgm, ax)
    return x


def gaussian_sharpen(
        x: np.ndarray,
        sigma: Union[float, Sequence[float]],
        alpha: float,
        axis: Union[int, Sequence[int]]
) -> np.ndarray:
    return x + alpha * (x - gaussian_filter(x, sigma, axis))
