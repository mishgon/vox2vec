from dataclasses import dataclass
from typing import Tuple, Optional, Union, Sequence
from pathlib import Path
import random
import numpy as np
from skimage.measure import block_reduce
from sklearn.model_selection import train_test_split
from imops import crop_to_box

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from vox2vec.utils.io import load_numpy, load_json
from vox2vec.utils.misc import get_random_sample
from vox2vec.utils.box import get_random_box
from vox2vec.typing import PreparedDataDirs, PretrainDataFractions
from .augmentations import ColorAugmentations, augment_color


class MIMDataModule(pl.LightningDataModule):
    def __init__(
            self,
            prepared_data_dirs: PreparedDataDirs,
            nlst_val_size: int = 1000,
            pretrain_data_fractions: PretrainDataFractions = PretrainDataFractions(),
            target_crop_size: Tuple[int, int, int] = (192, 192, 96),
            context_crop_size: Tuple[int, int, int] = (160, 160, 80),
            color_augmentations: ColorAugmentations = ColorAugmentations(),
            token_size_per_scale: Sequence[Tuple[int, int, int]] = ((4, 4, 2), (8, 8, 4), (16, 16, 8), (32, 32, 16)),
            num_blocks_per_scale: Sequence[int] = (512, 64, 8, 1),
            max_block_aspect_ratio: float = 3 / 2,
            mask_ratio_range: Tuple[float, float] = (0.6, 0.8),
            batch_size: int = 8,  # images per batch
            num_batches_per_epoch: int = 3000,
            num_workers: int = 0,
            prefetch_factor: Optional[int] = None,
            random_seed: int = 42,
    ) -> None:
        super().__init__()

        self.prepared_data_dirs = prepared_data_dirs
        self.nlst_val_size = nlst_val_size
        self.pretrain_data_fractions = pretrain_data_fractions
        self.target_crop_size = tuple(target_crop_size)
        self.context_crop_size = tuple(context_crop_size)
        self.color_augmentations = color_augmentations
        self.token_size_per_scale = list(map(tuple, token_size_per_scale))
        self.num_blocks_per_scale = list(num_blocks_per_scale)
        self.max_block_aspect_ratio = max_block_aspect_ratio
        self.mask_ratio_range = tuple(mask_ratio_range)
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.random_seed = random_seed

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = 'fit') -> None:
        num_images_per_epoch = self.batch_size * self.num_batches_per_epoch
        self.train_dataset = _MIMDataset(
            prepared_data_dirs=self.prepared_data_dirs,
            nlst_val_size=self.nlst_val_size,
            pretrain_data_fractions=self.pretrain_data_fractions,
            target_crop_size=self.target_crop_size,
            context_crop_size=self.context_crop_size,
            color_augmentations=self.color_augmentations,
            token_size_per_scale=self.token_size_per_scale,
            num_blocks_per_scale=self.num_blocks_per_scale,
            max_block_aspect_ratio=self.max_block_aspect_ratio,
            mask_ratio_range=self.mask_ratio_range,
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
        (target_images, target_image_masks, target_masked_token_indices,
         context_images, context_image_masks, context_tokens_masks,
         context_masked_token_indices) = zip(*batch)
        
        n = len(target_masked_token_indices)  # batch size
        m = len(target_masked_token_indices[0])  # num scales

        return (
            torch.from_numpy(np.array(target_images)),
            [torch.from_numpy(np.stack([target_image_masks[i][j] for i in range(n)])) for j in range(m)],
            [[torch.from_numpy(target_masked_token_indices[i][j]) for i in range(n)] for j in range(m)],
            [torch.from_numpy(np.stack([context_images[i][j] for i in range(n)])) for j in range(m)],
            [torch.from_numpy(np.stack([context_image_masks[i][j] for i in range(n)])) for j in range(m)],
            [torch.from_numpy(np.stack([context_tokens_masks[i][j] for i in range(n)])) for j in range(m)],
            [[torch.from_numpy(context_masked_token_indices[i][j]) for i in range(n)] for j in range(m)],
        )


class _MIMDataset(Dataset):
    def __init__(
            self,
            prepared_data_dirs: PreparedDataDirs,
            nlst_val_size: int,
            pretrain_data_fractions: PretrainDataFractions,
            target_crop_size: Tuple[int, int, int],
            context_crop_size: Tuple[int, int, int],
            color_augmentations: ColorAugmentations,
            token_size_per_scale: Sequence[Tuple[int, int, int]],
            num_blocks_per_scale: Sequence[int],
            max_block_aspect_ratio: float,
            mask_ratio_range: Tuple[float, float],
            num_images_per_epoch: int,
            random_seed: int
    ) -> None:
        super().__init__()

        random.seed(random_seed)

        self.target_crop_size = target_crop_size
        self.context_crop_size = context_crop_size
        self.color_augmentations = color_augmentations
        self.token_size_per_scale = token_size_per_scale
        self.num_blocks_per_scale = num_blocks_per_scale
        self.max_block_aspect_ratio = max_block_aspect_ratio
        self.mask_ratio_range = mask_ratio_range
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
            + get_random_sample(population=sorted(Path(prepared_data_dirs.flare23_labeled_train).iterdir()),
                                size=pretrain_data_fractions.flare23_labeled_train)
            + get_random_sample(population=sorted(Path(prepared_data_dirs.flare23_unlabeled_train).iterdir()),
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
            target_crop_size=self.target_crop_size,
            context_crop_size=self.context_crop_size,
            color_augmentations=self.color_augmentations,
            token_size_per_scale=self.token_size_per_scale,
            num_blocks_per_scale=self.num_blocks_per_scale,
            max_block_aspect_ratio=self.max_block_aspect_ratio,
            mask_ratio_range=self.mask_ratio_range
        )


def _get_targets_and_contexts(
        image: np.ndarray,
        voxel_spacing: Tuple[int, int, int],
        body_mask: np.ndarray,
        target_crop_size: Tuple[int, int, int],
        context_crop_size: Tuple[int, int, int],
        color_augmentations: ColorAugmentations,
        token_size_per_scale: Sequence[Tuple[int, int, int]],
        num_blocks_per_scale: Sequence[int],
        max_block_aspect_ratio: float,
        mask_ratio_range: Tuple[float, float],
) -> Tuple:
    img_size = np.array(image.shape, dtype='int64')
    tgt_size = np.array(target_crop_size, dtype='int64')
    cxt_size = np.array(context_crop_size, dtype='int64')

    tgt_box = get_random_box(img_size, tgt_size)
    tgt_img = crop_to_box(image, tgt_box)
    tgt_img = augment_color(tgt_img, voxel_spacing, color_augmentations)
    tgt_img = np.expand_dims(tgt_img, axis=0)  # add channel dim

    tgt_img_msks, tgt_msk_tkn_idxs, cxt_imgs, cxt_img_msks, cxt_tkn_msks, cxt_msk_tkn_idxs = [], [], [], [], [], []
    for tkn_size, num_blocks in zip(token_size_per_scale, num_blocks_per_scale):
        assert np.all(tgt_size % tkn_size == 0)
        assert np.all(cxt_size % tkn_size == 0)

        cxt_box = tgt_box[0] + get_random_box(tgt_size // tkn_size, cxt_size // tkn_size) * tkn_size
        cxt_img = crop_to_box(image, cxt_box)
        cxt_img = augment_color(cxt_img, voxel_spacing, color_augmentations)
        cxt_img = np.expand_dims(cxt_img, axis=0)  # add channel dim
        cxt_body_msk = crop_to_box(body_mask, cxt_box)
        cxt_img_msk, cxt_tkn_msk, _cxt_msk_tkn_idxs = _get_context_mask(
            cxt_body_msk, cxt_size, tkn_size, num_blocks,
            max_block_aspect_ratio, mask_ratio_range
        )
        pad_width = np.stack([cxt_box[0] - tgt_box[0], tgt_box[1] - cxt_box[1]], axis=1)
        tgt_img_msk = np.pad(cxt_img_msk, pad_width, constant_values=1.0)
        _tgt_msk_tkn_idxs = _cxt_msk_tkn_idxs + (cxt_box[0] - tgt_box[0]) // tkn_size

        tgt_img_msks.append(tgt_img_msk)
        tgt_msk_tkn_idxs.append(_tgt_msk_tkn_idxs)
        cxt_imgs.append(cxt_img)
        cxt_img_msks.append(cxt_img_msk)
        cxt_tkn_msks.append(cxt_tkn_msk)
        cxt_msk_tkn_idxs.append(_cxt_msk_tkn_idxs)

    return tgt_img, tgt_img_msks, tgt_msk_tkn_idxs, cxt_imgs, cxt_img_msks, cxt_tkn_msks, cxt_msk_tkn_idxs


def _get_context_mask(
        body_mask: np.ndarray,
        context_crop_size: np.ndarray,
        token_size: np.ndarray,
        num_blocks: int,
        max_block_aspect_ratio: float,
        mask_ratio_range: Tuple[float, float]
) -> np.ndarray:
    assert np.all(context_crop_size % token_size == 0)

    mask_ratio = random.uniform(*mask_ratio_range)
    body_tokens_mask = block_reduce(body_mask, token_size, func=np.mean) > 0.5
    num_body_tokens = np.count_nonzero(body_tokens_mask)
    num_tokens_per_block = num_body_tokens * mask_ratio / num_blocks
    context_crop_size_tokens = context_crop_size // token_size
    tokens_mask = np.ones(context_crop_size_tokens, dtype=bool)
    for _ in range(num_blocks):
        block_aspect_proportions = _get_block_aspect_proportions(max_block_aspect_ratio)
        block_size_tokens = np.int64(np.round(num_tokens_per_block ** (1 / 3) * block_aspect_proportions))
        block_size_tokens = np.minimum(block_size_tokens, context_crop_size_tokens - 1)
        block_box_tokens = get_random_box(context_crop_size_tokens, block_size_tokens)
        tokens_mask[tuple(map(slice, *block_box_tokens))] = False

    tokens_mask = tokens_mask | np.logical_not(body_tokens_mask)

    masked_token_indices = np.argwhere(tokens_mask == False)

    image_mask = tokens_mask = tokens_mask.astype('float32')
    for i in range(3):
        image_mask = np.repeat(image_mask, repeats=token_size[i], axis=i)

    return image_mask, tokens_mask, masked_token_indices


def _get_block_aspect_proportions(max_block_aspect_ratio: float) -> np.ndarray:
    assert max_block_aspect_ratio >= 1.0

    shortest_side = 1.0
    first_other_side = random.uniform(1.0, max_block_aspect_ratio)
    second_other_side = random.uniform(1.0, max_block_aspect_ratio)
    sides = np.array([shortest_side, first_other_side, second_other_side])  # (n, 3)
    sides = np.random.permutation(sides)
    aspect_proportions = sides / np.prod(sides) ** (1 / 3)

    return aspect_proportions
