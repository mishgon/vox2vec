from typing import Tuple
from pathlib import Path
import numpy as np
import random
from sklearn.model_selection import train_test_split
from imops import crop_to_box

from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from vox2vec.utils.io import load_numpy
from vox2vec.utils.box import get_random_box


class LIDCDataModule(pl.LightningDataModule):
    def __init__(
            self,
            lidc_dirpath: str,
            crop_size: Tuple[int, int, int] = (128, 128, 64),
            batch_size: int = 8,
            num_batches_per_epoch: int = 300,
            num_workers: int = 0,
            val_size: int = 100,
            train_size: float = 1.0,
            random_seed: int = 42,
    ):
        super().__init__()

        self.lidc_dirpath = Path(lidc_dirpath)
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_workers = num_workers
        self.val_size = val_size
        self.train_size = train_size
        self.random_seed = random_seed

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = 'fit') -> None:
        if stage == 'fit':
            num_images_per_epoch = self.num_batches_per_epoch * self.batch_size
            self.train_dataset = _LIDCTrainDataset(
                lidc_dirpath=self.lidc_dirpath,
                crop_size=self.crop_size,
                val_size=self.val_size,
                train_size=self.train_size,
                num_images_per_epoch=num_images_per_epoch,
                random_seed=self.random_seed
            )
            self.val_dataset = _LIDCValDataset(
                lidc_dirpath=self.lidc_dirpath,
                val_size=self.val_size,
                random_seed=self.random_seed
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=None,
        )


class _LIDCTrainDataset(Dataset):
    def __init__(
            self,
            lidc_dirpath: Path,
            crop_size: Tuple[int, int, int],
            val_size: int,
            train_size: float,
            num_images_per_epoch: int,
            random_seed: int
    ):
        super().__init__()

        self.lidc_dirpath = lidc_dirpath
        self.crop_size = crop_size

        ids = sorted(path.name for path in lidc_dirpath.iterdir())
        train_ids, val_ids = train_test_split(ids, test_size=val_size, random_state=random_seed)
        if train_size < 1.0:
            train_ids, _ = train_test_split(train_ids, train_size=train_size, random_state=random_seed)
        self.ids = train_ids

        self.num_images_per_epoch = num_images_per_epoch

    def __len__(self):
        return self.num_images_per_epoch

    def __getitem__(self, index):
        i = random.choice(self.ids)
        image = load_numpy(self.lidc_dirpath / i / 'image.npy.gz', decompress=True)
        body_mask = load_numpy(self.lidc_dirpath / i / 'body_mask.npy.gz', decompress=True)
        mask = load_numpy(self.lidc_dirpath / i / 'mask.npy.gz', decompress=True)

        if random.uniform(0, 1) < 0.9 and mask.any():
            pins = np.argwhere(mask)
            pins = pins[[random.randint(0, len(pins) - 1)]]
        else:
            pins = None
        crop_box = get_random_box(image.shape, self.crop_size, pins)
        image = crop_to_box(image, crop_box)
        body_mask = crop_to_box(body_mask, crop_box)
        mask = crop_to_box(mask, crop_box)

        image = image.astype('float32')
        mask = mask.astype('float32')

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return image, body_mask, mask


class _LIDCValDataset(Dataset):
    def __init__(
            self,
            lidc_dirpath: Path,
            val_size: int,
            random_seed: int
    ):
        super().__init__()

        self.lidc_dirpath = lidc_dirpath

        ids = sorted(path.name for path in lidc_dirpath.iterdir())
        train_ids, val_ids = train_test_split(ids, test_size=val_size, random_state=random_seed)
        self.ids = val_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        i = self.ids[index]
        image = load_numpy(self.lidc_dirpath / i / 'image.npy.gz', decompress=True)
        body_mask = load_numpy(self.lidc_dirpath / i / 'body_mask.npy.gz', decompress=True)
        mask = load_numpy(self.lidc_dirpath / i / 'mask.npy.gz', decompress=True)

        image = image.astype('float32')
        mask = mask.astype('float32')

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return image, body_mask, mask
