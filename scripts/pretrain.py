from typing import List
from omegaconf import DictConfig
import hydra

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.utilities import CombinedLoader

from vox2vec.eval.datamodules.lidc import LIDCDataModule
from vox2vec.eval.callbacks.online_probing import OnlineProbing

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


@hydra.main(version_base=None, config_path='../configs', config_name=None)
def main(config: DictConfig):
    """Run pre-training with online probing.

    Args:
        cfg (DictConfig):
            Configuration:
                - pretraining model class and hyperparameters;
                - pretraining datamodule class and hyperparameters;
                - trainer settings;
            Default configuration is defined in configs/pretrain.yaml.
            It can be modified from command line:
            >>> python pretrain.py pretrain_model=simclr pretrain_datamodule=simclr
    """
    if not config:
        raise RuntimeError('Please, specify config via ``--config-name`` CLI argument.')

    # 1. Pretraining model
    model: pl.LightningModule = hydra.utils.instantiate(config.pretrain_model)

    # 2. Pretraining datamodule
    pretrain_dm: pl.LightningDataModule = hydra.utils.instantiate(config.pretrain_datamodule)
    pretrain_dm.prepare_data()
    pretrain_dm.setup('fit')

    # 3. Online probing datamodule
    online_probing_dm = LIDCDataModule(
        lidc_dirpath=config.paths.lidc_dirpath,
        crop_size=config.online_probing.crop_size,
        batch_size=config.online_probing.batch_size,
        num_batches_per_epoch=config.pretrain_datamodule.num_batches_per_epoch,
        num_workers=config.online_probing.num_workers,
    )
    online_probing_dm.prepare_data()
    online_probing_dm.setup('fit')

    # 4. Callbacks
    callbacks = [
        OnlineProbing(
            hydra.utils.instantiate(config.heads.linear),
            hydra.utils.instantiate(config.heads.nonlinear),
            crop_size=config.online_probing.crop_size,
            sw_batch_size=config.online_probing.batch_size,
        ),
        ModelCheckpoint(
            filename='best',
            monitor='online_probing/head_1_dice_score_for_cls_0',
            save_last=True,
            mode='max',
        )
    ]

    # 5. Trainer
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks)

    # 6. Run pretraining with online probing
    trainer.fit(
        model=model,
        # train_dataloaders=pretrain_dm,
        train_dataloaders=CombinedLoader({
            'pretrain': pretrain_dm.train_dataloader(),
            'online_probing': online_probing_dm.train_dataloader()
        }),
        val_dataloaders=online_probing_dm.val_dataloader(),
        ckpt_path=config.ckpt_path
    )


if __name__ == '__main__':
    main()
