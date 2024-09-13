from omegaconf import DictConfig
import hydra

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


@hydra.main(version_base=None, config_path='../configs', config_name='pretrain_screener')
def main(config: DictConfig):
    # 1. Pretraining model
    model: pl.LightningModule = hydra.utils.instantiate(config.pretrain_model)

    # 2. Pretraining datamodule
    pretrain_dm: pl.LightningDataModule = hydra.utils.instantiate(config.pretrain_datamodule)
    pretrain_dm.prepare_data()
    pretrain_dm.setup('fit')

    # 3. Callbacks
    callbacks = [
        ModelCheckpoint(
            filename='best',
            monitor='simclr/loss',
            save_last=True,
            mode='min',
        ),
        LearningRateMonitor()
    ]

    # 4. Trainer
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks)

    # 5. Run pretraining with online probing
    trainer.fit(
        model=model,
        train_dataloaders=pretrain_dm,
        ckpt_path=config.ckpt_path
    )


if __name__ == '__main__':
    main()
