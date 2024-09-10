from omegaconf import DictConfig
import hydra

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


@hydra.main(version_base=None, config_path='../configs', config_name=None)
def main(config: DictConfig):
    if not config:
        raise RuntimeError('Please, specify config via ``--config-name`` CLI argument.')

    # 1. Model
    model: pl.LightningModule = hydra.utils.instantiate(config.eval_model)

    # 2. Datamodule
    dm: pl.LightningDataModule = hydra.utils.instantiate(config.eval_datamodule)
    dm.prepare_data()
    dm.setup('fit')

    # 3. Callbacks
    callbacks = [
        LearningRateMonitor()
    ]

    # 4. Trainer
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks)

    # 5. Run
    trainer.fit(
        model=model,
        datamodule=dm,
        ckpt_path=config.ckpt_path
    )


if __name__ == '__main__':
    main()
