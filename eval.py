from argparse import ArgumentParser
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, BackboneFinetuning

from vox2vec.default_params import *
from vox2vec.eval.btcv import BTCV
from vox2vec.nn import FPN3d, FPNLinearHead, FPNNonLinearHead
from vox2vec.eval.end_to_end import EndToEnd
from vox2vec.eval.probing import Probing
from vox2vec.utils.misc import save_json


def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument('--dataset', default='btcv')
    parser.add_argument('--btcv_dir', required=True)
    parser.add_argument('--cache_dir', required=True)
    parser.add_argument('--ckpt')
    parser.add_argument('--setup', required=True)
    parser.add_argument('--log_dir', required=True)

    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--spacing', nargs='+', type=float, default=SPACING)
    parser.add_argument('--patch_size', nargs='+', type=int, default=PATCH_SIZE)

    parser.add_argument('--batch_size', type=int, default=7)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--num_batches_per_epoch', type=int, default=300)
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--warmup_epochs', type=int, default=50)  # used only in finetuning setup

    parser.add_argument('--base_channels', type=int, default=BASE_CHANNELS)
    parser.add_argument('--num_scales', type=int, default=NUM_SCALES)

    return parser.parse_args()


def main(args):
    if args.dataset == 'btcv':
        datamodule = BTCV(
            root=args.btcv_dir,
            cache_dir=args.cache_dir,
            spacing=tuple(args.spacing),
            window_hu=WINDOW_HU,
            patch_size=tuple(args.patch_size),
            batch_size=args.batch_size,
            num_batches_per_epoch=args.num_batches_per_epoch,
            num_workers=args.num_workers,
            prefetch_factor=1,
            split=args.split,
        )
        num_classes = BTCV.num_classes
    else:
        raise NotImplementedError(f'Dataset {args.dataset} is not supported.')

    in_channels = 1
    backbone = FPN3d(in_channels, args.base_channels, args.num_scales)
    if args.setup == 'from_scratch':
        head = FPNLinearHead(args.base_channels, args.num_scales, num_classes)
        model = EndToEnd(backbone, head, patch_size=tuple(args.patch_size))
        callbacks = [
            ModelCheckpoint(save_top_k=1, monitor='val/avg_dice_score', filename='best', mode='max'),
        ]
    elif args.setup == 'probing':
        if args.ckpt is not None:
            backbone.load_state_dict(torch.load(args.ckpt))
        heads = [
            FPNLinearHead(args.base_channels, args.num_scales, num_classes),
            FPNNonLinearHead(args.base_channels, args.num_scales, num_classes)
        ]
        model = Probing(backbone, *heads, patch_size=tuple(args.patch_size))
        callbacks = [
            ModelCheckpoint(save_top_k=1, monitor='val/head_1_avg_dice_score', filename='best', mode='max'),
        ]
    elif args.setup == 'fine-tuning':
        if args.ckpt is not None:
            backbone.load_state_dict(torch.load(args.ckpt))
        head = FPNLinearHead(args.base_channels, args.num_scales, num_classes)
        model = EndToEnd(backbone, head, patch_size=tuple(args.patch_size))
        callbacks = [
            BackboneFinetuning(unfreeze_backbone_at_epoch=args.warmup_epochs),
            ModelCheckpoint(save_top_k=1, monitor='val/avg_dice_score', filename='best', mode='max'),
        ]
    else:
        raise ValueError(args.setup)

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=f'eval/{args.dataset}/{args.setup}/split_{args.split}'
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator='gpu',
        max_epochs=args.max_epochs,
    )

    trainer.fit(model, datamodule)

    log_dir = Path(logger.log_dir)
    test_metrics = trainer.test(model, datamodule=datamodule, ckpt_path=log_dir / 'checkpoints/best.ckpt')
    save_json(test_metrics, log_dir / 'test_metrics.json')


if __name__ == '__main__':
    main(parse_args())
