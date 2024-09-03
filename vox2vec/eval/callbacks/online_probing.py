from typing import Tuple

import torch
import torch.nn as nn
import lightning.pytorch as pl

from monai.inferers import sliding_window_inference

from vox2vec.nn.functional import (
    eval_mode, binary_dice_loss, segmentation_bce_loss, binary_soft_dice_score
)
from .visualize import draw


class BinaryOnlineProbing(pl.Callback):
    def __init__(
            self,
            *heads: nn.Module,
            lr: float,
            weight_decay: float,
            gradient_clip_val: float,
            crop_size: Tuple[int, int, int],
            sw_batch_size: int,
            backbone_warmup_epochs: int = 1,
            draw_n_first_val_examples: int = 10
    ):
        super().__init__()

        self.heads = nn.ModuleList(heads)
        self.weight_decay = weight_decay
        self.lr = lr
        self.gradient_clip_val = gradient_clip_val
        self.crop_size = crop_size
        self.sw_batch_size = sw_batch_size
        self.backbone_warmup_epochs = backbone_warmup_epochs
        self.draw_n_first_val_examples = draw_n_first_val_examples

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.heads.to(pl_module.device)
        self.optimizer = torch.optim.AdamW(
            params=self.heads.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        if trainer.current_epoch < self.backbone_warmup_epochs:
            return

        self.optimizer.zero_grad()

        batch = batch['online_probing']
        images, roi_masks, gt_masks = batch

        with torch.no_grad(), eval_mode(pl_module):
            outputs = pl_module.forward(images)

        for i, head in enumerate(self.heads):
            pred_logits = head(images, outputs)
            pred_probs = torch.sigmoid(pred_logits)

            bce_loss = segmentation_bce_loss(pred_logits, gt_masks, roi_masks)
            pl_module.log(f'binary_online_probing/head_{i}_bce_loss', bce_loss, on_epoch=True, on_step=True)

            positive = gt_masks.flatten(1, -1).any(1)
            if positive.any():
                dice_loss = binary_dice_loss(pred_probs[positive], gt_masks[positive], roi_masks[positive])
                pl_module.log(f'binary_online_probing/head_{i}_dice_loss', bce_loss, on_epoch=True, on_step=True)
            else:
                dice_loss = 0.0

            loss = bce_loss + dice_loss
            loss.backward()

        torch.nn.utils.clip_grad_norm_(self.heads.parameters(), self.gradient_clip_val)
        self.optimizer.step()

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                                outputs, batch, batch_idx, dataloader_idx=0):
        image, roi_mask, gt_mask = batch

        if not gt_mask.any():
            return

        for i, head in enumerate(self.heads):
            def predictor(x):
                return torch.sigmoid(head(x, pl_module.forward(x)))

            pred_probs = sliding_window_inference(
                inputs=image.unsqueeze(0),
                roi_size=self.crop_size,
                sw_batch_size=self.sw_batch_size,
                predictor=predictor,
                sw_device=pl_module.device,
                device='cpu',
                overlap=0.5,
                mode='gaussian',
            ).squeeze(0)
            pred_probs *= roi_mask
            pred_mask = pred_probs > 0.5

            dice_scores = binary_soft_dice_score(pred_probs, gt_mask)
            for j in range(len(dice_scores)):
                pl_module.log(f'val/head_{i}_dice_score_for_cls_{j}', dice_scores[j].item(), on_epoch=True)
            pl_module.log(f'val/head_{i}_avg_dice_score', dice_scores.mean().item(), on_epoch=True)

            if batch_idx < self.draw_n_first_val_examples:
                log_image = draw(image, gt_mask, pred_mask)
                trainer.logger.experiment.add_image(
                    f'val/head_{i}_image_{batch_idx}',
                    log_image, trainer.current_epoch
                )
