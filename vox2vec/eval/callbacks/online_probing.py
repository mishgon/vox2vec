from typing import Tuple

import torch
import torch.nn as nn
import lightning.pytorch as pl

from monai.inferers import sliding_window_inference

from vox2vec.nn.functional import (
    eval_mode, binary_dice_loss, segmentation_bce_loss, binary_dice_score
)
from ..visualize import draw


class OnlineProbing(pl.Callback):
    def __init__(
            self,
            *heads: nn.Module,
            crop_size: Tuple[int, int, int],
            sw_batch_size: int = 1,
            lr: float = 3e-4,
            weight_decay: float = 1e-6,
            gradient_clip_val: float = 1.0,
            backbone_warmup_epochs: int = 1,
            draw_n_first_val_examples: int = 10
    ):
        super().__init__()

        self.heads = nn.ModuleList(heads)
        self.crop_size = crop_size
        self.sw_batch_size = sw_batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
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

        for param in self.heads.parameters():
            param.grad = None

        batch = batch['online_probing']
        images, roi_masks, gt_masks = batch

        with torch.no_grad(), eval_mode(pl_module):
            outputs = pl_module.forward(images)

        for i, head in enumerate(self.heads):
            pred_logits = head(images, outputs)
            pred_probs = torch.sigmoid(pred_logits)

            bce_loss = segmentation_bce_loss(pred_logits, gt_masks, roi_masks)
            pl_module.log(f'online_probing/head_{i}_bce_loss', bce_loss, on_epoch=True, on_step=True)

            positive = gt_masks.flatten(1, -1).any(1)
            if positive.any():
                dice_loss = binary_dice_loss(pred_probs[positive], gt_masks[positive], roi_masks[positive])
                pl_module.log(f'online_probing/head_{i}_dice_loss', dice_loss, on_epoch=True, on_step=True)
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

            pred_prob = sliding_window_inference(
                inputs=image.unsqueeze(0),
                roi_size=self.crop_size,
                sw_batch_size=self.sw_batch_size,
                predictor=predictor,
                overlap=0.5,
                mode='gaussian',
                sw_device=pl_module.device,
                device='cpu',
            ).squeeze(0)
            pred_prob *= roi_mask
            pred_mask = pred_prob > 0.5

            soft_dice_scores = binary_dice_score(pred_prob, gt_mask)
            dice_scores = binary_dice_score(pred_mask, gt_mask)
            for j in range(len(soft_dice_scores)):
                pl_module.log(f'online_probing/head_{i}_soft_dice_score_for_cls_{j}', soft_dice_scores[j].item(), on_epoch=True)
                pl_module.log(f'online_probing/head_{i}_dice_score_for_cls_{j}', dice_scores[j].item(), on_epoch=True)
            if len(soft_dice_scores) > 1:
                pl_module.log(f'online_probing/head_{i}_avg_soft_dice_score', soft_dice_scores.mean().item(), on_epoch=True)
                pl_module.log(f'online_probing/head_{i}_avg_dice_score', dice_scores.mean().item(), on_epoch=True)

            if batch_idx < self.draw_n_first_val_examples:
                log_image = draw(image, gt_mask, pred_mask)
                trainer.logger.experiment.add_image(
                    f'online_probing/head_{i}_image_{batch_idx}',
                    log_image, trainer.current_epoch
                )
