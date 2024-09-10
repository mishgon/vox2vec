from typing import Optional, Tuple

import torch
from torch import nn, Tensor

import lightning.pytorch as pl

from monai.inferers import sliding_window_inference

from vox2vec.nn.functional import segmentation_bce_loss, binary_dice_loss, binary_dice_score
from ..visualize import draw


class MultilabelSegmentation(pl.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            head: nn.Module,
            crop_size: Tuple[int, int, int],
            sw_batch_size: int = 1,
            lr: float = 3e-4,
            weight_decay: float = 1e-6,
            warmup_steps: Optional[int] = None,
            total_steps: Optional[int] = None,
            draw_n_first_val_examples: int = 10,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=['backbone', 'head'])

        self.backbone = backbone
        self.head = head
        self.crop_size = crop_size
        self.sw_batch_size = sw_batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.draw_n_first_val_examples = draw_n_first_val_examples

    def forward(self, images: Tensor) -> Tensor:
        return self.head(images, self.backbone(images))

    def training_step(self, batch, batch_idx):
        images, roi_masks, gt_masks = batch
        pred_logits = self.forward(images)
        pred_probs = torch.sigmoid(pred_logits)
        bce_loss = segmentation_bce_loss(pred_logits, gt_masks, roi_masks)
        self.log(f'multilabel_segmentation/bce_loss', bce_loss, on_epoch=True, on_step=True)

        positive = gt_masks.flatten(1, -1).any(1)
        if positive.any():
            dice_loss = binary_dice_loss(pred_probs[positive], gt_masks[positive], roi_masks[positive])
            self.log(f'multilabel_segmentation/dice_loss', dice_loss, on_epoch=True, on_step=True)
        else:
            dice_loss = 0.0

        return bce_loss + dice_loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        image, roi, gt_mask = batch

        if not gt_mask.any():
            return

        def predictor(x):
            return torch.sigmoid(self.forward(x))

        pred_prob = sliding_window_inference(
            inputs=image.unsqueeze(0),
            roi_size=self.crop_size,
            sw_batch_size=self.sw_batch_size,
            predictor=predictor,
            overlap=0.5,
            mode='gaussian',
            sw_device=self.device,
            device='cpu',
        ).squeeze(0)
        pred_prob *= roi
        pred_mask = pred_prob > 0.5

        soft_dice_scores = binary_dice_score(pred_prob, gt_mask)
        dice_scores = binary_dice_score(pred_mask, gt_mask)
        for j in range(len(soft_dice_scores)):
            self.log(f'multilabel_segmentation/soft_dice_score_for_cls_{j}', soft_dice_scores[j].item(), on_epoch=True)
            self.log(f'multilabel_segmentation/dice_score_for_cls_{j}', dice_scores[j].item(), on_epoch=True)
        if len(soft_dice_scores) > 1:
            self.log(f'multilabel_segmentation/avg_soft_dice_score', soft_dice_scores.mean().item(), on_epoch=True)
            self.log(f'multilabel_segmentation/avg_dice_score', dice_scores.mean().item(), on_epoch=True)

        if batch_idx < self.draw_n_first_val_examples:
            log_image = draw(image, gt_mask, pred_mask)
            self.logger.experiment.add_image(
                f'multilabel_segmentation/image_{batch_idx}',
                log_image, self.current_epoch
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            eps=1e-3,
            weight_decay=self.weight_decay
        )

        if self.warmup_steps is None:
            return optimizer

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.total_steps,
            pct_start=self.warmup_steps / self.total_steps,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.trainer.training:
            # skip device transfer for the val and test dataloaders
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
