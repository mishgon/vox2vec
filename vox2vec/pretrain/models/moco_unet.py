from typing import Optional
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

from vox2vec.nn.unet_3d import UNet3d
from vox2vec.nn.functional import batched_take_features_from_map
from .ema import MomentumUpdater


class MoCoUNet(pl.LightningModule):
    def __init__(
            self,
            backbone: UNet3d,
            proj_hidden_dim: int = 512,
            proj_out_dim: int = 128,
            pred_hidden_dim: int = 512,
            temp: float = 0.1,
            queue_size: int = 65_536,
            tau: float = 0.999,
            lr: float = 3e-4,
            weight_decay: float = 1e-6,
            warmup_steps: Optional[int] = None,
            total_steps: Optional[int] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(backbone.out_channels, proj_hidden_dim),
            nn.LayerNorm(proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.LayerNorm(proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_out_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_out_dim, pred_hidden_dim),
            nn.LayerNorm(pred_hidden_dim),
            nn.GELU(),
            nn.Linear(pred_hidden_dim, pred_hidden_dim),
            nn.LayerNorm(pred_hidden_dim),
            nn.GELU(),
            nn.Linear(pred_hidden_dim, proj_out_dim)
        )

        self.momentum_backbone = deepcopy(self.backbone)
        for param in self.momentum_backbone.parameters():
            param.requires_grad = False
        self.momentum_projector = deepcopy(self.projector)
        for param in self.momentum_projector.parameters():
            param.requires_grad = False
        self.momentum_updater = MomentumUpdater(tau)

        self.register_buffer('target_embeds_queue', torch.randn(queue_size, proj_out_dim))
        self.target_embeds_queue = F.normalize(self.target_embeds_queue)

        self.temp = temp
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def on_train_start(self):
        self.last_step = 0

    def training_step(self, batch, batch_idx):
        batch = batch['pretrain']

        (target_images_batch, target_voxel_indices_batch,
         context_images_batch, context_masks_batch, context_voxel_indices_batch) = batch

        with torch.no_grad():
            target_feature_maps_batch = self.momentum_backbone(target_images_batch)
            target_features = batched_take_features_from_map(
                target_feature_maps_batch,
                target_voxel_indices_batch,
                stride=self.momentum_backbone.stem_stride,
                mode='trilinear'
            )
            target_embeds = F.normalize(self.momentum_projector(target_features))  # (n, d)

            assert len(self.target_embeds_queue) > len(target_embeds)

            self.target_embeds_queue = torch.cat([target_embeds, self.target_embeds_queue[:-len(target_embeds)]])

        context_feature_maps_batch = self.backbone(context_images_batch, context_masks_batch)
        context_features = batched_take_features_from_map(
            context_feature_maps_batch,
            context_voxel_indices_batch,
            stride=self.backbone.stem_stride,
            mode='trilinear'
        )
        context_embeds = F.normalize(self.predictor(self.projector(context_features)))  # (n, d)

        assert len(context_embeds) == len(target_embeds)

        logits = torch.matmul(context_embeds, self.target_embeds_queue.T) / self.temp  # (n, queue_size)
        targets = torch.arange(len(context_embeds), device=self.device)

        loss = F.cross_entropy(logits, targets)
        self.log(f'moco/loss', loss, on_epoch=True, on_step=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.trainer.global_step > self.last_step:
            # update momentum backbone and projector
            self.momentum_updater.update(self.backbone, self.momentum_backbone)
            self.momentum_updater.update(self.projector, self.momentum_projector)
            # log tau momentum
            # self.log("tau", self.momentum_updater.cur_tau)
            # update tau
            # self.momentum_updater.update_tau(
            #     cur_step=self.trainer.global_step,
            #     max_steps=self.trainer.estimated_stepping_batches,
            # )
        self.last_step = self.trainer.global_step

    def validation_step(self, batch, batch_idx):
        pass

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
            # skip device transfer for the val dataloader
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
