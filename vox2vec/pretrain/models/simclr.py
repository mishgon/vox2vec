from typing import Optional, Sequence, List

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

from medimm.fpn_3d import FPN3d

from vox2vec.nn.functional import batched_take_features_from_pyramid


class SimCLR(pl.LightningModule):
    def __init__(
            self,
            backbone: FPN3d,
            proj_hidden_dim: int = 512,
            proj_out_dim: int = 128,
            temp: float = 0.1,
            lr: float = 3e-4,
            weight_decay: float = 1e-6,
            warmup_steps: Optional[int] = None,
            total_steps: Optional[int] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(sum(backbone.out_channels), proj_hidden_dim),
            nn.LayerNorm(proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.LayerNorm(proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_out_dim)
        )

        self.temp = temp
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(images)

    def forward_voxel_embeds(
            self,
            images_batch: torch.Tensor,
            voxel_indices_batch: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        feature_pyramids_batch = self.backbone(images_batch)
        features = batched_take_features_from_pyramid(
            feature_pyramids_batch,
            voxel_indices_batch,
            stride=self.backbone.stem_stride,
            mode='trilinear'
        )
        embeds = F.normalize(self.projector(features))
        return embeds

    def training_step(self, batch, batch_idx):
        batch = batch['pretrain']

        images_batch_1, voxel_indices_batch_1, images_batch_2, voxel_indices_batch_2 = batch

        embeds_1 = self.forward_voxel_embeds(images_batch_1, voxel_indices_batch_1)
        embeds_2 = self.forward_voxel_embeds(images_batch_2, voxel_indices_batch_2)

        logits_11 = torch.matmul(embeds_1, embeds_1.T) / self.temp
        logits_11.fill_diagonal_(float('-inf'))
        logits_12 = torch.matmul(embeds_1, embeds_2.T) / self.temp
        logits_22 = torch.matmul(embeds_2, embeds_2.T) / self.temp
        logits_22.fill_diagonal_(float('-inf'))
        logits_1 = torch.cat([logits_12, logits_11], dim=1)
        logits_2 = torch.cat([logits_12.T, logits_22], dim=1)
        targets = torch.arange(len(logits_1), device=self.device)

        loss = (F.cross_entropy(logits_1, targets) + F.cross_entropy(logits_2, targets)) / 2
        self.log(f'simclr/loss', loss, on_epoch=True, on_step=True)

        return loss

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
