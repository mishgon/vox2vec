from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
import lightning.pytorch as pl

from vox2vec.nn.unet_3d import UNet3d, FPNLinearDenseHead3d
from vox2vec.nn.functional import batched_take_features_from_map


class SimCLRUNetKL(pl.LightningModule):
    def __init__(
            self,
            backbone: UNet3d,
            proj_hidden_dim: int = 2048,
            proj_out_dim: int = 128,
            temp: float = 0.1,
            beta: float = 1e-3,
            lr: float = 0.01,
            weight_decay: float = 1e-6,
            warmup_steps: Optional[int] = None,
            total_steps: Optional[int] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.logvar_head = FPNLinearDenseHead3d(
            out_channels=backbone.out_channels,
            fpn_stem_stride=backbone.stem_stride,
            fpn_out_channels=backbone.fpn_out_channels,
        )
        nn.init.constant_(self.logvar_head.convs[0].bias, -13.81551)
        self.projector = nn.Sequential(
            nn.Linear(backbone.out_channels, proj_hidden_dim),
            nn.LayerNorm(proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.LayerNorm(proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_out_dim, bias=False)
        )

        self.temp = temp
        self.beta = beta
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward_voxel_embeds(
            self,
            images_batch: torch.Tensor,
            masks_batch: torch.Tensor,
            voxel_indices_batch: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        feature_pyramids_batch = self.backbone.fpn(images_batch, masks_batch)
        mean_maps_batch = self.backbone.head(images_batch, feature_pyramids_batch, upsample=False)
        std_maps_batch = torch.exp(self.logvar_head(images_batch, feature_pyramids_batch, upsample=False) / 2.0) + 1e-5
        means = batched_take_features_from_map(
            mean_maps_batch,
            voxel_indices_batch,
            stride=self.backbone.stem_stride,
            mode='trilinear'
        )
        self.log('simclr/features_std', means.std(), on_step=True, on_epoch=True)
        stds = batched_take_features_from_map(
            std_maps_batch,
            voxel_indices_batch,
            stride=self.backbone.stem_stride,
            mode='trilinear'
        )
        q = D.Normal(means, stds)
        embeds = F.normalize(self.projector(q.rsample()))
        p = D.Normal(torch.zeros_like(means), torch.ones_like(stds))
        kl_reg = D.kl_divergence(q, p).mean()
        return embeds, kl_reg

    def training_step(self, batch, batch_idx):
        # batch = batch['pretrain']

        (images_batch_1, masks_batch_1, voxel_indices_batch_1,
         images_batch_2, masks_batch_2, voxel_indices_batch_2) = batch

        embeds_1, kl_reg_1 = self.forward_voxel_embeds(images_batch_1, masks_batch_1, voxel_indices_batch_1)
        embeds_2, kl_reg_2 = self.forward_voxel_embeds(images_batch_2, masks_batch_2, voxel_indices_batch_2)

        logits_11 = torch.matmul(embeds_1, embeds_1.T) / self.temp
        logits_11.fill_diagonal_(float('-inf'))
        logits_12 = torch.matmul(embeds_1, embeds_2.T) / self.temp
        logits_22 = torch.matmul(embeds_2, embeds_2.T) / self.temp
        logits_22.fill_diagonal_(float('-inf'))
        logits_1 = torch.cat([logits_12, logits_11], dim=1)
        logits_2 = torch.cat([logits_12.T, logits_22], dim=1)
        targets = torch.arange(len(logits_1), device=self.device)

        info_nce_loss = (F.cross_entropy(logits_1, targets) + F.cross_entropy(logits_2, targets)) / 2
        self.log(f'simclr/info_nce_loss', info_nce_loss, on_epoch=True, on_step=True)

        kl_reg = (kl_reg_1 + kl_reg_2) / 2
        self.log(f'simclr/kl_reg', kl_reg, on_epoch=True, on_step=True)

        loss = info_nce_loss + self.beta * kl_reg
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
