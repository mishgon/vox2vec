from typing import Optional, Sequence, List

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

from medimm.fpn_3d import FPN3d

from vox2vec.nn.functional import batched_take_features_from_pyramid


class VICReg(pl.LightningModule):
    def __init__(
            self,
            backbone: FPN3d,
            proj_hidden_dim: int = 8192,
            proj_out_dim: int = 8192,
            i_weight: float = 25.0,
            v_weight: float = 25.0,
            c_weight: float = 1.0,
            lr: float = 0.001,
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
            nn.Linear(proj_hidden_dim, proj_out_dim, bias=False)
        )

        self.i_weight = i_weight
        self.v_weight = v_weight
        self.c_weight = c_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(images)

    def forward_voxel_embeds(
            self,
            images_batch: torch.Tensor,
            masks_batch: torch.Tensor,
            voxel_indices_batch: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        feature_pyramids_batch = self.backbone(images_batch, masks_batch)
        features = batched_take_features_from_pyramid(
            feature_pyramids_batch,
            voxel_indices_batch,
            stride=self.backbone.stem_stride,
            mode='trilinear'
        )
        embeds = self.projector(features)
        return embeds

    def training_step(self, batch, batch_idx):
        batch = batch['pretrain']

        (images_batch_1, masks_batch_1, voxel_indices_batch_1,
         images_batch_2, masks_batch_2, voxel_indices_batch_2) = batch

        embeds_1 = self.forward_voxel_embeds(images_batch_1, masks_batch_1, voxel_indices_batch_1)
        embeds_2 = self.forward_voxel_embeds(images_batch_2, masks_batch_2, voxel_indices_batch_2)
        n, d = embeds_1.shape

        i_reg = F.mse_loss(embeds_1, embeds_2)
        self.log(f'vicreg/i_reg', i_reg, on_epoch=True, on_step=True)

        embeds_1 = embeds_1 - embeds_1.mean(dim=0)
        embeds_2 = embeds_2 - embeds_2.mean(dim=0)

        eps = 1e-4
        v_reg_1 = torch.mean(F.relu(1 - torch.sqrt(embeds_1.var(dim=0) + eps)))
        v_reg_2 = torch.mean(F.relu(1 - torch.sqrt(embeds_2.var(dim=0) + eps)))
        v_reg = (v_reg_1 + v_reg_2) / 2
        self.log(f'vicreg/v_reg', v_reg, on_epoch=True, on_step=True)

        c_reg_1 = off_diagonal(embeds_1.T @ embeds_1).div(n - 1).pow_(2).sum().div(d)
        c_reg_2 = off_diagonal(embeds_2.T @ embeds_2).div(n - 1).pow_(2).sum().div(d)
        c_reg = (c_reg_1 + c_reg_2) / 2
        self.log(f'vicreg/c_reg', c_reg, on_epoch=True, on_step=True)

        loss = (
            self.i_weight * i_reg
            + self.v_weight * v_reg
            + self.c_weight * c_reg
        )
        self.log(f'vicreg/loss', loss, on_epoch=True, on_step=True)

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


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Taken from https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/main_vicreg.py#L239.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
