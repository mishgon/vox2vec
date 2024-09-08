from typing import Optional
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

import lightning.pytorch as pl

from medimm.layers.norm import LayerNorm3d
from medimm.fpn_3d import FPN3d, ConvNeXtStage3d

from vox2vec.nn.functional import batched_take_features_from_map
from .ema import MomentumUpdater


class Projector(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            out_dim: int,
            hidden_factor: float = 4.0
    ):
        super().__init__()

        hidden_dim = int(embed_dim * hidden_factor)
        self.layers = nn.Sequential(
            nn.Conv3d(embed_dim, hidden_dim, kernel_size=1),
            LayerNorm3d(hidden_dim),  # nn.BatchNorm3d(hidden_dim),
            nn.GELU(),  # nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1),
            LayerNorm3d(hidden_dim),  # nn.BatchNorm3d(hidden_dim),
            nn.GELU(),  # nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, out_dim, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Predictor(nn.Module):
    def __init__(
            self,
            proj_out_dim: int,
            hidden_factor: float = 1.0,
            depth: int = 3,
            kernel_size: int = 5,
            mask_token: bool = True,
    ):
        super().__init__()

        hidden_dim = int(proj_out_dim * hidden_factor)
        self.in_conv = nn.Conv3d(proj_out_dim, hidden_dim, kernel_size=1)
        self.norm = LayerNorm3d(hidden_dim)
        self.convnext_blocks = ConvNeXtStage3d(hidden_dim, depth, kernel_size=kernel_size)
        self.out_conv = nn.Conv3d(hidden_dim, proj_out_dim, kernel_size=1)
        if mask_token:
            self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
            nn.init.trunc_normal_(self.mask_token)
        else:
            self.mask_token = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        mask = mask.unsqueeze(1)

        x = self.in_conv(x)

        if self.mask_token is not None:
            fill_values = self.mask_token.view(-1, 1, 1, 1)
        else:
            fill_values = torch.randn_like(x)
        x = x * mask + fill_values * (1 - mask)

        x = self.norm(x)
        x = self.convnext_blocks(x)
        x = self.out_conv(x)

        return x


class MIM(pl.LightningModule):
    def __init__(
            self,
            backbone: FPN3d,
            proj_hidden_factor: float = 4.0,
            proj_out_dim: int = 128,
            pred_hidden_factor: float = 1.0,
            pred_depth: int = 3,
            pred_kernel_size: int = 5,
            base_tau: float = 0.996,
            final_tau: float = 1.0,
            lr: float = 3e-4,
            weight_decay: float = 1e-6,
            warmup_steps: Optional[int] = None,
            total_steps: Optional[int] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.num_scales = len(backbone.out_channels)
        self.projectors = nn.ModuleList([
            Projector(backbone.out_channels[j], proj_out_dim, proj_hidden_factor)
            for j in range(self.num_scales)
        ])
        self.predictors = nn.ModuleList([
            Predictor(proj_out_dim, pred_hidden_factor, pred_depth, pred_kernel_size)
            for _ in range(self.num_scales)
        ])

        self.momentum_backbone = deepcopy(self.backbone)
        for param in self.momentum_backbone.parameters():
            param.requires_grad = False
        self.momentum_projectors = deepcopy(self.projectors)
        for param in self.momentum_projectors.parameters():
            param.requires_grad = False
        self.momentum_updater = MomentumUpdater(base_tau, final_tau)

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def forward(self, x):
        return self.backbone(x)

    def on_train_start(self):
        self.last_step = 0

    def training_step(self, batch, batch_idx):
        batch = batch['pretrain']

        (target_images, _, target_masked_token_indices,
         context_images, context_image_masks, context_tokens_masks, context_masked_token_indices) = batch

        with torch.no_grad():
            target_feature_pyramids = self.momentum_backbone(target_images)
            target_embed_pyramids = [self.momentum_projectors[j](target_feature_pyramids[j])
                                     for j in range(self.num_scales)]

        loss = 0.0
        for j in range(self.num_scales):
            target_embeds = batched_take_features_from_map(
                feature_maps_batch=target_embed_pyramids[j],
                voxel_indices_batch=target_masked_token_indices[j],
            )
            target_embeds = F.normalize(target_embeds)

            context_feature_pyramids = self.backbone(context_images[j], context_image_masks[j])
            context_embed_pyramids = [self.projectors[j](context_feature_pyramids[j])
                                      for j in range(self.num_scales)]
            pred_embed_pyramids = [self.predictors[j](context_embed_pyramids[j], context_tokens_masks[j])
                                   for j in range(self.num_scales)]
            pred_embeds = batched_take_features_from_map(
                feature_maps_batch=pred_embed_pyramids[j],
                voxel_indices_batch=context_masked_token_indices[j],
            )
            pred_embeds = F.normalize(pred_embeds)

            byol_loss_j = torch.mean(2 - 2 * torch.sum(pred_embeds * target_embeds, dim=1))
            self.log(f'mim/byol_loss_at_scale_{j}', byol_loss_j, on_epoch=True, on_step=True)

            loss += byol_loss_j

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.trainer.global_step > self.last_step:
            # update momentum backbone and projector
            self.momentum_updater.update(self.backbone, self.momentum_backbone) 
            # log tau momentum
            self.log('tau', self.momentum_updater.cur_tau)
            # update tau
            self.momentum_updater.update_tau(
                cur_step=self.trainer.global_step,
                max_steps=self.trainer.estimated_stepping_batches,
            )
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
