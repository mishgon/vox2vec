from typing import Optional, Sequence
import math

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

from vox2vec.nn.unet_3d import UNet3d
from vox2vec.nn.functional import batched_take_features_from_map


class SwAVUNet(pl.LightningModule):
    def __init__(
            self,
            backbone: UNet3d,
            num_prototypes: int = 512,
            temp: float = 0.1,
            sharpen_temp: float = 0.25,
            num_sinkhorn_iters: int = 3,
            memax_weight: float = 0.0,
            lr: float = 0.01,
            weight_decay: float = 1e-6,
            warmup_steps: Optional[int] = None,
            total_steps: Optional[int] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore='backbone')

        self.backbone = backbone
        self.projector = nn.Identity()  # NOTE
        prototype_dim = backbone.out_channels
        self.prototypes = nn.Parameter(torch.zeros(num_prototypes, prototype_dim))
        nn.init.uniform_(self.prototypes, -(1. / prototype_dim) ** 0.5, (1. / prototype_dim) ** 0.5)

        self.num_prototypes = num_prototypes
        self.temp = temp
        self.sharpen_temp = sharpen_temp
        self.num_sinkhorn_iters = num_sinkhorn_iters
        self.memax_weight = memax_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def forward(self, patches):
        return self.backbone(patches)

    def forward_voxel_embeds(
            self,
            images_batch: torch.Tensor,
            masks_batch: torch.Tensor,
            voxel_indices_batch: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        feature_maps_batch = self.backbone(images_batch, masks_batch)
        features = batched_take_features_from_map(
            feature_maps_batch,
            voxel_indices_batch,
            stride=self.backbone.stem_stride,
            mode='trilinear'
        )
        embeds = F.normalize(self.projector(features))
        prototypes = F.normalize(self.prototypes)
        logits = embeds @ prototypes.T / self.temp
        targets = torch.softmax(logits.detach() / self.sharpen_temp, dim=1)
        if self.num_sinkhorn_iters > 0:
            targets = self._sinkhorn(targets)
        probs = torch.softmax(logits, dim=1)
        memax_reg = math.log(self.num_prototypes) - entropy(probs.mean(dim=0), dim=0)
        return logits, targets, memax_reg

    def training_step(self, batch, batch_idx):
        # batch = batch['pretrain']

        (images_batch_1, masks_batch_1, voxel_indices_batch_1,
         images_batch_2, masks_batch_2, voxel_indices_batch_2) = batch

        logits_1, targets_1, memax_reg_1 = self.forward_voxel_embeds(
            images_batch_1, masks_batch_1, voxel_indices_batch_1
        )
        logits_2, targets_2, memax_reg_2 = self.forward_voxel_embeds(
            images_batch_2, masks_batch_2, voxel_indices_batch_2
        )

        bootstrap_loss_1 = F.cross_entropy(logits_1, targets_2)
        bootstrap_loss_2 = F.cross_entropy(logits_2, targets_1)
        bootstrap_loss = (bootstrap_loss_1 + bootstrap_loss_2) / 2
        self.log('swav/bootstrap_loss', bootstrap_loss, on_epoch=True, on_step=True)

        memax_reg = (memax_reg_1 + memax_reg_2) / 2
        self.log('swav/memax_reg', memax_reg, on_epoch=True, on_step=True)

        loss = bootstrap_loss + self.memax_weight * memax_reg
        self.log('swav/loss', loss, on_epoch=True, on_step=True)

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

    @torch.no_grad()
    def _sinkhorn(self, probas: torch.Tensor) -> torch.Tensor:
        batch_size, num_prototypes = probas.shape
        probas = probas / probas.sum()

        for _ in range(self.num_sinkhorn_iters):
            probas /= probas.sum(dim=0)
            probas /= num_prototypes

            probas /= probas.sum(dim=1, keepdim=True)
            probas /= batch_size

        probas *= batch_size
        return probas


def entropy(p: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.sum(torch.log(p ** (-p)), dim=dim)
