from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from medimm.fpn_3d import FPN3d
from vox2vec.nn.unet_3d import UNet3d
from vox2vec.nn.functional import eval_mode

from huggingface_hub import hf_hub_download


class Vox2VecForScreener(nn.Module):
    def __init__(
            self,
            name: Literal[
                'lowres_all',
                'lowres_0.1all',
                'lowres_0.01all',
                'lowres_0.001all',
                'lowres_0.0001all',
                'lowres_nlst',
            ],
            revision: str = '01e9c1d0cb8ac8a8967f4ae3a4f2e72faaf54334'
    ):
        super().__init__()

        self.name = name
        if 'lowres' in name:
            self.backbone = UNet3d(
                in_channels=1,
                stem_stride=(3, 3, 2),
                out_channels=96,
                fpn_out_channels=(96, 192, 384, 768),
                fpn_hidden_factors=4.0,
                fpn_depths=((3, 1), (3, 1), (9, 1), 3),
                stem_kernel_size=(3, 3, 2),
                stem_padding=0,
                final_ln=False,
                final_affine=False,
                final_gelu=False,
                mask_token=True
            )
        else:
            raise NotImplementedError

        self.out_channels = self.backbone.out_channels

        weights_path = hf_hub_download(
            repo_id='mishgon/vox2vec',
            filename=f'{name}.pt',
            revision=revision
        )
        self.load_state_dict(torch.load(weights_path))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)


class ConditionerForScreener(nn.Module):
    def __init__(
            self,
            # revision: str = 'b229a25fc48cc73f6eb7a3582dee36a308046f7f'
    ):
        super().__init__()

        self.backbone = UNet3d(
            in_channels=1,
            stem_stride=(3, 3, 2),
            out_channels=96,
            fpn_out_channels=(96, 192, 384, 768),
            fpn_hidden_factors=4.0,
            fpn_depths=((3, 1), (3, 1), (9, 1), 3),
            stem_kernel_size=(3, 3, 2),
            stem_padding=0,
            final_ln=False,
            final_affine=False,
            final_gelu=False,
            mask_token=True
        )
        self.prototypes = nn.Parameter(torch.zeros(512, 96))

        # weights_path = hf_hub_download(
        #     repo_id='mishgon/vox2vec',
        #     filename=f'{name}.pt',
        #     revision=revision
        # )
        # self.load_state_dict(torch.load(weights_path))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.backbone(x)
        x = F.normalize(x, dim=1)
        x = x.movedim(1, -1)
        prototypes = F.normalize(prototypes, dim=1)
        return (x @ prototypes.T).argmax(dim=-1)
