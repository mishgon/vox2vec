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
                # 
                'fullres_all',
                'fullres_0.1all',
                'fullres_0.01all',
                'fullres_0.001all',
                'fullres_0.0001all',
                'fullres_nlst',
                # 
                'vicreg_fullres_all',
                'vicreg_fullres_0.1all',
                'vicreg_fullres_0.01all',
                'vicreg_fullres_nlst',
                'vicreg_fullres_dim128_all',
                # 
                'vicreg_fullres_all_lr0.001',
                'vicreg_fullres_0.1all_lr0.001',
                'vicreg_fullres_0.01all_lr0.001',
                'vicreg_fullres_0.001all_lr0.001',
                'vicreg_fullres_0.0001all_lr0.001',
                'vicreg_fullres_nlst_lr0.001',
                'vicreg_fullres_dim128_all_lr0.001',
            ],
            revision: str = 'e0d56f30a8d8bfb61bc58acf2d38e5ee85ee22cf'
    ):
        super().__init__()

        self.name = name

        if 'lowres' in name:
            self.out_channels = 96
        elif 'dim128' in name:
            self.out_channels = 128
        else:
            self.out_channels = 32

        if 'lowres' in name:
            self.backbone = UNet3d(
                in_channels=1,
                stem_stride=(3, 3, 2),
                out_channels=self.out_channels,
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
            self.backbone = UNet3d(
                in_channels=1,
                stem_stride=1,
                out_channels=self.out_channels,
                fpn_out_channels=(16, 64, 256, 1024),
                fpn_hidden_factors=(1.0, 1.0, 4.0, 4.0),
                fpn_depths=((1, 1), (2, 1), (4, 1), 8),
                stem_kernel_size=7,
                stem_padding=3,
                final_ln=False,
                final_affine=False,
                final_gelu=False,
                mask_token=True
            )

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
