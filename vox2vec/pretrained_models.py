from typing import List, Literal

import torch
import torch.nn as nn

from medimm.fpn_3d import FPN3d
from vox2vec.nn.unet_3d import UNet3d
from vox2vec.nn.functional import eval_mode

from huggingface_hub import hf_hub_download


class Vox2VecForScreener(nn.Module):
    def __init__(
            self,
            name: Literal[
                'vicreg_dim32_nlst',
                'vicreg_dim32_all',
                'vicreg_dim32_0.1all',
                'vicreg_dim32_0.01all',
                'vicreg_dim128_all',
                'simclr_dim32_nlst',
                'simclr_dim32_all',
                'simclr_dim32_0.1all',
                'simclr_dim32_0.01all',
            ],
            revision: str = 'b141733e692867646c029414521d5616be0854d9'
    ):
        super().__init__()

        if name in ['simclr_dim128_nlst', 'vicreg_dim128_all']:
            self.out_channels = 128
        else:
            self.out_channels = 32

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
        self.name = name

        # weights_path = hf_hub_download(
        #     repo_id='mishgon/vox2vec',
        #     filename=f'{name}.pt',
        #     revision=revision
        # )
        # self.load_state_dict(torch.load(weights_path))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)
