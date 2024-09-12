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
                'moco_dim32_nlst',
                'simclr_dim32_nlst',
                'moco_dim128_nlst',
                'moco_dim32_0.1nlst',
                'moco_dim32_0.01nlst',
                'moco_dim32_all'
            ],
            revision: str = '44add8acb1d9d307d7ea31c702ae20a0a6d5f80e'
    ):
        super().__init__()

        self.out_channels = int(name.split('_')[1][len('dim'):])
        self.backbone = UNet3d(
            in_channels=1,
            stem_stride=1,
            out_channels=self.out_channels,
            fpn_out_channels=(32, 64, 128, 256),
            fpn_depths=((1, 1), (2, 1), (4, 1), 8),
            stem_kernel_size=7,
            stem_padding=3,
            final_ln=False,
            final_affine=False,
            final_gelu=False,
            mask_token=True
        )
        self.name = name

        weights_path = hf_hub_download(
            repo_id='mishgon/vox2vec',
            filename=f'{name}.pt',
            revision=revision
        )
        self.load_state_dict(torch.load(weights_path))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)
