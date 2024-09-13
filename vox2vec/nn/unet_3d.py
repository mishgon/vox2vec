from typing import Union, Tuple, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medimm.fpn_3d import FPN3d, FPNLinearDenseHead3d, crop_and_pad_to
from medimm.layers.norm import LayerNorm3d


class UNet3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            stem_stride: Union[int, Tuple[int, int, int]],
            out_channels: int,
            fpn_out_channels: Sequence[int],
            fpn_hidden_factors: Union[float, Sequence[float]],
            fpn_depths: Sequence[Union[int, Tuple[int, int]]],
            stem_kernel_size: Optional[Union[int, Tuple[int, int, int]]] = None,
            stem_padding: Optional[Union[int, Tuple[int, int, int]]] = None,
            drop_path_rate: float = 0.0,
            final_ln: bool = False,
            final_affine: bool = False,
            final_gelu: bool = False,
            mask_token: bool = False,
            **convnext_block_kwargs
    ):
        super().__init__()

        if isinstance(stem_stride, int):
            stem_stride = (stem_stride, stem_stride, stem_stride)
        stem_stride = tuple(stem_stride)

        self.fpn = FPN3d(
            in_channels=in_channels,
            stem_stride=stem_stride,
            out_channels=fpn_out_channels,
            hidden_factors=fpn_hidden_factors,
            depths=fpn_depths,
            stem_kernel_size=stem_kernel_size,
            stem_padding=stem_padding,
            drop_path_rate=drop_path_rate,
            final_ln=True,
            final_affine=True,
            final_gelu=True,
            mask_token=mask_token,
            **convnext_block_kwargs
        )
        self.head = FPNLinearDenseHead3d(
            out_channels=out_channels,
            fpn_stem_stride=stem_stride,
            fpn_out_channels=fpn_out_channels
        )
        self.final_norm = LayerNorm3d(out_channels, affine=final_affine) if final_ln else nn.Identity()
        self.final_act = nn.GELU() if final_gelu else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stem_stride = stem_stride
        self.fpn_out_channels = fpn_out_channels

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.final_act(self.final_norm(self.head(image, self.fpn(image, mask), upsample=False)))


class UNetDenseHead3d(nn.Module):
    def __init__(
            self,
            out_channels: int,
            unet_stem_stride: int,
            unet_out_channels: Sequence[int],
            hidden_channels: int,
    ):
        super().__init__()

        if isinstance(unet_stem_stride, int):
            unet_stem_stride = (unet_stem_stride, unet_stem_stride, unet_stem_stride)
        self.unet_stem_stride = tuple(unet_stem_stride)

        self.layers = nn.Sequential(
            nn.Conv3d(unet_out_channels, hidden_channels, kernel_size=1),
            LayerNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            LayerNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            LayerNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1)
        )

    def forward(
            self,
            image: torch.Tensor,
            feature_map: torch.Tensor,
            upsample: bool = True
    ) -> torch.Tensor:
        x = self.layers(feature_map)

        if not upsample:
            return x

        # upsample and pad logits to the original images' spatial resolution
        x = F.interpolate(x, scale_factor=self.unet_stem_stride, mode='trilinear')
        if x.shape[2:] != image.shape[2:]:
            x = crop_and_pad_to(x, image)

        return x


class UNetLinearDenseHead3d(nn.Module):
    def __init__(
            self,
            out_channels: int,
            unet_stem_stride: int,
            unet_out_channels: Sequence[int],
    ):
        super().__init__()

        if isinstance(unet_stem_stride, int):
            unet_stem_stride = (unet_stem_stride, unet_stem_stride, unet_stem_stride)
        self.unet_stem_stride = tuple(unet_stem_stride)

        self.conv = nn.Conv3d(unet_out_channels, out_channels, kernel_size=1)

    def forward(
            self,
            image: torch.Tensor,
            feature_map: torch.Tensor,
            upsample: bool = True
    ) -> torch.Tensor:
        x = self.conv(feature_map)

        if not upsample:
            return x

        # upsample and pad logits to the original images' spatial resolution
        x = F.interpolate(x, scale_factor=self.unet_stem_stride, mode='trilinear')
        if x.shape[2:] != image.shape[2:]:
            x = crop_and_pad_to(x, image)

        return x
