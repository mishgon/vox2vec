from typing import Union, Sequence, Callable, Literal, Tuple, Optional
from contextlib import contextmanager
import itertools
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def eval_mode(module: nn.Module, enable_dropout: bool = False):
    """Copypasted from pl_bolts.callbacks.ssl_online.set_training
    """
    original_mode = module.training

    try:
        module.eval()
        if enable_dropout:
            for m in module.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
        yield module
    finally:
        module.train(original_mode) 


def binary_dice_loss(
        pred_probs: torch.Tensor,
        gt_masks: torch.Tensor,
        roi_masks: Optional[torch.Tensor] = None,
        reduction: Literal['mean', 'none'] = 'mean'
) -> torch.Tensor:
    """Dice loss.

    Args:
        pred_probs (torch.Tensor):
            Tensor of size (N, C, H, W, ...).
        gt_masks (torch.Tensor):
            Tensor of size (N, C, H, W, ...).
        spatial_dims (Union[int, Sequence[int]], optional):
            Which dims correspond to image spatial sizes. Defaults to (2, 3).
        reduce (Callable, optional):
            Func to reduce loss tensor of size (N, C). Defaults to torch.mean.

    Returns:
        torch.Tensor: Dice loss.
    """
    if roi_masks is not None:
        roi_masks = roi_masks.unsqueeze(1)
        pred_probs = torch.where(roi_masks, pred_probs, torch.zeros_like(pred_probs))
        gt_masks = torch.where(roi_masks, gt_masks, torch.zeros_like(gt_masks))
    intersection = torch.sum(pred_probs * gt_masks, dim=(-3, -2, -1))
    volumes_sum = torch.sum(pred_probs ** 2 + gt_masks ** 2, dim=(-3, -2, -1))
    dice_losses = 1 - 2 * intersection / (volumes_sum + 1)
    if reduction == 'mean':
        return torch.mean(dice_losses)
    elif reduction == 'none':
        return dice_losses
    else:
        raise ValueError(reduction)


def segmentation_bce_loss(
        pred_logits: torch.Tensor,
        gt_masks: torch.Tensor,
        roi_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if roi_masks is not None:
        pred_logits = pred_logits.movedim(1, -1)[roi_masks]
        gt_masks = gt_masks.movedim(1, -1)[roi_masks]
    return F.binary_cross_entropy_with_logits(pred_logits, gt_masks)


def binary_soft_dice_score(
        pred_probs: torch.Tensor,
        gt_masks: torch.Tensor,
        reduction: Literal['mean', 'none'] = 'none'
) -> torch.Tensor:
    intersection = torch.sum(pred_probs * gt_masks, dim=(-3, -2, -1))
    volumes_sum = torch.sum(pred_probs ** 2 + gt_masks ** 2, dim=(-3, -2, -1))
    eps = 1e-5
    dice_scores = (2 * intersection + eps) / (volumes_sum + eps)
    if reduction == 'mean':
        return torch.mean(dice_scores)
    elif reduction == 'none':
        return dice_scores
    else:
        raise ValueError(reduction)


def take_features_from_map(
        feature_map: torch.Tensor,
        voxel_indices: torch.Tensor,
        stride: Union[int, Tuple[int, int, int]] = 1,
        mode: Literal['nearest', 'trilinear'] = 'trilinear',
) -> torch.Tensor:
    if stride == 1:
        return feature_map.movedim(0, -1)[voxel_indices.unbind(1)]

    stride = torch.tensor(stride).to(voxel_indices)
    min_indices = torch.tensor(0).to(voxel_indices)
    max_indices = torch.tensor(feature_map.shape[-3:]).to(voxel_indices) - 1
    if mode == 'nearest':
        indices = voxel_indices // stride
        indices = torch.clamp(indices, min_indices, max_indices)
        return feature_map.movedim(0, -1)[indices.unbind(1)]
    elif mode == 'trilinear':
        x = feature_map.movedim(0, -1)
        points = (voxel_indices + 0.5) / stride - 0.5
        starts = torch.floor(points).long()  # (n, 3)
        stops = starts + 1  # (n, 3)
        f = 0.0
        for mask in itertools.product((0, 1), repeat=3):
            mask = torch.tensor(mask, device=voxel_indices.device, dtype=bool)
            corners = torch.where(mask, starts, stops)  # (n, 3)
            corners = torch.clamp(corners, min_indices, max_indices)  # (n, 3)
            weights = torch.prod(torch.where(mask, 1 - (points - starts), 1 - (stops - points)), dim=-1, keepdim=True)  # (n, 1)
            f = f + weights.to(x) * x[corners.unbind(-1)]  # (n, d)
        return f
    else:
        raise ValueError(mode)


def batched_take_features_from_map(
        feature_maps_batch: torch.Tensor,
        voxel_indices_batch: Sequence[torch.Tensor],
        stride: Union[int, Tuple[int, int, int]] = 1,
        mode: Literal['nearest', 'trilinear'] = 'trilinear',
) -> torch.Tensor:
    return torch.cat([
        take_features_from_map(feature_map, voxel_indices, stride, mode)
        for feature_map, voxel_indices in zip(feature_maps_batch, voxel_indices_batch, strict=True)
    ])


def take_features_from_pyramid(
        feature_pyramid: Sequence[torch.Tensor],
        voxel_indices: torch.Tensor,
        stride: Union[int, Tuple[int, int, int]] = 1,
        mode: Literal['nearest', 'trilinear'] = 'trilinear',
) -> torch.Tensor:
    """Select features from feature pyramid by their indices w.r.t. base feature map.

    Args:
        feature_pyramid (Sequence[torch.Tensor]): Sequence of tensors of shapes ``(c_i, h_i, w_i, d_i)``.
        voxels (torch.Tensor): tensor of shape ``(n, 3)``

    Returns:
        torch.Tensor: tensor of shape ``(n, \sum_i c_i)``
    """
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    assert isinstance(stride, tuple)

    features = []
    for i, x in enumerate(feature_pyramid):
        stride_i = tuple(s * 2 ** i for s in stride)
        features.append(take_features_from_map(x, voxel_indices, stride_i, mode))

    return torch.cat(features, dim=-1)


def batched_take_features_from_pyramid(
        feature_pyramids_batch: torch.Tensor,
        voxel_indices_batch: Sequence[torch.Tensor],
        stride: Union[int, Tuple[int, int, int]] = 1,
        mode: Literal['nearest', 'trilinear'] = 'trilinear',
) -> torch.Tensor:
    return torch.cat([
        take_features_from_pyramid([x[i] for x in feature_pyramids_batch], voxel_indices, stride, mode)
        for i, voxel_indices in enumerate(voxel_indices_batch)
    ])
