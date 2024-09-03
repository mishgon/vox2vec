from typing import Optional
from pathlib import Path
import seaborn as sns
import colorcet as cc
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from torch import Tensor
import torch
from torchvision.utils import make_grid


def draw(
        image: Tensor,
        gt_mask: Tensor,
        pred_mask: Tensor,
        dim: int = 2,
        slice_idx: Optional[int] = None,
        alpha: float = 0.5,
) -> Tensor:
    """Generate an RGBA image, visualizing the ground truth and predicted masks.

    Args:
        image (Tensor):
            Tensor of size (1, h, w, d).
        gt_mask (Tensor):
            Tensor of size (num_classes, h, w, d).
        pred_mask (Tensor):
            Tensor of size (num_classes, h, w, d).
        dim (int):
            Axis along which predictions will be drawn. Can be in [0, 1, 2].
        slice_idx (Optional[int], optional):
            Slice index to visualize.
            Defaults to None, which means the slice is chosen automatically.
        alpha (float, optional):
            Mask transparence parameter. Defaults to 0.5.

    Returns:
        Tensor: tensor of size
    """
    gt_mask = gt_mask.bool()
    pred_mask = pred_mask.bool()

    c, h, w, d = gt_mask.shape

    if slice_idx is None:
        slice_idx = gt_mask.movedim(dim + 1, 1).flatten(-2).any(-1).sum(0).argmax()
    else:
        slice_idx = torch.as_tensor(slice_idx)

    # despite image is in grayscale, it requires 3 channels
    image = image.index_select(dim + 1, slice_idx).squeeze(dim + 1).repeat(3, 1, 1)  # (3, h, w)
    pred_mask = pred_mask.index_select(dim + 1, slice_idx).squeeze(dim + 1)  # (c, h, w)
    gt_mask = gt_mask.index_select(dim + 1, slice_idx).squeeze(dim + 1)

    num_classes = pred_mask.shape[0]
    palette = list(sns.color_palette(cc.glasbey, n_colors=num_classes))

    # draw segmentation masks for gt and pred
    pred_out = gt_out = image
    for class_idx, (pred_mask_class, gt_mask_class) in enumerate(zip(pred_mask, gt_mask)):
        color = Tensor(palette[class_idx])

        gt_masked_img = torch.where(gt_mask_class, color.view(3, 1, 1), gt_out)
        gt_out = gt_out * (1 - alpha) + gt_masked_img * alpha

        pred_masked_img = torch.where(pred_mask_class, color.view(3, 1, 1), pred_out)
        pred_out = pred_out * (1 - alpha) + pred_masked_img * alpha

    gt_out = (gt_out * 255).type(torch.uint8)
    pred_out = (pred_out * 255).type(torch.uint8)

    # create legend image from scratch
    _, height, width = image.shape
    legend_image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(legend_image)
    font = ImageFont.truetype(str(Path(__file__).parent / 'Roboto-Regular.ttf'), size=14)
    y_offset = 5
    for class_idx, color in enumerate(palette):
        draw.rectangle(
            (10, y_offset, 40, y_offset + 10),
            fill=tuple((np.array(color) * 255).astype(np.uint8))
        )
        draw.text((50, y_offset - 3), f'class_{class_idx}', font=font, fill='black')
        y_offset += 20

    # torch is not capable of constructing tensor from PIL image, as I understand
    legend_tensor = torch.from_numpy(np.array(legend_image)).permute(2, 0, 1)

    return make_grid([pred_out, gt_out, legend_tensor], nrow=3, padding=2)
