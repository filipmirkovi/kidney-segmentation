from typing import Union, List, Tuple, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
from torchvision.utils import draw_segmentation_masks, make_grid

COLORS = ["blue", "orange", "green", "red", "purple"]


def make_images_with_masks(
    image: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    colors: list[str] = None,
    alpha: float = 0.6,
) -> plt.Figure:
    assert (
        len(image.shape) == 4
    ), f"The image should be batched! It has shape {image.shape}, but shape B,C,H,W is expected!"
    assert (
        len(masks.shape) == 4
    ), f"The mask tesnor should be bathced! It has shape {masks.shape}, but shape B,num_masks,H,W is expected!"

    binary_masks = masks > 0.5
    batch_size = image.shape[0]
    image = image.mul(0.5).add(0.5)
    colors = COLORS[:num_classes]
    masked_images = [
        draw_segmentation_masks(
            image[i], binary_masks[i][:num_classes], alpha=alpha, colors=colors
        )
        for i in range(batch_size)
    ]
    if batch_size <= 4:
        nrow = 2
    elif batch_size <= 9:
        nrow = 3
    else:
        nrow = 4

    image_grid = make_grid(masked_images, nrow=nrow)
    figure, ax = plt.subplots()

    ax.imshow(image_grid.permute(1, 2, 0))
    return figure


def visualize_segmentation_masks(
    target: Union[torch.Tensor, np.ndarray],
    prediction: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (15, 5),
    background_idx: Optional[int] = None,
) -> plt.Figure:
    batch, seg_regions, H, W = target.shape
    colors = COLORS[:seg_regions]
    mask_idx = [i for i in range(seg_regions) if i != background_idx]
    canvas = torch.zeros((3, H, W))
    target_bitmap = target > 0.5
    prediction_bitmap = prediction > 0.5
    target_segmap = []
    prediction_segmap = []
    print(target.shape, target_bitmap.shape, batch)
    for i in range(batch):
        target_segmap.append(
            draw_segmentation_masks(
                canvas, target_bitmap[i, mask_idx], alpha=alpha, colors=colors
            )
        )
        prediction_segmap.append(
            draw_segmentation_masks(
                canvas, prediction_bitmap[i, mask_idx], alpha=alpha, colors=colors
            )
        )

    target_grid = make_grid(target_segmap)
    prediction_grid = make_grid(prediction_segmap)
    figure, ax = plt.subplots(2, 1, figsize=figsize)

    ax[0].imshow(target_grid.permute(1, 2, 0))
    ax[1].imshow(prediction_grid.permute(1, 2, 0))
    return figure
