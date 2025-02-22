from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_segmentation_masks, make_grid


def make_images_with_masks(
    image: torch.Tensor, masks: torch.Tensor, save_path: str | Path
):
    assert (
        len(image.shape) == 4
    ), f"The image should be batched! It has shape {image.shape}, but shape B,C,H,W is expected!"
    assert (
        len(masks.shape) == 4
    ), f"The mask tesnor should be bathced! It has shape {masks.shape}, but shape B,num_masks,H,W is expected!"

    binary_masks = masks > 0.5
    batch_size = image.shape[0]
    masked_images = [
        draw_segmentation_masks(image[i], binary_masks[i], alpha=0.4)
        for i in range(batch_size)
    ]
    image_grid = make_grid(masked_images)
    plt.imshow(image_grid)
    plt.savefig(save_path)
