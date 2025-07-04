import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class ImageSplitter:
    def __init__(self, patch_size: int | tuple[int], num_channels: int):
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.rearange = Rearrange(
            "b (c pw ph) L -> (b L) c pw ph",
            c=self.num_channels,
            pw=self.patch_size,
            ph=self.patch_size,
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = self.unfold(image)
        image = self.rearange(image)
        return image


def custom_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for DSWrapper dataset.

    Args:
        batch: List of tuples (image_patches, mask_patches) where each has shape (topk, ...)

    Returns:
        Tuple of (batched_images, batched_masks) with shape (batch_size * topk, ...)
    """
    # Separate images and masks
    images, masks = zip(*batch)

    # Stack along a new dimension to create (batch_size, topk, ...)
    # Then reshape to (batch_size * topk, ...)
    batched_images = torch.cat(images, dim=0)
    batched_masks = torch.cat(masks, dim=0)

    return batched_images, batched_masks
