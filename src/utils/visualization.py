from typing import Union, List, Tuple, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
from torchvision.utils import draw_segmentation_masks, make_grid


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
    colors = ["blue", "orange", "green", "red", "purple"]
    colors = colors[:num_classes]
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
    image: Optional[Union[torch.Tensor, np.ndarray]] = None,
    class_names: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (15, 5),
    show_difference: bool = True,
    background_idx: Optional[int] = None,
) -> plt.Figure:
    """
    Visualize target and predicted segmentation masks side by side.

    Args:
        target: Ground truth mask (H, W) or (C, H, W) for multi-class
        prediction: Predicted mask (H, W) or (C, H, W) for multi-class
        image: Optional original image (3, H, W) or (H, W, 3) to overlay masks on
        class_names: List of class names for legend
        colors: List of colors for each class (hex or named colors)
        alpha: Transparency for mask overlay (0-1)
        figsize: Figure size (width, height)
        save_path: Path to save the visualization
        show_difference: Whether to show a difference plot
        background_idx: Index of background class (will be transparent)

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy arrays
    target = _to_numpy(target)
    prediction = _to_numpy(prediction)

    # Convert multi-channel masks to single channel if needed
    if target.ndim == 3:
        target = np.argmax(target, axis=0)
    if prediction.ndim == 3:
        prediction = np.argmax(prediction, axis=0)

    # Handle image preprocessing
    if image is not None:
        image = _to_numpy(image)
        if image.ndim == 3 and image.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
            image = np.transpose(image, (1, 2, 0))
        # Normalize image to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0

    # Get unique classes
    unique_classes = np.unique(np.concatenate([target.flatten(), prediction.flatten()]))
    num_classes = len(unique_classes)

    # Setup colors
    if colors is None:
        colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    else:
        colors = [plt.cm.colors.to_rgba(c) for c in colors]

    # Create colormap
    cmap = ListedColormap(colors)

    # Create figure
    num_plots = 3 if show_difference else 2
    if image is not None:
        num_plots += 1

    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot original image if provided
    if image is not None:
        axes[plot_idx].imshow(image)
        axes[plot_idx].set_title("Original Image")
        axes[plot_idx].axis("off")
        plot_idx += 1

    # Plot target mask
    _plot_mask(axes[plot_idx], target, cmap, image, alpha, background_idx)
    axes[plot_idx].set_title("Target Mask")
    axes[plot_idx].axis("off")
    plot_idx += 1

    # Plot prediction mask
    _plot_mask(axes[plot_idx], prediction, cmap, image, alpha, background_idx)
    axes[plot_idx].set_title("Predicted Mask")
    axes[plot_idx].axis("off")
    plot_idx += 1

    # Plot difference if requested
    if show_difference:
        difference = _create_difference_mask(target, prediction, background_idx)
        diff_colors = [
            "green",
            "red",
            "blue",
        ]  # correct, false_positive, false_negative
        diff_cmap = ListedColormap(diff_colors)

        axes[plot_idx].imshow(difference, cmap=diff_cmap, alpha=0.8)
        if image is not None:
            axes[plot_idx].imshow(image, alpha=0.3)
        axes[plot_idx].set_title("Difference\n(Green=Correct, Red=False+, Blue=False-)")
        axes[plot_idx].axis("off")

    # Add legend
    if class_names is not None:
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], label=class_names[i])
            for i in range(min(len(class_names), num_classes))
        ]
        fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    plt.tight_layout()

    # Save if path provided
    return fig


def _to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def _plot_mask(ax, mask, cmap, image, alpha, background_idx):
    """Helper function to plot a single mask."""
    if image is not None:
        ax.imshow(image, alpha=0.5)

    # Create masked array to handle background transparency
    if background_idx is not None:
        mask_plot = np.ma.masked_where(mask == background_idx, mask)
    else:
        mask_plot = mask

    ax.imshow(mask_plot, cmap=cmap, alpha=alpha)


def _create_difference_mask(target, prediction, background_idx):
    """Create a difference mask showing correct/incorrect predictions."""
    # 0: correct, 1: false positive, 2: false negative
    difference = np.zeros_like(target)

    # Correct predictions
    correct = target == prediction
    difference[correct] = 0

    # False positives (predicted as class but should be background)
    if background_idx is not None:
        false_positive = (target == background_idx) & (prediction != background_idx)
        difference[false_positive] = 1

        # False negatives (predicted as background but should be class)
        false_negative = (target != background_idx) & (prediction == background_idx)
        difference[false_negative] = 2
    else:
        # Without background, just show incorrect predictions
        incorrect = ~correct
        difference[incorrect] = 1

    return difference
