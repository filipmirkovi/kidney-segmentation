from typing import Literal
from pydantic import BaseModel
import torch.nn as nn
import torch


class ImageInfo(BaseModel):
    height: int
    width: int
    channels: int


def num_params(
    model: nn.Module,
    units: Literal["million", "billion", "thousand", "none"] = "million",
) -> str:
    num_params = sum([param.numel() for param in model.parameters()])
    match units:
        case "none":
            return f"{num_params}"
        case "thousand":
            return f"{num_params/1e3} thousand"
        case "million":
            return f"{num_params/1e6} million"
        case "billion":
            return f"{num_params/1e9} billion"
        case _:
            raise ValueError(
                f"{units} is an unkown type of unit. Please provide: 'million', 'billion', 'thousand', 'none'."
            )


def calculate_per_class_recall(
    logits: torch.Tensor, labels: torch.Tensor, num_classes=None, eps=1e-8
):
    """
    Calculate recall for each class given logits and labels.

    Recall = True Positives / (True Positives + False Negatives)
           = True Positives / Total Actual Positives

    Args:
        logits (torch.Tensor): Raw model outputs of shape (N, C) for classification
                              or (N, C, H, W) for segmentation
        labels (torch.Tensor): Ground truth labels of shape (N,) for classification
                              or (N, H, W) for segmentation
        num_classes (int): Number of classes. If None, inferred from logits.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        torch.Tensor: Per-class recall values of shape (num_classes,)
    """
    if num_classes is None:
        num_classes = logits.shape[1]

    # Get predictions from logits
    predictions = torch.argmax(logits, dim=1)

    # Flatten tensors for easier computation (handles both classification and segmentation)
    predictions = predictions.view(-1)
    labels = labels.view(-1)

    # Initialize recall tensor
    recall_per_class = torch.zeros(
        num_classes, dtype=torch.float32, device=logits.device
    )

    # Calculate recall for each class
    for class_idx in range(num_classes):
        # True positives: predicted as class_idx AND actually class_idx
        tp = ((predictions == class_idx) & (labels == class_idx)).sum().float()

        # False negatives: predicted as NOT class_idx BUT actually class_idx
        fn = ((predictions != class_idx) & (labels == class_idx)).sum().float()

        # Total actual positives for this class
        total_actual_positives = tp + fn

        # Calculate recall (handle division by zero)
        if total_actual_positives > 0:
            recall_per_class[class_idx] = tp / (total_actual_positives + eps)
        else:
            # If no ground truth samples for this class, recall is undefined
            # We'll set it to 0 or NaN based on preference
            recall_per_class[class_idx] = 0.0  # or torch.nan

    return recall_per_class
