from collections import defaultdict
import torch


def iou_score(
    y_true: torch.Tensor, y_pred: torch.Tensor, logits: bool = False
) -> torch.Tensor:
    """
    Returns intersection over union score per class.
    """
    if logits:
        y_pred = (y_pred > 0).float()
    else:
        y_pred = (y_pred > 0.5).float()

    intersection = torch.logical_and(y_pred, y_true).float()
    union = torch.logical_or(y_pred, y_true).float()
    iou = intersection.sum(dim=(-1, -2)) / union.sum(dim=(-1, -2))
    return iou
