import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss implementation for multi-class segmentation with background class.

    This implementation:
    - Handles multiple foreground classes plus a background class
    - Accepts segmentation masks with class indices as targets
    - Applies softmax to convert logits to probabilities
    - Uses a smooth factor to avoid division by zero
    - Optionally applies class weights to handle class imbalance

    Optimized for segmentation tasks where targets are masks with class indices.
    """

    def __init__(
        self,
        n_classes: int,
        add_background: bool = True,
        smooth: float = 1e-5,
        apply_softmax: bool = True,
        reduction: str = "mean",
        class_weights: list[float] = None,
    ):
        """
        Args:
            n_classes (int): Number of foreground classes
            add_background (bool): Whether background is included in n_classes count
            smooth (float): Small value to avoid division by zero
            apply_softmax (bool): Whether to apply softmax to the input logits
            reduction (str): 'none', 'mean', or 'sum'
            class_weights (torch.Tensor, optional): Optional class weights tensor of shape (n_classes,)
        """
        super(SoftDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.add_background = add_background
        self.smooth = smooth
        self.apply_softmax = apply_softmax
        self.reduction = reduction

        # If background is separate, total classes is n_classes + 1
        self.total_classes = n_classes + 1 if add_background else n_classes

        # Setup class weights if provided
        if class_weights is not None:
            assert (
                len(class_weights) == self.total_classes
            ), f"Class weights length {len(class_weights)} doesn't match total classes {self.total_classes}"
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = torch.ones(1, self.total_classes)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions of shape (B, C, H, W)
                                  where C is the number of classes including background
            targets (torch.Tensor): Segmentation masks of shape (B, C, H, W) with class indices (0 to C-1)

        Returns:
            torch.Tensor: Computed Dice Loss
        """
        # Get batch size and spatial dimensions
        batch_size = inputs.size(0)

        # Convert input logits to probabilities if needed
        if self.apply_softmax:
            inputs = F.softmax(inputs, dim=1)

        # Convert segmentation mask targets to one-hot encoding

        # Calculate Dice Loss for each class
        dice_scores = torch.zeros(batch_size, self.n_classes)

        # Flatten spatial dimensions
        input_flat = einops.rearrange(inputs, "b c h w -> b c (h w)")
        target_flat = einops.rearrange(targets, "b c h w -> b c (h w)")

        intersection = torch.sum(input_flat * target_flat, dim=-1)  # Shape: (B, C)
        inputs_sum = torch.sum(input_flat, dim=-1)  # Shape: (B, C)
        targets_sum = torch.sum(target_flat, dim=-1)  # Shape: (B, C)

        # Calculate Dice score for this class
        dice_scores = (2.0 * intersection + self.smooth) / (
            inputs_sum + targets_sum + self.smooth
        )

        dice_loss = torch.ones(1, self.n_classes).to(dice_scores.device) - dice_scores
        dice_loss = dice_scores * self.class_weights.to(dice_scores.device)
        if self.reduction == "none":
            return dice_loss
        elif self.reduction == "sum":
            return torch.sum(dice_loss)
        else:  # 'mean'
            # Mean across both batch and class dimensions
            return torch.mean(dice_loss)
