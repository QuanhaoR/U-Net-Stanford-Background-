import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class Dice Loss (manual implementation)."""

    def __init__(self, smooth=1e-6, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        # logits: (B, C, H, W), target: (B, H, W)
        B, C, H, W = logits.shape

        # Mask out ignore_index
        mask = target != self.ignore_index  # (B, H, W)

        # Clamp target before one_hot to avoid OOB errors (255 -> 0)
        target_safe = target.clone()
        target_safe[~mask] = 0

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # Convert target to one-hot
        target_one_hot = F.one_hot(target_safe, num_classes=C)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Apply mask
        mask = mask.unsqueeze(1).float()  # (B, 1, H, W)
        probs = probs * mask
        target_one_hot = target_one_hot * mask

        # Dice per class
        intersection = (probs * target_one_hot).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score  # (B, C)

        # Average over classes and batch
        return dice_loss.mean()


class CombinedLoss(nn.Module):
    """Cross-Entropy Loss + Dice Loss"""

    def __init__(self, ce_weight=1.0, dice_weight=1.0, ignore_index=255):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits, target):
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


def get_loss_fn(name):
    if name == "ce":
        return nn.CrossEntropyLoss(ignore_index=255)
    elif name == "dice":
        return DiceLoss(ignore_index=255)
    elif name == "combined":
        return CombinedLoss(ce_weight=1.0, dice_weight=1.0, ignore_index=255)
    else:
        raise ValueError(f"Unknown loss: {name}")
