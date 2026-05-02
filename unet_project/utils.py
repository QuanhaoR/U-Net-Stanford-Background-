import torch
import numpy as np


@torch.no_grad()
def compute_miou(logits, target, n_classes=8, ignore_index=255):
    """Compute mean Intersection-over-Union."""
    pred = torch.argmax(logits, dim=1)  # (B, H, W)

    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    ious = []
    for cls in range(n_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            ious.append(float("nan"))  # class not present
        else:
            ious.append(intersection / union)

    return np.nanmean(ious)


@torch.no_grad()
def compute_accuracy(logits, target, ignore_index=255):
    """Compute pixel accuracy."""
    pred = torch.argmax(logits, dim=1)
    mask = target != ignore_index
    correct = (pred[mask] == target[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0
