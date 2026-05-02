import torch
import numpy as np
from PIL import Image
from pathlib import Path

from model import UNet
from dataset import StanfordBackground


# Color palette for 8 classes (RGB)
PALETTE = np.array([
    [128, 128, 128],  # 0 sky: gray
    [0,   128, 0  ],  # 1 tree: dark green
    [128, 64,  0  ],  # 2 road: brown
    [0,   255, 0  ],  # 3 grass: green
    [0,   0,   255],  # 4 water: blue
    [255, 128, 0  ],  # 5 building: orange
    [128, 0,   128],  # 6 mountain: purple
    [255, 0,   0  ],  # 7 object: red
], dtype=np.uint8)


def colorize(mask):
    """Convert class index mask (H,W) to RGB image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(8):
        rgb[mask == c] = PALETTE[c]
    return rgb


@torch.no_grad()
def predict(model, image_tensor, device):
    """Run inference on a single image tensor (C,H,W)."""
    model.eval()
    image = image_tensor.unsqueeze(0).to(device)
    logits = model(image)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    return pred


def visualize_sample(image_tensor, gt_mask, pred_mask, save_path):
    """Create and save a 3-panel visualization: image | ground truth | prediction."""
    img = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gt_rgb = colorize(gt_mask)
    pred_rgb = colorize(pred_mask)

    # Stack side-by-side
    h, w = img.shape[:2]
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, :w] = img
    canvas[:, w:2*w] = gt_rgb
    canvas[:, 2*w:3*w] = pred_rgb

    Image.fromarray(canvas).save(save_path)
    return canvas


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize U-Net segmentation results")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--loss", type=str, default="ce",
                        choices=["ce", "dice", "combined"],
                        help="Loss type used for training (affects legend only)")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of validation samples to visualize")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="vis_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = UNet(n_channels=3, n_classes=8).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load validation dataset
    data_root = Path(__file__).resolve().parent.parent / "data"
    val_dataset = StanfordBackground(data_root, split="val", img_size=args.img_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    n = min(args.num_samples, len(val_dataset))
    for i in range(n):
        image, label = val_dataset[i]
        # Convert 255 back to 0 for visualization (ignore_index -> background)
        label_vis = label.clone()
        label_vis[label_vis == 255] = 0

        pred = predict(model, image, device)

        fname = out_dir / f"sample_{i:02d}.png"
        visualize_sample(image, label_vis.numpy(), pred, fname)

        # Per-sample IoU
        mask = label != 255
        intersection = (pred[mask.numpy()] == label[mask].numpy()).sum()
        union = mask.sum().item()
        pix_acc = intersection / union if union > 0 else 0
        print(f"[{i+1}/{n}] Saved {fname}  pixel_acc={pix_acc:.3f}")

    print(f"\nDone. Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
