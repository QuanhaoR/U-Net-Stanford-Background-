import torch
from torch.utils.data import DataLoader
from pathlib import Path

from model import UNet
from dataset import StanfordBackground
from utils import compute_miou, compute_accuracy


@torch.no_grad()
def evaluate(checkpoint, loss_name="ce", data_root=None, img_size=256, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_root is None:
        data_root = Path(__file__).resolve().parent.parent / "data"

    model = UNet(n_channels=3, n_classes=8).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    dataset = StanfordBackground(data_root, split="val", img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    miou_list, acc_list = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        miou_list.append(compute_miou(logits, labels, 8))
        acc_list.append(compute_accuracy(logits, labels))

    avg_miou = sum(miou_list) / len(miou_list)
    avg_acc = sum(acc_list) / len(acc_list)

    print(f"[{loss_name}] mIoU: {avg_miou:.4f}  Acc: {avg_acc:.4f}")
    return avg_miou, avg_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.loss, img_size=args.img_size, batch_size=args.batch_size)
