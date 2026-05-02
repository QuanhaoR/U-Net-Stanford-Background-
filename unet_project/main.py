import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

from train import train


def main():
    parser = argparse.ArgumentParser(description="U-Net on Stanford Background Dataset")
    parser.add_argument("--loss", type=str, default="ce",
                        choices=["ce", "dice", "combined"],
                        help="Loss function")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=256,
                        help="Resize images to this size")
    return parser.parse_args()


if __name__ == "__main__":
    args = main()

    config = {
        "loss": args.loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "img_size": args.img_size,
    }

    best_miou = train(config)
    print(f"\nBest val mIoU ({config['loss']}): {best_miou:.4f}")
