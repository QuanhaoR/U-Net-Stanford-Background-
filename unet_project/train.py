import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import os
import time
from pathlib import Path

from model import UNet
from dataset import StanfordBackground, CLASS_NAMES
from losses import get_loss_fn
from utils import compute_miou, compute_accuracy


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, n_classes=8):
    model.eval()
    total_loss = 0
    miou_list = []
    acc_list = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        miou_list.append(compute_miou(logits, labels, n_classes))
        acc_list.append(compute_accuracy(logits, labels))

    avg_loss = total_loss / len(loader.dataset)
    avg_miou = sum(miou_list) / len(miou_list)
    avg_acc = sum(acc_list) / len(acc_list)
    return avg_loss, avg_miou, avg_acc


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    data_root = Path(__file__).resolve().parent.parent / "data"
    train_dataset = StanfordBackground(data_root, split="train", img_size=config["img_size"])
    val_dataset = StanfordBackground(data_root, split="val", img_size=config["img_size"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            shuffle=False, num_workers=4, pin_memory=True)

    n_classes = 8
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = UNet(n_channels=3, n_classes=n_classes).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss & optimizer
    criterion = get_loss_fn(config["loss"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                            factor=0.5, patience=5)

    # Wandb
    run_name = f"unet_{config['loss']}_lr{config['lr']}_bs{config['batch_size']}"
    wandb.init(project="hw2-unet", name=run_name, config=config, dir="/tmp")

    best_miou = 0.0
    for epoch in range(1, config["epochs"] + 1):
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_miou, val_acc = validate(model, val_loader, criterion, device, n_classes)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - start
        log = {"train_loss": train_loss, "val_loss": val_loss,
               "val_miou": val_miou, "val_acc": val_acc,
               "lr": current_lr, "epoch_time": epoch_time}
        wandb.log(log, step=epoch)

        print(f"[{epoch:3d}/{config['epochs']}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"mIoU={val_miou:.4f} acc={val_acc:.4f} "
              f"lr={current_lr:.2e} time={epoch_time:.1f}s")

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / f"best_{config['loss']}.pth")
            wandb.run.summary["best_miou"] = best_miou

    wandb.finish()
    return best_miou
