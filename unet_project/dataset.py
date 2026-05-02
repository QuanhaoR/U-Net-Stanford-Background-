import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path

# Semantic classes from Stanford Background Dataset
CLASS_NAMES = ["sky", "tree", "road", "grass", "water", "building", "mountain", "object"]


class StanfordBackground(Dataset):
    def __init__(self, data_root, split="train", val_ratio=0.2, seed=42, img_size=256):
        self.data_root = Path(data_root)
        self.img_size = img_size

        image_dir = self.data_root / "images"
        label_dir = self.data_root / "labels"

        ids = [f.stem for f in sorted(image_dir.glob("*.jpg"))]

        # Deterministic split
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(ids))
        split_pt = int(len(ids) * (1 - val_ratio))
        if split == "train":
            self.ids = [ids[i] for i in idx[:split_pt]]
        else:
            self.ids = [ids[i] for i in idx[split_pt:]]

        self.image_dir = image_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]

        # Load image
        img = Image.open(self.image_dir / f"{sid}.jpg").convert("RGB")
        orig_w, orig_h = img.size

        # Load label (regions = semantic class)
        label = np.loadtxt(self.label_dir / f"{sid}.regions.txt", dtype=np.int64)
        # label shape: (H, W) — note row-major, so H=orig_h, W=orig_w

        # Resize image and label
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        label = Image.fromarray(label.astype(np.int32))
        label = label.resize((self.img_size, self.img_size), Image.NEAREST)
        label = np.array(label, dtype=np.int64)

        # Convert -1 (unknown) to 255 (ignore_index)
        label[label == -1] = 255

        image = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).long()

        return image, label
