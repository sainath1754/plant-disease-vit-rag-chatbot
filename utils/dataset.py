# ============================================================
# utils/dataset.py — Dataset Loading & Splitting
# ============================================================

import os
import sys
import platform
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO,
                    BATCH_SIZE, RANDOM_SEED, CLASS_NAMES)

# ── Windows fix: num_workers > 0 causes freezing on Windows ──
_IS_WINDOWS  = platform.system() == "Windows"
_NUM_WORKERS = 0 if _IS_WINDOWS else 4
_PIN_MEMORY  = torch.cuda.is_available()   # only useful when a GPU exists


# ── Augmentation / Normalisation Pipelines ───────────────────

def get_train_transforms():
    """Augmentation for training (rotation, flip, scale)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms():
    """Deterministic transforms for validation / testing."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


# ── Custom Dataset ────────────────────────────────────────────

class PlantVillageDataset(Dataset):
    """
    Loads PlantVillage leaf images from a folder whose sub-directories
    represent disease class labels.

    Expected structure:
        PlantVillage/
            Apple___Apple_scab/
                img001.jpg
                ...
            Tomato___healthy/
                img999.jpg
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples   = []   # list of (path, label_idx)
        self.classes   = CLASS_NAMES

        self._load_samples()

    # ── private ──────────────────────────────────────────────

    def _load_samples(self):
        class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(img_path), class_to_idx[cls_name]))

        print(f"[Dataset] Loaded {len(self.samples)} images "
              f"across {len(self.classes)} classes.")

    # ── public ───────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Data Loaders ──────────────────────────────────────────────

def get_data_loaders(data_dir: str, batch_size: int = BATCH_SIZE):
    """
    Returns train / val / test DataLoaders with a 70:15:15 split.
    """
    full_dataset = PlantVillageDataset(data_dir, transform=None)
    n            = len(full_dataset)

    # Reproducible shuffle
    indices = list(range(n))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)

    train_end = int(n * TRAIN_RATIO)
    val_end   = train_end + int(n * VAL_RATIO)

    train_idx = indices[:train_end]
    val_idx   = indices[train_end:val_end]
    test_idx  = indices[val_end:]

    # Wrap subsets with correct transforms
    train_ds = _SubsetWithTransform(full_dataset, train_idx, get_train_transforms())
    val_ds   = _SubsetWithTransform(full_dataset, val_idx,   get_eval_transforms())
    test_ds  = _SubsetWithTransform(full_dataset, test_idx,  get_eval_transforms())

    print(f"[Dataset] num_workers={_NUM_WORKERS}, pin_memory={_PIN_MEMORY}")
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=_NUM_WORKERS,
                              pin_memory=_PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=_NUM_WORKERS,
                              pin_memory=_PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=_NUM_WORKERS,
                              pin_memory=_PIN_MEMORY)

    print(f"[Dataset] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ── Helper: Subset with per-split transform ───────────────────

class _SubsetWithTransform(Dataset):
    def __init__(self, dataset: PlantVillageDataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[self.indices[idx]]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
