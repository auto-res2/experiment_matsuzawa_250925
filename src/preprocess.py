from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

__all__ = ["build_dataloaders"]


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """Return train & validation loaders according to config."""
    batch_size = cfg.get("batch_size", 64)
    num_workers = min(8, os.cpu_count() or 2)

    if cfg["dataset"].lower() == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        root = Path(os.getenv("DATA", "./data"))
        train_set = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform)
        val_set = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform)
    else:
        # generic ImageFolder
        root = Path(cfg["dataset_root"]).expanduser()
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        train_set = torchvision.datasets.ImageFolder(root / "train", transform=transform)
        val_set = torchvision.datasets.ImageFolder(root / "val", transform=transform)

    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    return train_loader, val_loader
