import os
import numpy as np
from typing import Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets

__all__ = ["build_transforms", "create_datasets"]

# ----------------------------------------------------------------------------- #
#                               TRANSFORMS                                      #
# ----------------------------------------------------------------------------- #

def _default_mean_std(dataset_name: str):
    if dataset_name in {"cifar10", "cifar10c", "cifar100", "cifar100c"}:
        return [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_transforms(cfg: Dict[str, Any]):
    """Construct training / validation transforms automatically."""
    ds_name = cfg["dataset"].get("name", "imagenet").lower()
    tf_cfg = cfg.get("transforms", {})

    if ds_name in {"cifar10", "cifar10c", "cifar100", "cifar100c"}:
        img_size = int(tf_cfg.get("resize", 32))
    else:
        img_size = int(tf_cfg.get("resize", 224))

    mean, std = _default_mean_std(ds_name)

    normalize = transforms.Normalize(mean=mean, std=std)
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return transform_train, transform_val

# ----------------------------------------------------------------------------- #
#                            DATASET HELPERS                                    #
# ----------------------------------------------------------------------------- #

class _HFWrapper(Dataset):
    """Wrap a HuggingFace image classification split so it acts like a PyTorch dataset."""
    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"]
        label = int(sample["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# ------------------------------- CIFAR-10-C ---------------------------------- #

def _create_cifar10c(cfg: Dict[str, Any], transform) -> Dataset:
    """Load CIFAR-10-C from local npz *or* HuggingFace hub path of the form hf://<repo>."""
    path = cfg["dataset"].get("path")
    if path is None or str(path).startswith("hf://"):
        # HuggingFace variant
        repo = path[len("hf://"):] if path else "randall-lab/cifar10-c"
        from datasets import load_dataset
        hf_ds = load_dataset(repo, split="test", trust_remote_code=True)
        return _HFWrapper(hf_ds, transform)

    # Local npz (original authors’ release)
    data_npz = np.load(path)
    data, labels = data_npz["data"], data_npz["labels"].astype(np.int64)
    class _DS(Dataset):
        def __len__(self): return len(labels)
        def __getitem__(self, idx):
            img = transforms.functional.to_pil_image(data[idx])
            return transform(img), labels[idx]
    return _DS()

# ------------------------------- CIFAR-100 & CLEAN --------------------------- #

def _create_cifar10(cfg: Dict[str, Any], split: str, transform) -> Dataset:
    root = cfg["dataset"].get("path", "./data")
    train = split == "train"
    return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)


def _create_cifar100(cfg: Dict[str, Any], split: str, transform) -> Dataset:
    root = cfg["dataset"].get("path", "./data")
    train = split == "train"
    return datasets.CIFAR100(root=root, train=train, download=True, transform=transform)


def _create_cifar100c(cfg: Dict[str, Any], transform) -> Dataset:
    path = cfg["dataset"].get("path")
    if path is None or str(path).startswith("hf://"):
        repo = path[len("hf://"):] if path else "randall-lab/cifar100-c"
        from datasets import load_dataset
        hf_ds = load_dataset(repo, split="test", trust_remote_code=True)
        return _HFWrapper(hf_ds, transform)
    data_npz = np.load(path)
    data, labels = data_npz["data"], data_npz["labels"].astype(np.int64)
    class _DS(Dataset):
        def __len__(self): return len(labels)
        def __getitem__(self, idx):
            img = transforms.functional.to_pil_image(data[idx])
            return transform(img), labels[idx]
    return _DS()

# ------------------------------- ImageNet-C ---------------------------------- #

def _create_imagenetc(cfg: Dict[str, Any], transform) -> Dataset:
    """ImageNet-C loader – expects directory with corruption folders OR hf path."""
    path = cfg["dataset"].get("path")
    if path is None:
        raise FileNotFoundError("ImageNet-C path not provided in config")
    if str(path).startswith("hf://"):
        repo = path[len("hf://"):]
        from datasets import load_dataset
        hf_ds = load_dataset(repo, split="test", trust_remote_code=True)
        return _HFWrapper(hf_ds, transform)
    # Local directory version (same layout as original ImageNet-C release)
    return datasets.ImageFolder(root=path, transform=transform)

# ------------------------------- FACTORY ------------------------------------- #

DATASET_FACTORY = {
    "cifar10": _create_cifar10,
    "cifar10c": _create_cifar10c,
    "cifar100": _create_cifar100,
    "cifar100c": _create_cifar100c,
    "imagenetc": _create_imagenetc,
    # More specialised video or stream datasets can be appended here.
}

# ----------------------------------------------------------------------------- #
#                          DATASET DISPATCHER                                   #
# ----------------------------------------------------------------------------- #

def create_datasets(cfg: Dict[str, Any], tf_train, tf_val) -> Tuple[Dataset, Dataset, Dataset]:
    name = cfg["dataset"].get("name", "fakedata").lower()

    # Corruption datasets → only test split exists; we replicate as train/val placeholders
    if name in {"cifar10c", "cifar100c", "imagenetc"}:
        test_set = DATASET_FACTORY[name](cfg, tf_val)
        train_set = test_set
        val_set = test_set
        return train_set, val_set, test_set

    # Standard datasets with explicit splits
    creator = DATASET_FACTORY.get(name)
    if creator is None:
        raise NotImplementedError(f"Dataset '{name}' is not implemented.")
    train_set = creator(cfg, split="train", transform=tf_train)
    val_set = creator(cfg, split="val", transform=tf_val)
    test_set = creator(cfg, split="test", transform=tf_val)
    return train_set, val_set, test_set
