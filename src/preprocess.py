"""src/preprocess.py
Common preprocessing pipeline. Any dataset-specific logic is marked with placeholders and can be
replaced without touching the surrounding infrastructure.
"""
from __future__ import annotations

import random
from typing import Dict, Any, Tuple

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

__all__ = ["load_datasets"]

def _prepare_synthetic_dataset(num_samples: int, num_features: int, num_classes: int, seed: int = 42):
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=max(2, num_features // 2),
        n_redundant=0,
        n_classes=num_classes,
        random_state=seed,
    )
    return X.astype(np.float32), y.astype(np.int64)


def load_datasets(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """Return train/val/test DataLoaders and the (input_dim, num_classes)."""

    dataset_cfg = cfg["dataset"]
    dataset_name = dataset_cfg.get("name", "DATASET_PLACEHOLDER").lower()

    # ----------------------------------------------------------------------------------
    # PLACEHOLDER: Replace with custom dataset loading logic.
    # ----------------------------------------------------------------------------------
    if dataset_name in {"synthetic", "placeholder"}:
        n_samples = dataset_cfg.get("num_samples", 1000)
        n_features = dataset_cfg.get("num_features", 20)
        n_classes = dataset_cfg.get("num_classes", 2)
        X, y = _prepare_synthetic_dataset(n_samples, n_features, n_classes)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Add real dataset handling in load_datasets()."
        )
    # ----------------------------------------------------------------------------------

    # Train/val/test split
    test_size = dataset_cfg.get("test_split", 0.2)
    val_size = dataset_cfg.get("val_split", 0.2)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=test_size + val_size, random_state=42)
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=relative_val_size, random_state=42
    )

    # Convert to torch tensors
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    batch_size = cfg["training"].get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    return train_loader, val_loader, test_loader, input_dim, num_classes
