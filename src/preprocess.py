"""src/preprocess.py
Dataset loader now supports *real* Hugging-Face Reddit comments dataset and
creates PyTorch-friendly tensors for text-classification.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from transformers import AutoTokenizer


class HFDataset(Dataset):
    """Generic wrapper converting HF samples to token ids."""

    def __init__(self, split, tokenizer, text_key: str, label_key: str, max_len: int = 128):
        self.ds = split
        self.tok = tokenizer
        self.text_key = text_key
        self.label_key = label_key
        self.max_len = max_len

    def __len__(self):  # type: ignore[override]
        return len(self.ds)

    def __getitem__(self, idx):  # type: ignore[override]
        item = self.ds[idx]
        enc = self.tok(
            item[self.text_key],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(item[self.label_key], dtype=torch.long)
        return enc


class DummyClassificationDataset(Dataset):
    """Synthetic dataset for smoke tests."""

    def __init__(self, n_samples: int = 1024, n_features: int = 32, n_classes: int = 2):
        self.x = torch.randn(n_samples, n_features)
        self.y = torch.randint(0, n_classes, size=(n_samples,))

    def __len__(self):  # type: ignore[override]
        return self.x.size(0)

    def __getitem__(self, idx):  # type: ignore[override]
        return self.x[idx], self.y[idx]


# -----------------------------------------------------------------------------
#  PUBLIC FACTORY FUNCTION
# -----------------------------------------------------------------------------

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds_cfg = cfg["dataset"]
    name = ds_cfg.get("name")
    batch_size = ds_cfg.get("batch_size", 32)

    # ------------------------------------------------------------------
    #  1) HF Reddit comments – node classification turned text-classification
    # ------------------------------------------------------------------
    if name == "tensorshield/reddit_dataset_157":
        split_train = load_dataset(name, split="train[:70%]")
        split_val = load_dataset(name, split="train[70%:85%]")
        split_test = load_dataset(name, split="train[85%:]")
        tok_name = ds_cfg.get("tokenizer", "bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        text_key = ds_cfg.get("text_key", "body")
        label_key = ds_cfg.get("label_key", "subreddit_id")
        max_len = ds_cfg.get("max_len", 128)
        train_ds = HFDataset(split_train, tokenizer, text_key, label_key, max_len)
        val_ds = HFDataset(split_val, tokenizer, text_key, label_key, max_len)
        test_ds = HFDataset(split_test, tokenizer, text_key, label_key, max_len)

        def collate_fn(batch):
            keys = batch[0].keys()
            out = {k: torch.stack([b[k] for b in batch]) for k in keys}
            return out

        loader_kwargs = dict(batch_size=batch_size, num_workers=2, shuffle=True, collate_fn=collate_fn)
        return (
            DataLoader(train_ds, **loader_kwargs),
            DataLoader(val_ds, **{**loader_kwargs, "shuffle": False}),
            DataLoader(test_ds, **{**loader_kwargs, "shuffle": False}),
        )

    # ------------------------------------------------------------------
    #  2) Synthetic fallback (smoke test) -------------------------------
    # ------------------------------------------------------------------
    if name == "DATASET_PLACEHOLDER":
        dataset = DummyClassificationDataset(
            n_samples=ds_cfg.get("n_samples", 1024),
            n_features=ds_cfg.get("n_features", 32),
            n_classes=ds_cfg.get("n_classes", 2),
        )
        n_total = len(dataset)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        n_test = n_total - n_train - n_val
        train_ds, val_ds, test_ds = random_split(
            dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
        )
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0),
            DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0),
        )

    raise ValueError(f"Unknown dataset name '{name}' in config.")
