"""src/preprocess.py
Common preprocessing pipeline with dataset placeholders.

This module contains *complete* preprocessing utilities that can be re-used by
any dataset.  For datasets that need specialised logic, the relevant subclass
can override `transform`.  The default implementation performs feature
standardisation (zero-mean, unit-variance).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset


class StandardScalerPreprocessor:
    """Simple standard scaler that works on tensors of shape (N, F, ...)."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, dataset: Dataset) -> None:
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        sums = None
        sq_sums = None
        count = 0
        for data, _ in loader:
            # Flatten feature dimension – assumes (batch, features, ...)
            vec = data.view(data.size(0), -1)
            batch_sum = vec.sum(0)
            batch_sq_sum = (vec ** 2).sum(0)
            if sums is None:
                sums = batch_sum
                sq_sums = batch_sq_sum
            else:
                sums += batch_sum
                sq_sums += batch_sq_sum
            count += vec.size(0)
        self.mean = sums / count
        var = sq_sums / count - self.mean ** 2
        self.std = torch.sqrt(var + 1e-8)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Preprocessor not fitted.")
        orig_device = data.device
        return ((data - self.mean.to(orig_device)) / self.std.to(orig_device)).float()

    # Convenience wrapper for fit+transform on the fly --------------------------------------------------
    def fit_transform(self, dataset: Dataset) -> Tuple[Dataset, "StandardScalerPreprocessor"]:
        self.fit(dataset)

        class _WrappedDataset(Dataset):
            def __init__(self, ds, scaler: StandardScalerPreprocessor):
                self.ds = ds
                self.scaler = scaler

            def __len__(self):
                return len(self.ds)

            def __getitem__(self, idx):
                x, y = self.ds[idx]
                x = self.scaler.transform(x)
                return x, y

        return _WrappedDataset(dataset, self), self