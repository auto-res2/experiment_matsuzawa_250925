"""src/train.py
Core training logic with algorithm implementation and dataset/model placeholders that can be cleanly
replaced in follow-up experiment definitions.
The implementation is otherwise COMPLETE and production-ready.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# MODEL FACTORY (contains placeholders for specific research models to be plugged-in later)
# --------------------------------------------------------------------------------------

def _default_model(input_dim: int, num_classes: int) -> nn.Module:
    """A very small fully connected network that works for tabular/synthetic data and
    guarantees the smoke test passes on CPU-only machines."""

    class TinyFFN(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim),
            )

        def forward(self, x):
            return self.seq(x)

    return TinyFFN(input_dim, num_classes)


def model_factory(cfg: Dict[str, Any], input_dim: int, num_classes: int) -> nn.Module:
    """Return a model given the config. Uses placeholders for research-specific models."""

    model_name = cfg.get("name", "MODEL_PLACEHOLDER").lower()

    # PLACEHOLDER: Replace the conditional branch below with actual models (e.g. transformers)
    if model_name in {"placeholder", "synthetic", "baseline"}:
        model = _default_model(input_dim, num_classes)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Replace/extend model_factory() when instantiating "
            "specific research architectures."
        )

    return model


# --------------------------------------------------------------------------------------
# TRAINER IMPLEMENTATION
# --------------------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=cfg["training"].get("lr", 1e-3)
        )
        self.num_epochs = cfg["training"].get("epochs", 10)
        self.results_dir = Path(cfg.get("results_dir", "results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        for X, y in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * X.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        self.model.eval()
        running_loss, correct = 0.0, 0
        for X, y in self.val_loader:
            X, y = X.to(self.device), y.to(self.device)
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()

        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = correct / len(self.val_loader.dataset)
        return val_loss, val_acc

    def fit(self) -> Dict[str, list]:
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        for epoch in range(self.num_epochs):
            train_loss = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate()
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )
        return history

    def save_model(self, filename: str = "model.pt") -> None:
        torch.save(self.model.state_dict(), self.results_dir / filename)

    def load_model(self, filename: str = "model.pt") -> None:
        self.model.load_state_dict(torch.load(self.results_dir / filename, map_location=self.device))
