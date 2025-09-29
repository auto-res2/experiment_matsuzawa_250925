"""src/train.py
Core training logic with algorithm implementation and placeholder data loading.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluate import Evaluator


class Trainer:
    """Universal trainer that can handle any PyTorch model & dataset.

    The main training loop is *fully* implemented (no placeholders).  The logic
    is dataset/model-agnostic – only the *data* and *model* themselves are
    provided as run-time objects.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = config
        self.checkpoint_dir = Path(self.cfg["training"].get("checkpoint_dir", ".research/iteration2/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric_name = self.cfg["evaluation"]["metrics"][0]  # first metric regarded as primary
        self.best_metric_value = -float("inf")
        self.best_ckpt_path = self.checkpoint_dir / "best_model.pt"
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            f"val_{self.best_metric_name}": [],
        }

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------
    def fit(self) -> Dict:
        epochs: int = int(self.cfg["training"]["epochs"])
        evaluator = Evaluator(self.device, self.cfg["evaluation"]["metrics"])
        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            self.history["train_loss"].append(train_loss)
            # Validation
            val_metrics = evaluator.evaluate(self.model, self.val_loader)
            val_primary_metric = val_metrics[self.best_metric_name]
            self.history[f"val_{self.best_metric_name}"].append(val_primary_metric)
            # Progress bar / stdout
            print(
                f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.4f} | "
                f"Val {self.best_metric_name}: {val_primary_metric:.4f}"
            )
            # Checkpointing
            self._save_checkpoint(epoch, val_primary_metric)
        # End of fit – return history for plotting
        return self.history

    def load_best_model(self) -> nn.Module:
        """Reload the best checkpoint and return the model on *CPU* (for safety)."""
        if self.best_ckpt_path.exists():
            state = torch.load(self.best_ckpt_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(state["model_state_dict"])
            self.model.eval()
        else:
            raise FileNotFoundError("Best checkpoint not found – was training executed?")
        return self.model.cpu()

    # -----------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        total = 0
        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}")
        for batch in pbar:
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        epoch_loss = running_loss / total
        return epoch_loss

    def _save_checkpoint(self, epoch: int, metric_value: float) -> None:
        """Save *latest* ckpt and *best* ckpt (based on primary metric)."""
        # Latest checkpoint (overwritten each epoch to save disk)
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metric_value": metric_value,
            },
            latest_path,
        )
        # Best checkpoint (if improved)
        if metric_value > self.best_metric_value:
            self.best_metric_value = metric_value
            shutil.copy(latest_path, self.best_ckpt_path)
