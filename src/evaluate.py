"""src/evaluate.py
Universal evaluation module with consistent metrics, figure generation, and JSON result dumping.
Now also supports transformer-style dict inputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

plt.switch_backend("Agg")  # no GUI backend required

__all__ = ["Evaluator"]


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        results_dir: Path,
    ) -> None:
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PRIVATE: helper to move inputs to device regardless of dict / tensor
    # ------------------------------------------------------------------
    def _to_device(self, X):
        if isinstance(X, dict):
            return {k: v.to(self.device) for k, v in X.items()}
        return X.to(self.device)

    @torch.no_grad()
    def _collect_predictions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        all_preds, all_labels = [], []
        for X, y in self.test_loader:
            X = self._to_device(X)
            logits = self.model(**X).logits if isinstance(X, dict) else self.model(X)
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y)
        return torch.cat(all_preds), torch.cat(all_labels)

    def evaluate(self, experiment_name: str = "experiment") -> Dict[str, Any]:
        preds, labels = self._collect_predictions()
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="weighted", zero_division=0)
        recall = recall_score(labels, preds, average="weighted", zero_division=0)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        cm = confusion_matrix(labels, preds)
        report = classification_report(labels, preds, zero_division=0, output_dict=True)

        metrics = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        # ------------------------------------------------------------
        # FIGURES
        # ------------------------------------------------------------
        self._plot_training_curves(experiment_name)
        self._plot_confusion_matrix(cm, experiment_name)

        # ------------------------------------------------------------
        # JSON OUTPUT
        # ------------------------------------------------------------
        result_path = self.results_dir / f"{experiment_name}_results.json"
        with open(result_path, "w", encoding="utf-8") as fp:
            json.dump({"metrics": metrics, "classification_report": report}, fp, indent=2)

        return {"metrics": metrics, "classification_report": report, "json": str(result_path)}

    # ------------------------------------------------------------
    # PRIVATE HELPERS FOR FIGURES
    # ------------------------------------------------------------
    def _plot_training_curves(self, experiment_name: str) -> None:
        history_path = self.results_dir / f"{experiment_name}_history.json"
        if not history_path.exists():
            return  # training curves not available (e.g., model loaded from disk)
        with open(history_path, "r", encoding="utf-8") as fp:
            history = json.load(fp)
        epochs = list(range(1, len(history["train_loss"]) + 1))
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["train_loss"], label="Training Loss", marker="o")
        plt.plot(epochs, history["val_loss"], label="Validation Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        for x, y in zip(epochs, history["train_loss"]):
            plt.text(x, y, f"{y:.2f}")
        for x, y in zip(epochs, history["val_loss"]):
            plt.text(x, y, f"{y:.2f}")
        plt.legend()
        fname = self.results_dir / "training_loss.pdf"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

        # Accuracy curve
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["val_acc"], label="Validation Accuracy", marker="o", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        for x, y in zip(epochs, history["val_acc"]):
            plt.text(x, y, f"{y:.2f}")
        plt.legend()
        fname = self.results_dir / "accuracy.pdf"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

    def _plot_confusion_matrix(self, cm, experiment_name: str) -> None:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        fname = self.results_dir / "confusion_matrix.pdf"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
