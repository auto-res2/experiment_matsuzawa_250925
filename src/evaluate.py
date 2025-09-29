"""src/evaluate.py
Universal evaluation framework that works across all experimental variations.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader


class Evaluator:
    """Evaluator computes a fixed set of metrics for *classification* models.

    This evaluation logic never changes across experimental variants – only the
    data and models vary.
    """

    SUPPORTED_METRICS = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
    }

    def __init__(self, device: torch.device, metrics: List[str]):
        self.device = device
        self.metrics = [m.lower() for m in metrics]
        for m in self.metrics:
            if m not in self.SUPPORTED_METRICS:
                raise ValueError(f"Unsupported metric: {m}")

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict:
        model.eval()
        all_preds = []
        all_labels = []
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels.cpu())
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        results = {}
        average_type = "macro"  # consistent averaging across experiments
        for m in self.metrics:
            func = self.SUPPORTED_METRICS[m]
            if m == "accuracy":
                results[m] = float(func(y_true, y_pred))
            else:
                results[m] = float(func(y_true, y_pred, average=average_type, zero_division=0))
        # Always compute confusion matrix for visualization later (not used for early-stopping)
        results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        return results