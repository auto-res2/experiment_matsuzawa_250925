import os
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["Evaluator"]


class Evaluator:
    """Accuracy, calibration, latency – plus automatic figure generation."""

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.generated_figures: List[str] = []  # paths collected during runtime

    # --------------------------------------------------------------------- #
    @staticmethod
    def _accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
        return float(pred.eq(target).float().mean().item() * 100.0)

    @staticmethod
    def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
        """Vectorised Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.any():
                ece += np.abs(confidences[mask].mean() - accuracies[mask].mean()) * mask.mean()
        return float(ece)

    # --------------------------------------------------------------------- #
    def evaluate(self, model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        model.eval()
        logits_all, labels_all, latencies = [], [], []

        use_cuda_events = torch.cuda.is_available()
        starter, ender = None, None
        if use_cuda_events:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if use_cuda_events:
                    starter.record()
                out = model(x)
                if use_cuda_events:
                    ender.record(); torch.cuda.synchronize(); latencies.append(starter.elapsed_time(ender))

                logits_all.append(out.cpu())
                labels_all.append(y.cpu())

        logits = torch.cat(logits_all)
        labels = torch.cat(labels_all)
        probs = logits.softmax(dim=1).numpy()
        preds = logits.argmax(dim=1)

        # Fix brier score calculation for multiclass
        def _multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
            """Calculate Brier score for multiclass classification."""
            n_classes = y_prob.shape[1]
            y_true_one_hot = np.eye(n_classes)[y_true]
            return np.mean(np.sum((y_prob - y_true_one_hot) ** 2, axis=1))

        metrics = {
            "accuracy_top1": self._accuracy(preds, labels),
            "ece": self._ece(probs, labels.numpy()),
            "brier": _multiclass_brier_score(labels.numpy(), probs),
            "latency_ms_mean": float(np.mean(latencies)) if latencies else float("nan"),
            "latency_ms_std": float(np.std(latencies)) if latencies else float("nan"),
        }

        # ----------------------- FIGURES --------------------------------- #
        fig_dir = self.cfg.get("figures_dir", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Confusion Matrix
        cm = confusion_matrix(labels.numpy(), preds.numpy())
        fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, cmap="Blues", cbar=False, ax=ax_cm)
        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True"); ax_cm.set_title("Confusion Matrix")
        fp_cm = os.path.join(fig_dir, f"confmat_{timestamp}.pdf"); fig_cm.savefig(fp_cm, bbox_inches="tight"); plt.close(fig_cm)
        self.generated_figures.append(fp_cm)

        # Reliability Diagram
        fig_cal, ax_cal = plt.subplots(figsize=(5, 5))
        frac_pos, mean_pred = calibration_curve(labels.numpy() == preds.numpy(), probs.max(axis=1), n_bins=15)
        ax_cal.plot(mean_pred, frac_pos, marker='o'); ax_cal.plot([0, 1], [0, 1], '--');
        ax_cal.set_xlabel("Confidence"); ax_cal.set_ylabel("Accuracy"); ax_cal.set_title("Reliability Diagram")
        fp_cal = os.path.join(fig_dir, f"reliab_{timestamp}.pdf"); fig_cal.savefig(fp_cal, bbox_inches="tight"); plt.close(fig_cal)
        self.generated_figures.append(fp_cal)

        return metrics
