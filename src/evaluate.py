import os
import json
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
from torch_scatter import scatter

# -----------------------------------------------------------------------------
# METRIC UTILITIES
# -----------------------------------------------------------------------------

def _accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    preds = logits.argmax(dim=-1)
    return (preds[mask] == labels[mask]).float().mean().item()


def _macro_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    preds = logits.argmax(dim=-1)[mask].cpu().numpy()
    labs = labels[mask].cpu().numpy()
    return f1_score(labs, preds, average="macro")


def _fairness_gap(preds: torch.Tensor, labels: torch.Tensor, edge_index: torch.Tensor,
                  mask: torch.Tensor):
    deg = scatter(torch.ones(edge_index.size(1), device=preds.device), edge_index[0], dim=0, dim_size=labels.size(0))
    quint = torch.quantile(deg.float(), torch.tensor([0.2, 0.8], device=deg.device))
    low_mask = (deg <= quint[0]) & mask
    high_mask = (deg >= quint[1]) & mask
    if low_mask.sum() == 0 or high_mask.sum() == 0:
        return 0.0, 0.0, 0.0
    low_acc = (preds[low_mask] == labels[low_mask]).float().mean().item()
    high_acc = (preds[high_mask] == labels[high_mask]).float().mean().item()
    return abs(low_acc - high_acc), low_acc, high_acc


# -----------------------------------------------------------------------------
# FIGURE HELPERS
# -----------------------------------------------------------------------------

def _annotate_line(ax, x, y):
    for xi, yi in zip(x, y):
        ax.text(xi, yi, f"{yi:.2f}")


def _lineplot(values: List[float], ylabel: str, title: str, out_file: str):
    x = list(range(1, len(values) + 1))
    plt.figure()
    plt.plot(x, values, marker="o", label=title)
    _annotate_line(plt.gca(), x, values)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


def _confusion_plot(conf: np.ndarray, cls_names: List[str], out_file: str):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues",
                xticklabels=cls_names, yticklabels=cls_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------

def evaluate_model(model, data, results: Dict, cfg: Dict):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))

    preds = logits.argmax(dim=-1)
    labels = data.y.to(device)

    acc = _accuracy(logits, labels, data.test_mask)
    f1 = _macro_f1(logits, labels, data.test_mask)
    gap, low_acc, high_acc = _fairness_gap(preds, labels, data.edge_index.to(device), data.test_mask)
    conf = confusion_matrix(labels[data.test_mask].cpu(), preds[data.test_mask].cpu())

    results.update({
        "test_accuracy": acc,
        "test_macro_f1": f1,
        "fairness_gap": gap,
        "low_deg_acc": low_acc,
        "high_deg_acc": high_acc,
        "confusion_matrix": conf.tolist()
    })

    save_dir = cfg["evaluation"]["save_dir"]
    _lineplot(results["train_loss"], "Loss", "Training Loss", os.path.join(save_dir, "training_loss.pdf"))
    _lineplot(results["val_acc"], "Accuracy", "Validation Accuracy", os.path.join(save_dir, "accuracy.pdf"))
    _confusion_plot(conf, [str(c) for c in range(conf.shape[0])],
                    os.path.join(save_dir, "confusion_matrix.pdf"))

    json_path = os.path.join(save_dir, f"{cfg['experiment_name']}_results.json")
    with open(json_path, "w") as fp:
        json.dump(results, fp, indent=2)

    return results