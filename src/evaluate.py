"""src/evaluate.py – unchanged except for support of HF model outputs"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader

sns.set_theme(style="whitegrid")


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Compute accuracy / loss; supports both tensor and HF dict inputs."""

    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, running_loss = 0, 0, 0.0
    for batch in loader:
        # parse batch ---------------------------------------------
        if isinstance(batch, dict):
            targets = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
        else:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
        loss = criterion(logits, targets)
        running_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return {
        "test_loss": running_loss / total,
        "test_accuracy": correct / total,
    }

# --------------- plotting helpers (unchanged) -------------------------------
from typing import Any

def _annotate_line(ax, x: List[Any], y: List[Any], fmt: str = "{:.2f}") -> None:
    for xi, yi in zip(x, y):
        ax.annotate(fmt.format(yi), (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

def _annotate_bar(ax) -> None:
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2.0, height),
                    ha='center', va='bottom', fontsize=8)


def generate_figures(
    logs: Dict[str, List[float]],
    eval_results: Dict[str, float],
    output_dir: Path,
    experiment_name: str,
) -> List[str]:
    """Creates PDF figures (loss / acc / energy)."""

    figure_paths: List[str] = []
    epochs = list(range(1, len(logs["train_loss"]) + 1))

    # -------- Loss -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, logs["train_loss"], label="train")
    ax.plot(epochs, logs["val_loss"], label="val")
    _annotate_line(ax, epochs, logs["val_loss"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs. Validation Loss")
    ax.legend()
    path = output_dir / f"training_loss_{experiment_name}.pdf"
    fig.tight_layout(); fig.savefig(path)
    plt.close(fig)
    figure_paths.append(path.name)

    # -------- Accuracy -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [100 * x for x in logs["train_acc"]], label="train")
    ax.plot(epochs, [100 * x for x in logs["val_acc"]], label="val")
    _annotate_line(ax, epochs, [100 * x for x in logs["val_acc"]], fmt="{:.1f}%")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training vs. Validation Accuracy"); ax.legend()
    path = output_dir / f"accuracy_{experiment_name}.pdf"
    fig.tight_layout(); fig.savefig(path)
    plt.close(fig)
    figure_paths.append(path.name)

    # -------- Energy ---------------------------------------------------------
    if any(logs["epoch_energy_j"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(epochs, logs["epoch_energy_j"], color="tab:green")
        _annotate_bar(ax)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Energy (J)")
        ax.set_title("Energy Consumption Per Epoch")
        path = output_dir / f"energy_{experiment_name}.pdf"
        fig.tight_layout(); fig.savefig(path)
        plt.close(fig)
        figure_paths.append(path.name)
    return figure_paths


def save_and_print_results(
    description: str,
    logs: Dict[str, List[float]],
    eval_results: Dict[str, float],
    figure_files: List[str],
    output_dir: Path,
) -> None:
    """Write JSON + pretty stdout."""

    results = {
        "description": description,
        "logs": logs,
        "eval_results": eval_results,
        "figures": figure_files,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # stdout ------------------------------------------------------------------
    print("=" * 80)
    print("EXPERIMENT DESCRIPTION:\n" + description)
    print("-" * 80)
    print("NUMERICAL RESULTS:")
    to_display = {**{k: v[-1] for k, v in logs.items() if isinstance(v, list)}, **eval_results}
    for k, v in to_display.items():
        if isinstance(v, float):
            print(f"  {k:>20}: {v:.4f}")
        else:
            print(f"  {k:>20}: {v}")
    print("-" * 80)
    print("Figures:")
    for f in figure_files:
        print("  ", f)
    print("Results saved to:", json_path)
    print("=" * 80)
