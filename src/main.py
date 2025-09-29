"""src/main.py
Main execution script with flexible configuration system.

CLI usage:
    uv run python -m src.main --smoke-test
    uv run python -m src.main --full-experiment
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

from src.evaluate import Evaluator
from src.preprocess import StandardScalerPreprocessor
from src.train import Trainer

# -------------------------------------------------------------
# Helper – deterministic behaviour
# -------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------
# Placeholder dataset + model factories
# -------------------------------------------------------------

class DummyClassificationDataset(Dataset):
    """A synthetic dataset for smoke-testing when placeholders are not replaced."""

    def __init__(self, num_samples: int, num_features: int, num_classes: int):
        self.X = torch.randn(num_samples, num_features)
        self.y = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds_cfg = cfg["dataset"]
    if ds_cfg["name"] == "DATASET_PLACEHOLDER":  # PLACEHOLDER: replace with real dataset loader
        dataset: Dataset = DummyClassificationDataset(
            num_samples=int(ds_cfg.get("num_samples", 1000)),
            num_features=int(ds_cfg.get("num_features", 32)),
            num_classes=int(ds_cfg.get("num_classes", 2)),
        )
    else:
        raise NotImplementedError("Dataset loading logic must be implemented for actual dataset.")

    # Preprocessing (fit on full dataset then split)
    scaler = StandardScalerPreprocessor()
    dataset_scaled, _ = scaler.fit_transform(dataset)

    # Train/Val/Test split
    total_len = len(dataset_scaled)
    val_len = int(0.15 * total_len)
    test_len = int(0.15 * total_len)
    train_len = total_len - val_len - test_len
    train_ds, val_ds, test_ds = random_split(
        dataset_scaled, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(cfg["seed"])
    )
    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


class PlaceholderFFN(nn.Module):
    """Simple feed-forward network acting as MODEL_PLACEHOLDER."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def get_model(cfg: Dict, input_dim: int) -> nn.Module:
    m_cfg = cfg["model"]
    if m_cfg["name"] == "MODEL_PLACEHOLDER":  # PLACEHOLDER: replace with actual model builder
        hidden_dim = int(m_cfg.get("hidden_dim", 64))
        num_classes = int(cfg["dataset"].get("num_classes", 2))
        model = PlaceholderFFN(input_dim, hidden_dim, num_classes)
    else:
        raise NotImplementedError("Model construction must be implemented for actual model.")
    return model


# -------------------------------------------------------------
# Plotting utilities (publication-quality .pdf)
# -------------------------------------------------------------

IMAGES_DIR = Path(".research/iteration2/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_curves(history: Dict[str, list], run_name: str) -> Path:
    sns.set(style="whitegrid")
    # Loss curve
    fig, ax = plt.subplots()
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    # Annotate final loss
    ax.annotate(f"{history['train_loss'][-1]:.4f}", (epochs[-1], history["train_loss"][-1]))
    ax.legend()
    loss_path = IMAGES_DIR / f"training_loss_{run_name}.pdf"
    fig.savefig(loss_path, bbox_inches="tight")
    plt.close(fig)

    # Val accuracy curve (if present)
    acc_key = next(k for k in history.keys() if k.startswith("val_"))
    fig, ax = plt.subplots()
    ax.plot(epochs, history[acc_key], label=acc_key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(acc_key.split("val_")[1].capitalize())
    ax.set_title("Validation Curve")
    ax.annotate(f"{history[acc_key][-1]:.4f}", (epochs[-1], history[acc_key][-1]))
    ax.legend()
    acc_path = IMAGES_DIR / f"val_{acc_key}_{run_name}.pdf"
    fig.savefig(acc_path, bbox_inches="tight")
    plt.close(fig)

    return loss_path, acc_path


def plot_confusion_matrix(cm: list, class_names: list, run_name: str) -> Path:
    cm_tensor = torch.tensor(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_tensor, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    cm_path = IMAGES_DIR / f"confusion_matrix_{run_name}.pdf"
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    return cm_path


# -------------------------------------------------------------
# Main routine
# -------------------------------------------------------------

RESULTS_DIR = Path(".research/iteration2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def run_experiment(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)

    # ------------------------------------------------------------------
    # Experiment description (printed before numerical results)
    # ------------------------------------------------------------------
    description = (
        "Experiment Description:\n" + json.dumps(cfg, indent=2)
    )
    print(description)

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # ------------------------------------------------------------------
    # Data & Model
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    input_dim = train_loader.dataset[0][0].numel()
    model = get_model(cfg, input_dim)

    # ------------------------------------------------------------------
    # Optimiser & Loss
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=cfg,
    )
    history = trainer.fit()

    # ------------------------------------------------------------------
    # Evaluation on *best* checkpoint
    # ------------------------------------------------------------------
    best_model = trainer.load_best_model()
    evaluator = Evaluator(device=torch.device("cpu"), metrics=cfg["evaluation"]["metrics"])
    test_results = evaluator.evaluate(best_model, test_loader)

    # ------------------------------------------------------------------
    # Figures & Saving
    # ------------------------------------------------------------------
    run_name = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    loss_fig, acc_fig = plot_training_curves(history, run_name)
    cm_fig = plot_confusion_matrix(
        cm=test_results.pop("confusion_matrix"),
        class_names=[str(i) for i in range(cfg["dataset"].get("num_classes", 2))],
        run_name=run_name,
    )

    # ------------------------------------------------------------------
    # Results JSON
    # ------------------------------------------------------------------
    results_to_save = {
        "config": cfg,
        "history": history,
        "test_metrics": test_results,
        "figures": {
            "training_loss": str(loss_fig),
            "val_curve": str(acc_fig),
            "confusion_matrix": str(cm_fig),
        },
    }
    results_path = RESULTS_DIR / f"results_{run_name}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_to_save, f, indent=2)

    # ------------------------------------------------------------------
    # Stdout – numerical results & figure names
    # ------------------------------------------------------------------
    print("\nNumerical Results:")
    print(json.dumps(test_results, indent=2))
    print("\nFigures generated:")
    print(" -", loss_fig.name)
    print(" -", acc_fig.name)
    print(" -", cm_fig.name)


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Common Core Experiment Runner")
    mode_grp = parser.add_mutually_exclusive_group(required=True)
    mode_grp.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    mode_grp.add_argument("--full-experiment", action="store_true", help="Run full experiment")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_file = Path("config/smoke_test.yaml") if args.smoke_test else Path("config/full_experiment.yaml")
    run_experiment(cfg_file)