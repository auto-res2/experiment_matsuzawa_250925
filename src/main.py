"""src/main.py – entry-point with updated model factory for HF Transformers & ECO_GAT"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from preprocess import build_dataloaders
from train import Trainer, ECO_LASQ_GAT
from evaluate import evaluate_model, generate_figures, save_and_print_results

# -----------------------------------------------------------------------------
#  Reproducibility helpers -----------------------------------------------------
# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
#  Dynamic model factory -------------------------------------------------------
# -----------------------------------------------------------------------------


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg["model"]
    name = model_cfg.get("name")

    # 1) HF sequence classification model ------------------------------------
    if name.startswith("hf:"):
        from transformers import AutoModelForSequenceClassification

        checkpoint = name.split("hf:")[1]
        num_labels = model_cfg.get("num_labels", 10)
        return AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

    # 2) ECO-LASQ-GAT ---------------------------------------------------------
    if name == "eco_gat:ECO_LASQ_GAT":
        return ECO_LASQ_GAT(
            num_feats=model_cfg.get("num_feats", 128),
            num_classes=model_cfg.get("num_classes", 10),
            hidden=model_cfg.get("hidden", 128),
            num_layers=model_cfg.get("num_layers", 2),
            lam_E=model_cfg.get("lam_E", 0.0),
        )

    # 3) Simple MLP fallback ---------------------------------------------------
    if name == "MODEL_PLACEHOLDER":
        in_features = model_cfg.get("in_features", 32)
        n_classes = model_cfg.get("n_classes", 2)
        hidden = model_cfg.get("hidden", 64)
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_classes),
        )

    # 4) Arbitrary dotted-path import -----------------------------------------
    module_path, class_name = name.split(":")
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)
    return model_cls(**model_cfg.get("args", {}))


# -----------------------------------------------------------------------------
#  Argument parsing -----------------------------------------------------------
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ECO-LASQ Experimental Runner")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    group.add_argument("--full-experiment", action="store_true", help="Run full experiment as per YAML config")
    p.add_argument("--config", type=str, default=None, help="YAML config path override")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# -----------------------------------------------------------------------------
#  Main -----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg_path = (
        args.config
        or Path(__file__).parent.parent
        / "config"
        / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")
    )
    with open(cfg_path, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    out_root = Path(cfg.get("output_dir", "runs")); out_root.mkdir(parents=True, exist_ok=True)
    exp_name = cfg.get("experiment_name", "experiment") + time.strftime("_%Y%m%d-%H%M%S")
    out_dir = out_root / exp_name; out_dir.mkdir(parents=True)

    # ------------------------------------------------------------------
    seed_everything(args.seed)

    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg)

    # Optimiser & scheduler -------------------------------------------
    optim_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optim_cfg.get("lr", 1e-3), weight_decay=optim_cfg.get("weight_decay", 0.0)
    )
    scheduler = None
    sched_cfg = cfg.get("scheduler", {})
    if sched_cfg.get("type") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.get("training", {}).get("num_epochs", 10)
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=out_dir,
        num_epochs=cfg.get("training", {}).get("num_epochs", 10),
        early_stop_patience=cfg.get("training", {}).get("early_stop_patience"),
        mixed_precision=cfg.get("training", {}).get("mixed_precision", False),
    )

    logs = trainer.run()
    eval_results = evaluate_model(trainer.model, test_loader, device)
    figs = generate_figures(logs, eval_results, out_dir, exp_name)
    save_and_print_results(cfg.get("description", ""), logs, eval_results, figs, out_dir)


if __name__ == "__main__":
    main()
