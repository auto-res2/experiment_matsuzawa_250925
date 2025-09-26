"""src/main.py
Entry point for running smoke tests or full experiments via the CLI.
Usage:
  uv run python -m src.main --smoke-test
  uv run python -m src.main --full-experiment
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import torch
import yaml

from . import preprocess as preprocess_module  # noqa: F401  (for explicit import style)
from .preprocess import load_datasets
from .train import Trainer, model_factory
from .evaluate import Evaluator

# -------------------------
# CONFIGURATION UTILITIES
# -------------------------

def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="COMMON CORE FOUNDATION Experiment Runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run quick smoke-test experiment")
    group.add_argument("--full-experiment", action="store_true", help="Run full experiment")
    return parser.parse_args()


# ---------------------
# MAIN EXECUTION LOGIC
# ---------------------

def main():
    args = parse_args()
    cfg_path = (
        Path("config/smoke_test.yaml")
        if args.smoke_test
        else Path("config/full_experiment.yaml")
    )
    cfg = load_config(cfg_path)

    # Override random seeds for reproducibility
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Data preparation
    train_loader, val_loader, test_loader, input_dim, num_classes = load_datasets(cfg)

    # Model & trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory(cfg.get("model", {}), input_dim, num_classes)

    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    history = trainer.fit()

    # Save model & training history
    experiment_name = cfg.get("experiment_name", "experiment")
    history_path = Path(cfg.get("results_dir", "results")) / f"{experiment_name}_history.json"
    with open(history_path, "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)
    trainer.save_model(f"{experiment_name}_model.pt")

    # Evaluate
    evaluator = Evaluator(trainer.model, test_loader, device, Path(cfg.get("results_dir", "results")))
    results = evaluator.evaluate(experiment_name)

    # ----------------------------------------------------------------------------------
    # STDOUT REPORTING (MANDATED BY SPEC)
    # ----------------------------------------------------------------------------------
    print("\n================== EXPERIMENT DESCRIPTION ==================")
    print(f"Experiment Name: {experiment_name}")
    print("Configuration:")
    print(json.dumps(cfg, indent=2))

    print("\n================== EXPERIMENTAL NUMERICAL DATA =============")
    print(json.dumps(results, indent=2))

    print("\n================== FIGURE FILES GENERATED ==================")
    figures = [f for f in Path(cfg.get("results_dir", "results")).glob("*.pdf")]
    for f in figures:
        print(f.name)

    # The main function always ends silently otherwise.


if __name__ == "__main__":
    main()
