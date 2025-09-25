#!/usr/bin/env python
"""Entry-point for CurvTrack experiments.

USAGE
-----
Smoke-test only
    uv run python -m src.main --smoke-test

Full experiment (runs smoke-test first)
    uv run python -m src.main --full-experiment
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import yaml

from .evaluate import evaluate
from .preprocess import build_dataloaders
from .train import CurvTrack, collect_norm_blocks

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
RESULT_DIR = Path(".research/iteration1")
IMG_DIR = RESULT_DIR / "images"
for _d in (RESULT_DIR, IMG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def load_cfg(fname: str) -> dict:
    with open(CONFIG_DIR / fname, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_experiment(cfg: dict, device: torch.device) -> Dict[str, float]:
    import timm  # local import to keep package list minimal at load-time

    # data
    train_loader, val_loader = build_dataloaders(cfg)

    # model + CurvTrack wrapper
    model = timm.create_model(cfg["model_name"], pretrained=True).to(device)
    blocks = collect_norm_blocks(model, cfg.get("block_size", 16), device)
    # Print debug info about blocks found
    print(f"Found {len(blocks)} normalization blocks for adaptation")
    print(f"Model: {cfg['model_name']} for dataset: {cfg['dataset']}")
    # Keep model in eval mode but ensure norm layers can still compute gradients
    model.eval()
    adaptor = CurvTrack(model, blocks).to(device)

    # only evaluation pass (no training loop in this demo)
    results = evaluate(adaptor, val_loader, device)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CurvTrack experimental runner")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="run quick validation run")
    g.add_argument("--full-experiment", action="store_true", help="run full experiment (after smoke test)")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Always run smoke first
    smoke_cfg = load_cfg("smoke_test.yaml")
    print("=== Running smoke-test ===")
    smoke_res = run_experiment(smoke_cfg, device)
    threshold = smoke_cfg.get("smoke_test_pass_threshold", 0.25)
    print(f"Smoke test accuracy: {smoke_res['accuracy']:.4f}, threshold: {threshold}")
    passed = smoke_res["accuracy"] >= threshold
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    smoke_path = RESULT_DIR / f"smoke_results_{stamp}.json"
    with open(smoke_path, "w", encoding="utf-8") as f:
        json.dump(smoke_res, f, indent=2)
    print(json.dumps(smoke_res, indent=2))

    if args.smoke_test:
        return
    if not passed:
        print("Smoke test FAILED – aborting full experiment", file=sys.stderr)
        sys.exit(1)

    # full run
    print("=== Smoke-test passed – running full experiment ===")
    full_cfg = load_cfg("full_experiment.yaml")
    full_res = run_experiment(full_cfg, device)
    full_path = RESULT_DIR / f"full_results_{stamp}.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_res, f, indent=2)
    print(json.dumps(full_res, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
