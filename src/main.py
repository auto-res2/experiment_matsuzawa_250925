"""src/main.py – unchanged apart from default full-experiment path."""
from __future__ import annotations
import argparse, yaml, time
from pathlib import Path

from .train import train

DEFAULT_SMOKE = Path(__file__).parent.parent / "config/smoke_test.yaml"
DEFAULT_FULL  = Path(__file__).parent.parent / "config/full_experiment.yaml"


def _args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true")
    g.add_argument("--full-experiment", action="store_true")
    g.add_argument("--config", type=Path)
    return p.parse_args()


def main():
    a = _args()
    if a.smoke_test:
        cfg_path = DEFAULT_SMOKE
    elif a.full_experiment:
        cfg_path = DEFAULT_FULL
    else:
        cfg_path = a.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run_name = cfg.get("run_name", f"run_{int(time.time())}")
    train(cfg, run_name)

if __name__ == "__main__":
    main()
