import argparse
import json
import os
import textwrap
from pathlib import Path

import yaml

from . import preprocess as preprocess_module
from . import train as train_module
from . import evaluate as evaluate_module

# -----------------------------------------------------------------------------
# CONFIG LOADER
# -----------------------------------------------------------------------------

def _load_yaml(path: str):
    with open(path, "r") as fp:
        return yaml.safe_load(fp)

# -----------------------------------------------------------------------------
# CLI  –  unified entry-point
# -----------------------------------------------------------------------------

def cli():
    parser = argparse.ArgumentParser(description="HAQ-GAT Experiment Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--full-experiment", action="store_true", help="Run full reference experiment")
    parser.add_argument("--config", type=str, help="Path to custom YAML config")
    args = parser.parse_args()

    if args.smoke_test:
        cfg_path = Path("config/smoke_test.yaml")
    elif args.full_experiment:
        cfg_path = Path("config/full_experiment.yaml")
    elif args.config:
        cfg_path = Path(args.config)
    else:
        raise ValueError("Specify one of --smoke-test / --full-experiment / --config <file>.")

    cfg = _load_yaml(cfg_path)

    desc = textwrap.dedent(f"""
    =========================================================
    Experiment Name : {cfg['experiment_name']}
    Dataset         : {cfg['dataset']['name']}
    Model           : {cfg['model']['name']}
    =========================================================
    """)
    print(desc)

    data = preprocess_module.preprocess_data(cfg)
    results, model = train_module.train_pipeline(cfg, data)
    results = evaluate_module.evaluate_model(model, data, results, cfg)

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    cli()