import argparse
import yaml
from pathlib import Path
import random, numpy as np, torch

from .train import Trainer


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run(cfg: dict):
    _set_seeds(int(cfg.get("seed", 0)))
    Trainer(cfg).run()


def main():
    parser = argparse.ArgumentParser("FLASH-TTA research – unified runner")
    parser.add_argument("--config", type=str, required=False, help="YAML config file path")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick FakeData test")
    parser.add_argument("--full-experiment", action="store_true", help="Run bundled full experiment YAML")
    args = parser.parse_args()

    if args.smoke_test and args.full_experiment:
        raise ValueError("Select at most one of --smoke-test / --full-experiment")

    if args.smoke_test:
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "smoke_test.yaml"
    elif args.full_experiment:
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "full_experiment.yaml"
    elif args.config:
        cfg_path = Path(args.config)
    else:
        parser.error("No configuration provided – use --config or predefined flags")

    cfg = _load_yaml(str(cfg_path))

    # Support YAML with a list under experiments:
    if "experiments" in cfg:
        base = {k: v for k, v in cfg.items() if k != "experiments"}
        for exp in cfg["experiments"]:
            merged = {**base, **exp}
            _run(merged)
    else:
        _run(cfg)


if __name__ == "__main__":
    main()
