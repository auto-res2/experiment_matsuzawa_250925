"""main.py – command-line interface stub.

Because the original script contained no runnable experiment, this CLI
acts purely as an informative front-end: it parses the required
``--smoke-test`` / ``--full-experiment`` flags and then immediately
terminates with an explanatory message.  This honour the directory and
file-structure contract while avoiding silent no-ops.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml  # PyYAML – declared in pyproject.toml

# Local imports – intentionally *not* used because the functions inside raise
# RuntimeError to indicate missing logic.  Importing them nonetheless ensures
# that potential circular-dependency issues surface early.
from . import evaluate  # noqa: F401  # pylint: disable=unused-import
from . import preprocess  # noqa: F401  # pylint: disable=unused-import
from . import train  # noqa: F401  # pylint: disable=unused-import

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_cfg(path: Path):
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError as exc:  # pragma: no cover – guard for misuse
        print(f"Configuration file not found: {path}", file=sys.stderr)
        raise SystemExit(1) from exc


def parse_args(argv: list[str] | None = None):  # noqa: D401
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Stub CLI generated because no runnable experiment was provided.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the (non-existent) smoke-test configuration.",
    )
    group.add_argument(
        "--full-experiment",
        action="store_true",
        help="Run the (non-existent) full experiment configuration.",
    )

    return parser.parse_args(argv)


def main() -> None:  # noqa: D401
    args = parse_args()

    cfg_file = CONFIG_DIR / (
        "smoke_test.yaml" if args.smoke_test else "full_experiment.yaml"
    )
    _ = _load_cfg(cfg_file)

    # With a proper implementation we would now orchestrate data loading,
    # training and evaluation.  Because none of these exist we inform the
    # user and exit gracefully.
    msg = (
        "No runnable experiment code was provided in the original script. "
        "All generated modules are therefore stubs that deliberately raise "
        "RuntimeError when invoked.  Please supply actual experiment logic "
        "to replace these stubs."
    )
    print(msg)


if __name__ == "__main__":  # pragma: no cover
    main()
