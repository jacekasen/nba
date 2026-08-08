"""Run the end-to-end ML pipeline in one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ml.config import CURRENT_PREDICTIONS_PATH, MODEL_FEATURES_PATH, PLAYER_SEASONS_PATH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to invoke module commands.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training and reuse existing artifacts.",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip current-player prediction generation.",
    )
    return parser.parse_args()


def _run_module(python_executable: Path, module_name: str) -> None:
    cmd = [str(python_executable), "-m", module_name]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    """CLI entry point for running the full pipeline."""
    args = _parse_args()

    _run_module(args.python_executable, "ml.data")
    _run_module(args.python_executable, "ml.features")

    if not args.skip_train:
        _run_module(args.python_executable, "ml.train")
    else:
        print("Skipping training (--skip-train).")

    if not args.skip_predict:
        _run_module(args.python_executable, "ml.predict")
    else:
        print("Skipping prediction generation (--skip-predict).")

    print("=== End-to-End Pipeline Complete ===")
    print(f"Dataset path: {PLAYER_SEASONS_PATH}")
    print(f"Feature path: {MODEL_FEATURES_PATH}")
    if not args.skip_predict:
        print(f"Prediction path: {CURRENT_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
