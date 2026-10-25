"""Train one model with the shared protocol.

Examples:
    python scripts/train.py --config configs/models/resnet50.yaml --seed 42 --data-dir data/raw/chest_xray
    python scripts/train.py --config configs/models/baseline_cnn.yaml --data-dir data/raw/chest_xray --debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pneumonia.data import load_split_tables  # noqa: E402
from pneumonia.evaluate import evaluate_run  # noqa: E402
from pneumonia.gradcam import run_gradcam  # noqa: E402
from pneumonia.train import fit  # noqa: E402
from pneumonia.utils import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Model config, e.g. configs/models/resnet50.yaml")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder that contains the official train/, val/ and test/ folders")
    parser.add_argument("--seed", type=int, default=None, help="Overrides the seed from configs/base.yaml")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Small outputs: config.yaml, history.csv, metrics.json, Grad-CAM (committed)")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"),
                        help="Large outputs: checkpoints, TensorBoard logs (not committed)")
    parser.add_argument("--debug", action="store_true",
                        help="Few images, 1 epoch — a quick check that everything runs (results go to runs/debug/)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    results_dir = args.runs_dir / "debug" if args.debug else args.results_dir
    run = fit(config, args.data_dir, results_dir, args.runs_dir, debug=args.debug)
    test_df = load_split_tables(config, debug=args.debug)["test"]
    metrics = evaluate_run(run, config, test_df)
    run_gradcam(run, config, test_df)
    test = metrics["test"]
    print(f"Test: accuracy {test['accuracy']:.4f} | recall {test['recall']:.4f} | "
          f"specificity {test['specificity']:.4f} | F1 {test['f1']:.4f} | ROC-AUC {test['roc_auc']:.4f} | "
          f"{metrics['inference_ms_median']:.1f} ms/image")
    print(f"Done: best epoch {run['best_epoch']} of {run['epochs_trained']}, "
          f"{run['training_seconds'] / 60:.1f} min. Outputs: {run['results_dir']} and {run['runs_dir']}")


if __name__ == "__main__":
    main()
