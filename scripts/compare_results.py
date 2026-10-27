"""Build the comparison table (results/comparison.md, .csv) and the report figures from all runs in results/.

Examples:
    python scripts/compare_results.py
    python scripts/compare_results.py --data-dir data/raw/chest_xray     # also draws the Grad-CAM image grid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pneumonia.compare import build_gradcam_report, build_report, summary_markdown  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Dataset folder (with test/); needed only for the Grad-CAM image grid")
    parser.add_argument("--gradcam-seed", type=int, default=42, help="Which seed's runs to show in the Grad-CAM grid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_report(args.results_dir, args.figures_dir)
    border = build_gradcam_report(args.results_dir, args.figures_dir, args.data_dir, args.gradcam_seed)
    print(summary_markdown(summary))
    print(border.round(3).to_string(index=False))
    print(f"Tables: {args.results_dir}/comparison.md, comparison.csv, comparison_runs.csv, gradcam_border_share.csv")
    print(f"Figures: {args.figures_dir}/comparison_*.png, gradcam_*")


if __name__ == "__main__":
    main()
