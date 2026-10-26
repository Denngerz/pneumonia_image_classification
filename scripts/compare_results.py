"""Build the comparison table (results/comparison.md, .csv) and the report figures from all runs in results/.

Example:
    python scripts/compare_results.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pneumonia.compare import build_report, summary_markdown  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_report(args.results_dir, args.figures_dir)
    print(summary_markdown(summary))
    print(f"Tables: {args.results_dir}/comparison.md, comparison.csv, comparison_runs.csv")
    print(f"Figures: {args.figures_dir}/comparison_*.png")


if __name__ == "__main__":
    main()
