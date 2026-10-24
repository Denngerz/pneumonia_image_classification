"""Create data/splits/{train,val,test}.csv from the downloaded dataset (run once, results are committed).

Example:
    python scripts/make_splits.py --data-dir data/raw/chest_xray
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pneumonia.splits import make_splits, save_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder that contains the official train/, val/ and test/ folders")
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = make_splits(args.data_dir, args.val_fraction, args.seed)
    save_splits(splits, args.output_dir)
    for name, df in splits.items():
        counts = df["class_name"].value_counts().to_dict()
        print(f"{name:5s}: {len(df):5d} images, {df['patient_id'].nunique():5d} patients, {counts}")


if __name__ == "__main__":
    main()
