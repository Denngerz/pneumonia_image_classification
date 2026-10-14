# scripts/ — command-line entry points

Thin "buttons" that call code from `pneumonia/`. Run them from the repository root (locally or in a Kaggle notebook cell with `!`).

| Planned script | Purpose | Example |
|---|---|---|
| `make_splits.py` | Create `data/splits/*.csv` (run once) | `python scripts/make_splits.py --data-dir data/raw/chest_xray` |
| `train.py` | Train + evaluate one model from a config | `python scripts/train.py --config configs/models/resnet50.yaml --seed 42` |
| `compare_results.py` | Build the comparison table and plots from all runs | `python scripts/compare_results.py` |

All scripts accept `--data-dir` (so the same command works locally and on Kaggle) and `train.py` has a `--debug` mode
(few images, 1 epoch) for quick checks on CPU.
