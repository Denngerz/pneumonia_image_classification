# scripts/ — command-line entry points

Thin "buttons" that call code from `pneumonia/`. Run them from the repository root (locally or in a Kaggle notebook cell with `!`).

| Planned script | Purpose | Example |
|---|---|---|
| `make_splits.py` | Create `data/splits/*.csv` (run once) | `python scripts/make_splits.py --data-dir data/raw/chest_xray` |
| `train.py` | Train + evaluate one model from a config | `python scripts/train.py --config configs/models/resnet50.yaml --seed 42` |
| `compare_results.py` | Build the comparison table and plots from all runs | `python scripts/compare_results.py` |

All scripts accept `--data-dir` (so the same command works locally and on Kaggle) and `train.py` has a `--debug` mode
(16 images per class, 1 epoch, ~1 minute on a laptop CPU) for quick checks.

`train.py` writes `config.yaml` and `history.csv` to `results/<model>_seed<seed>/` and `best.pt`, `last.pt` and TensorBoard
logs to `runs/<model>_seed<seed>/`. Debug runs write to `runs/debug/` instead, so they never end up in `results/`.
Change the folders with `--results-dir` / `--runs-dir` (e.g. `/kaggle/working/...` on Kaggle).
