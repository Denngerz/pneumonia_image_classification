# scripts/ — command-line entry points

Thin "buttons" that call code from `pneumonia/`. Run them from the repository root (locally or in a Kaggle notebook cell with `!`).

| Script | Purpose | Example |
|---|---|---|
| `make_splits.py` | Create `data/splits/*.csv` (run once; re-running gives byte-identical files) | `python scripts/make_splits.py --data-dir data/raw/chest_xray` |
| `train.py` | Train one model, evaluate it on the test set, make Grad-CAM pictures | `python scripts/train.py --config configs/models/resnet50.yaml --seed 42 --data-dir data/raw/chest_xray` |
| `compare_results.py` | Comparison table + report figures from all runs in `results/` (`--data-dir` only for the Grad-CAM image grid) | `python scripts/compare_results.py --data-dir data/raw/chest_xray` |

`make_splits.py` and `train.py` need `--data-dir` (so the same command works locally and on Kaggle); `train.py` has a
`--debug` mode (16 images per class, 1 epoch, ~1 minute on a laptop CPU) for quick checks.

`train.py` writes `config.yaml` and `history.csv` to `results/<model>_seed<seed>/` and `best.pt`, `last.pt` and TensorBoard
logs to `runs/<model>_seed<seed>/`. Debug runs write to `runs/debug/` instead, so they never end up in `results/`.
Change the folders with `--results-dir` / `--runs-dir` (e.g. `/kaggle/working/...` on Kaggle).
