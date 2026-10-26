# results/ — small experiment outputs (committed)

One folder per run, named `<model>_seed<seed>` (e.g. `resnet50_seed42/`), containing:

| File | Contents |
|---|---|
| `config.yaml` | Exact merged settings used for the run |
| `history.csv` | Loss and validation metrics per epoch |
| `metrics.json` | Test metrics (accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrix), parameter count, best epoch, training time, inference ms/image (batch 1, median), GPU name |
| `test_predictions.csv` | PNEUMONIA probability and prediction for every test image — used for ROC curves and averaging over seeds |
| `gradcam.jpg` | Grad-CAM heat maps: 8 fixed test images (same for every run) + the model's most confident mistakes |
| `gradcam_fixed.npz` | The heat maps of the 8 fixed images as arrays, to build side-by-side comparisons of all models |

Also here:
- `comparison.md` / `comparison.csv` — per-model mean ± std over seeds (primary metric: ROC-AUC), and
  `comparison_runs.csv` — one row per run; all built by `scripts/compare_results.py`
- `figures/comparison_*.png` — metric bars, ROC curves, training curves, confusion matrices, quality vs speed
  (shown in `notebooks/03_results.ipynb`)
- `figures/gradcam_comparison_seed42.jpg` — Grad-CAM of all 6 models on the same 8 test images (needs the dataset:
  `python scripts/compare_results.py --data-dir data/raw/chest_xray`)
- `gradcam_border_share.csv` + `figures/gradcam_border_share.png` — share of Grad-CAM heat on the outer 10 % of the
  image per model (a simple shortcut check; 0.36 = evenly spread map)
- `figures/` — PNG plots used in the report (curves, confusion matrices, ROC, Grad-CAM)

Large outputs (model checkpoints, TensorBoard logs) go to `runs/`, which is **not** committed.
After a Kaggle run, download the run's `results/<run>/` folder from the notebook output and commit it.
