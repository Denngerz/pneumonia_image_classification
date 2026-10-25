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
- `comparison.csv` / `comparison.md` — table of all runs, built by `scripts/compare_results.py`
- `figures/` — PNG plots used in the report (curves, confusion matrices, ROC, Grad-CAM)

Large outputs (model checkpoints, TensorBoard logs) go to `runs/`, which is **not** committed.
After a Kaggle run, download the run's `results/<run>/` folder from the notebook output and commit it.
