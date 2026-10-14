# results/ — small experiment outputs (committed)

One folder per run, named `<model>_seed<seed>` (e.g. `resnet50_seed42/`), containing:

| File | Contents |
|---|---|
| `config.yaml` | Exact merged settings used for the run |
| `history.csv` | Loss and validation metrics per epoch |
| `metrics.json` | Final test metrics, parameter count, training and inference time |

Also here:
- `comparison.csv` / `comparison.md` — table of all runs, built by `scripts/compare_results.py`
- `figures/` — PNG plots used in the report (curves, confusion matrices, ROC, Grad-CAM)

Large outputs (model checkpoints, TensorBoard logs) go to `runs/`, which is **not** committed.
After a Kaggle run, download the run's `results/<run>/` folder from the notebook output and commit it.
