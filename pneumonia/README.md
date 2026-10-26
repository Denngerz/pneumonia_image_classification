# pneumonia/ — shared Python code (the "recipe")

Every model is trained and evaluated through these modules, so the comparison stays fair.
Notebooks and scripts **import** from here; they never copy this logic.

| Planned module | Responsibility |
|---|---|
| `data.py` | Dataset class (grayscale → resize 224 → 3 channels, optional in-memory cache of resized images), augmentation with `torchvision.transforms.v2`, ImageNet normalization, class-balancing weighted sampler; `build_dataloaders(config, data_dir, debug)` returns train/val/test loaders from `data/splits/*.csv` |
| `splits.py` | Patient-level, stratified train/val split of the official train set; official test set kept untouched |
| `models.py` | Model registry: custom baseline CNN + timm models, all with a 2-class output head |
| `train.py` | Shared training loop: class-weighted loss, optimizer, LR schedule, early stopping, logging, checkpoints to `runs/` |
| `evaluate.py` | One evaluation for all models: `compute_metrics` (accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrix), inference timing, `evaluate_run` → `metrics.json` + `test_predictions.csv` |
| `compare.py` | Collects all `results/*/metrics.json`, builds the comparison table (mean ± std over seeds) and the report figures |
| `gradcam.py` | Grad-CAM heat maps on fixed test images + each model's mistakes → `gradcam.jpg`, `gradcam_fixed.npz` (target layers are listed in `models.py`) |
| `utils.py` | Seeding, device selection (GPU if available, else CPU), timers, config loading |

Imports work when commands are run from the repository root (no install step needed).
