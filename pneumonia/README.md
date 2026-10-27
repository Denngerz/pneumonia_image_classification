# pneumonia/ — shared Python code (the "recipe")

Every model is trained and evaluated through these modules, so the comparison stays fair.
Notebooks and scripts **import** from here; they never copy this logic.

| Module | Responsibility |
|---|---|
| `data.py` | Dataset class (grayscale → resize 224 → 3 channels, optional in-memory cache of resized images), augmentation with `torchvision.transforms.v2`, ImageNet normalization, class-balancing weighted sampler; `build_dataloaders(config, data_dir, debug)` returns train/val/test loaders from `data/splits/*.csv` |
| `splits.py` | Patient-level, stratified train/val split of the official train set; official test set kept untouched |
| `models.py` | Model registry: custom baseline CNN + timm models (pinned ImageNet weight tags), all with a 2-class output head; Grad-CAM target layer per model |
| `train.py` | Shared training loop: cross-entropy (classes balanced by the sampler in `data.py`), AdamW, warm-up + cosine LR, AMP on GPU, early stopping on validation ROC-AUC, `history.csv`, checkpoints to `runs/` |
| `evaluate.py` | One evaluation for all models: `compute_metrics` (accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrix), inference timing, `evaluate_run` → `metrics.json` + `test_predictions.csv` |
| `compare.py` | Collects all `results/*/metrics.json`, builds the comparison table (mean ± std over seeds), the report figures, the cross-model Grad-CAM grid and the Grad-CAM border measure |
| `gradcam.py` | Grad-CAM heat maps on fixed test images + each model's mistakes → `gradcam.jpg`, `gradcam_fixed.npz` (target layers are listed in `models.py`) |
| `utils.py` | Seeding, device selection (GPU if available, else CPU), timers, config loading |

Imports work when commands are run from the repository root (no install step needed).
