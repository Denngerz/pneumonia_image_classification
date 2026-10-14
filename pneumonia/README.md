# pneumonia/ — shared Python code (the "recipe")

Every model is trained and evaluated through these modules, so the comparison stays fair.
Notebooks and scripts **import** from here; they never copy this logic.

| Planned module | Responsibility |
|---|---|
| `data.py` | Dataset class, transforms (resize 224, 3 channels, ImageNet normalization, augmentation), DataLoaders built from `data/splits/*.csv` |
| `splits.py` | Patient-level, stratified train/val split of the official train set; official test set kept untouched |
| `models.py` | Model registry: custom baseline CNN + timm models, all with a 2-class output head |
| `train.py` | Shared training loop: class-weighted loss, optimizer, LR schedule, early stopping, logging, checkpoints to `runs/` |
| `evaluate.py` | One evaluation function for all models: accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrix, timing |
| `gradcam.py` | Grad-CAM heatmaps |
| `utils.py` | Seeding, device selection (GPU if available, else CPU), timers, config loading |

Imports work when commands are run from the repository root (no install step needed).
