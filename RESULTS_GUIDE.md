# Results guide — where to find the training results

Everything below is committed in this repository, except the trained model weights, which live on Kaggle (section 5).
Start at the top and go down only as far as you need.

## 1. The results on one page

- [results/SUMMARY.md](results/SUMMARY.md) — main numbers, the recommended model, the findings and which figures to use in
  the report.

Full explanation of how everything was done (code, maths, hyperparameters, models, defence Q&A):
[docs/PROJECT_EXPLAINED_EN.md](docs/PROJECT_EXPLAINED_EN.md) · [docs/PROJECT_EXPLAINED_UK.md](docs/PROJECT_EXPLAINED_UK.md).

## 2. Comparison of all models

| File | What is inside |
|---|---|
| [results/comparison.md](results/comparison.md) | Table "mean ± std over 3 seeds" per model: ROC-AUC, accuracy, recall, specificity, precision, F1, training time, ms per image |
| [results/comparison.csv](results/comparison.csv) | The same table for Excel / pandas |
| [results/comparison_runs.csv](results/comparison_runs.csv) | One row per run (18 runs), including the confusion matrix (TN / FP / FN / TP) |
| [results/gradcam_border_share.csv](results/gradcam_border_share.csv) | How much each model's Grad-CAM attention falls on the image edges instead of the lungs |

## 3. Figures

Folder [results/figures/](results/figures/):

| Figure | Shows |
|---|---|
| `comparison_metrics.png` | All metrics per model (bar = mean of 3 seeds, dots = single seeds) |
| `comparison_roc.png` | ROC curves of all models |
| `comparison_confusion_matrices.png` | Where the models make mistakes |
| `comparison_training_curves.png` | How training went (train and validation loss per epoch) |
| `comparison_efficiency.png` | Quality vs speed vs model size |
| `gradcam_comparison_seed42.jpg` | Where each model "looks" on the same 8 test X-rays |
| `gradcam_border_share.png` | The Grad-CAM shortcut check in one chart |
| `eda_*.png` | The dataset itself: class counts, sample images, image sizes, images per patient |

All of these, with short explanations, are in [notebooks/03_results.ipynb](notebooks/03_results.ipynb). To rebuild the
tables and figures from the result files: `python scripts/compare_results.py --data-dir data/raw/chest_xray`.

## 4. Every training run separately

18 folders `results/<model>_seed<seed>/` (6 models × seeds 42, 43, 44), e.g.
[results/resnet50_seed42/](results/resnet50_seed42/):

| File | What is inside |
|---|---|
| `metrics.json` | Final test metrics, confusion matrix, number of parameters, best epoch, training and inference time, GPU |
| `history.csv` | Training per epoch: learning rate, train / validation loss, validation ROC-AUC, validation accuracy |
| **`test_predictions.csv`** | **The pneumonia detection result for every test X-ray**: image path, true label, the model's probability of pneumonia and its decision at threshold 0.5 |
| `gradcam.jpg` | Heat maps: 8 fixed test images + this model's most confident mistakes |
| `gradcam_fixed.npz` | The same 8 heat maps as numbers |
| `config.yaml` | Exact settings of the run and the library versions |

**Finding a model's mistakes:** in `test_predictions.csv`, the rows where `pred` ≠ `label` are wrong predictions
(`label` / `pred`: 0 = NORMAL, 1 = PNEUMONIA; `prob_pneumonia` = the model's probability of pneumonia).

## 5. Trained models (weights) and full logs — on Kaggle only

Model weights are too large for git, so they stay in the Kaggle notebook outputs:

- Open <https://www.kaggle.com/code/denngerz/02-kaggle-train> → **Output** tab → folder `runs/` with `best.pt` (weights of
  the best epoch) and TensorBoard logs.
- Choose a run in **Versions** (on the right); the full text log of the training is in the **Logs** tab.
- Each version has its own outputs: seed 42 runs are versions "ID-8 … ID-13" (one model each), seeds 43 and 44 are
  versions "ID-14" (baseline, VGG16, ResNet50) and "ID-15" (DenseNet121, EfficientNet-B0, MobileNetV3).

## 6. What is *not* a result

- The local folder `runs/` (created when you train on your own computer) contains only `--debug` test runs on 32 images —
  their numbers mean nothing.
- `results_*.zip` files downloaded from Kaggle are just the transport format; their content is already in `results/`.
