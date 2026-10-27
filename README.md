# pneumonia_image_classification
Comparison of Convolutional Neural Network (CNN) architectures for pneumonia classification in X-ray images, evaluating model performance, accuracy, and generalization to identify the most effective approach for medical image classification.

> Status: **done** — 6 models × 3 seeds trained and compared. Results in one page: [results/SUMMARY.md](results/SUMMARY.md);
> full table: [results/comparison.md](results/comparison.md); decisions: [docs/Plan.md](docs/Plan.md).

| Model | Test ROC-AUC (mean ± std, 3 seeds) | ms / image (T4) | Parameters |
|---|---|---|---|
| Baseline CNN (from scratch) | 0.935 ± 0.008 | 1.0 | 0.4 M |
| VGG16 | 0.978 ± 0.008 | 9.7 | 134.3 M |
| ResNet50 | 0.978 ± 0.001 | 8.4 | 23.5 M |
| DenseNet121 | 0.978 ± 0.002 | 24.5 | 7.0 M |
| EfficientNet-B0 | 0.971 ± 0.002 | 13.1 | 4.0 M |
| MobileNetV3-Large | 0.975 ± 0.004 | 10.3 | 4.2 M |

## Overview

- **Data:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kermany et al., 2018 (NORMAL vs PNEUMONIA)
- **Models:** baseline CNN, VGG16, ResNet50, DenseNet121, EfficientNet-B0, MobileNetV3-Large — all trained with one shared pipeline
- **Stack:** Python 3.11, PyTorch, timm, scikit-learn; training on Kaggle Notebooks (GPU)

## Quick start (local, CPU)

```bash
py -3.11 -m venv .venv                       # macOS/Linux: python3.11 -m venv .venv
.venv\Scripts\activate                       # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest                                       # ~1 min, no dataset or GPU needed
```

With the dataset in `data/raw/chest_xray` (see [data/README.md](data/README.md)):

```bash
python scripts/train.py --config configs/models/resnet50.yaml --data-dir data/raw/chest_xray --debug
python scripts/compare_results.py --data-dir data/raw/chest_xray
```

Real training runs on Kaggle's free GPU: [docs/KAGGLE_GUIDE.md](docs/KAGGLE_GUIDE.md).

## Repository layout

| Folder | Contents |
|---|---|
| [`docs/`](docs/) | [Plan.md](docs/Plan.md) — decisions and rationale; [COMMIT_GUIDE.md](docs/COMMIT_GUIDE.md) — how to commit with GitHub Desktop; [KAGGLE_GUIDE.md](docs/KAGGLE_GUIDE.md) — how to train a model on Kaggle; **[PROJECT_EXPLAINED_EN.md](docs/PROJECT_EXPLAINED_EN.md) / [PROJECT_EXPLAINED_UK.md](docs/PROJECT_EXPLAINED_UK.md) — full explanation of the project for the defence (code, maths, hyperparameters, models, results, Q&A)** |
| [`configs/`](configs/) | Experiment settings: shared `base.yaml` + one file per model |
| [`data/`](data/) | `raw/` dataset (not in git) and committed `splits/` |
| [`pneumonia/`](pneumonia/) | Shared Python code: data, models, training, evaluation |
| [`scripts/`](scripts/) | Command-line entry points: make splits, train, compare results |
| [`notebooks/`](notebooks/) | Data exploration, Kaggle training launcher, results visualization |
| [`results/`](results/) | Metrics per run, comparison table, figures |
| [`tests/`](tests/) | Quick automated checks |
| [`report/`](report/) | University report and presentation |

Each folder has a `README.md` with details.

## Contributing

- Humans: read [docs/COMMIT_GUIDE.md](docs/COMMIT_GUIDE.md) — branch → commit → Pull Request, never commit to `main`.
- AI assistants: read [AGENTS.md](AGENTS.md) (Claude Code: [CLAUDE.md](CLAUDE.md)) and the
  [commit-workflow skill](.claude/skills/commit-workflow/SKILL.md).
