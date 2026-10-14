# pneumonia_image_classification
Comparison of Convolutional Neural Network (CNN) architectures for pneumonia classification in X-ray images, evaluating model performance, accuracy, and generalization to identify the most effective approach for medical image classification.

> Status: planning done, no code yet. See [docs/Plan.md](docs/Plan.md).

## Overview

- **Data:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kermany et al., 2018 (NORMAL vs PNEUMONIA)
- **Models:** baseline CNN, VGG16, ResNet50, DenseNet121, EfficientNet-B0, MobileNetV3-Large — all trained with one shared pipeline
- **Stack:** Python 3.11, PyTorch, timm, scikit-learn; training on Kaggle Notebooks (GPU)

## Repository layout

| Folder | Contents |
|---|---|
| [`docs/`](docs/) | [Plan.md](docs/Plan.md) — decisions and rationale; [COMMIT_GUIDE.md](docs/COMMIT_GUIDE.md) — how to commit with GitHub Desktop |
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
