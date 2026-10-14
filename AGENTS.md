# AGENTS.md — Instructions for AI coding assistants

This file is the single source of instructions for any AI assistant working in this repository
(Claude Code, OpenAI Codex, Cursor, GitHub Copilot, Gemini, etc.). Read it fully before making changes.

## Project

**Comparison of Convolutional Neural Network models in Pneumonia Image Classification** — a university team project (VGTU).
We train several CNN architectures on chest X-ray images (NORMAL vs PNEUMONIA) and compare them fairly.

Full background, decisions and rationale: [Plan.md](docs/Plan.md). Treat `docs/Plan.md` as the source of truth for scope;
if a request contradicts it, point that out to the user before proceeding.

## Required reading before specific tasks

| Task | Read first |
|---|---|
| Any commit, push, branch or Pull Request | **[.claude/skills/commit-workflow/SKILL.md](.claude/skills/commit-workflow/SKILL.md)** — follow it step by step |
| Explaining git to a teammate | [COMMIT_GUIDE.md](docs/COMMIT_GUIDE.md) (GitHub Desktop guide for humans) |
| Anything about data, models, training or evaluation | [Plan.md](docs/Plan.md) sections 4–8 |

## Tech stack

- Python 3.11, `venv` + `requirements.txt`
- PyTorch + torchvision + timm (ImageNet-pretrained models)
- scikit-learn, numpy, pandas, matplotlib, seaborn, albumentations, pytorch-grad-cam, tensorboard, PyYAML, tqdm
- Training runs on **Kaggle Notebooks** (GPU); local development on CPU with a small data subset

Do not introduce other deep learning frameworks (TensorFlow/Keras, JAX) or experiment-tracking services that need
accounts (W&B, Comet) without the user's explicit agreement.

## Project structure

```
pneumonia_image_classification/
├── README.md, AGENTS.md, CLAUDE.md   # root files (tools expect AGENTS.md / CLAUDE.md here)
├── requirements.txt
├── .claude/skills/commit-workflow/   # AI commit/PR procedure
├── docs/                 # Plan.md (decisions), COMMIT_GUIDE.md (git for humans)
├── configs/              # base.yaml + models/<model>.yaml — the ONLY place experiments differ
├── data/
│   ├── raw/              # downloaded dataset (git-ignored)
│   └── splits/           # train/val/test CSVs — committed, shared by everyone
├── pneumonia/            # shared Python package: data, splits, models, train, evaluate, gradcam, utils
├── scripts/              # CLI entry points: make_splits.py, train.py, compare_results.py
├── notebooks/            # 01_eda, 02_kaggle_train, 03_results — import from pneumonia/, no copied logic
├── results/              # per-run config.yaml, history.csv, metrics.json + comparison table + figures/ (committed)
├── runs/                 # checkpoints, TensorBoard logs (git-ignored, created at runtime)
├── tests/                # fast pytest checks, no dataset or GPU needed
└── report/               # university report and presentation drafts
```

Each folder has a `README.md` listing what belongs there and the planned file names — read it before adding files,
and keep it up to date when you add or rename something. Do not create new top-level folders without the user's agreement.

Placement rules:
- Reusable logic → `pneumonia/` (top-level package, imported from the repo root; no install step).
- Something a person runs → `scripts/` (thin wrapper calling `pneumonia/`), with `--data-dir` and, for training, `--debug`.
- Notebooks only explore data, launch training on Kaggle, or visualize results. They import from `pneumonia/`.
- A new model = a registry entry in `pneumonia/models.py` + a config in `configs/models/`. Nothing else changes.
- Run outputs: small files → `results/<model>_seed<seed>/`; large files → `runs/`.

The code is cloned into Kaggle notebooks with `git clone` from the public GitHub repo, so everything committed is public.

## Fair-comparison rules (must not be broken)

The whole value of this project is a fair comparison. Any change must keep these invariants:

1. **Same data for every model**: the same train/val split files, generated once with a fixed seed. Official `test/` is
   used only for final evaluation — never for training, tuning or early stopping.
2. **Same pipeline for every model**: image size 224×224, preprocessing, augmentation, loss, optimizer, LR schedule,
   batch size, epoch budget and early-stopping rule come from shared code/config — never hard-coded per model.
3. Differences between experiments are expressed **only in config files** (e.g. model name), not by editing training code
   for one model.
4. **Seeds are fixed** (Python, NumPy, PyTorch) and recorded in each run's saved config.
5. **Class imbalance** is handled the same way for all models (class-weighted loss or weighted sampler).
6. Metrics are computed by one shared evaluation function: accuracy, precision, recall/sensitivity, specificity, F1,
   ROC-AUC, confusion matrix, parameter count, training and inference time.
7. Avoid data leakage: split by patient ID where the file name allows (`personN_…`).

If a requested change would break one of these rules, stop and explain the conflict to the user.

## Coding conventions

- PEP 8, 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Type hints on public functions; short docstrings explaining *what* and *why*.
- Small, single-purpose functions. No copy-pasted training loops.
- **No hard-coded absolute paths** (`C:\Users\...`, `/kaggle/input/...`). Paths come from config or CLI arguments so the
  same code runs locally and on Kaggle.
- **Device-agnostic code**: pick `cuda` if available, otherwise `cpu`. Never assume a GPU exists.
- Provide a quick "debug" mode (few images, 1 epoch) so code can be checked on CPU in minutes.
- Keep notebooks short; clear heavy outputs before committing training notebooks.
- Add new dependencies to `requirements.txt` and mention them in the PR description.

## Never commit

- The dataset or any image files from it (`chest_xray/`, `*.jpeg` data folders)
- Model checkpoints / weights (`*.pt`, `*.pth`, `*.ckpt`, `*.onnx`)
- Large generated outputs (TensorBoard event files, big CSV dumps) unless the user explicitly asks
- Secrets: `kaggle.json`, API keys, tokens, `.env` files
- Virtual environments (`venv/`, `.venv/`) and caches (`__pycache__/`, `.ipynb_checkpoints/`)

## Git rules (summary — details in the skill)

- **Never commit directly to `main`.** Create a branch `<type>/<short-topic>` (e.g. `feat/resnet50-config`, `docs/plan`).
- Commit messages: `Type: very short description` — e.g. `Feat: add ResNet50 config`, `Bugfix: fix split seed`.
  Types: `Feat`, `Bugfix`, `Refactor`, `Docs`, `Exp`, `Data`, `Test`, `Chore`. No scope, no trailing period.
- Open a Pull Request into `main`; describe what, why, how it was tested, and results if any.
- Never force-push, rewrite shared history, or skip hooks.
- Only commit/push/open PRs when the user asks you to.

Full procedure: [.claude/skills/commit-workflow/SKILL.md](.claude/skills/commit-workflow/SKILL.md).

## Communication

- Team members have mixed experience. Explain ML and git concepts in plain language when asked.
- Report results honestly: if a test fails or training was not actually run, say so.
