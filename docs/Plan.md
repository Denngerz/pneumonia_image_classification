# Project Plan — Comparison of CNN Models in Pneumonia Image Classification

> Status: **foundation and folder structure agreed**. No code written yet — next: hyperparameters and division of work.

## 1. Goal

Compare several Convolutional Neural Network (CNN) architectures on the task of classifying chest X-ray images as
**NORMAL** or **PNEUMONIA**. We evaluate classification quality, generalization, model size and speed to find the most
effective approach for this medical imaging task.

The core requirement is a **fair comparison**: every model is trained and evaluated with exactly the same data, splits,
preprocessing, augmentation, training settings and metric code. Only the architecture changes.

## 2. Decisions summary

| Topic | Decision |
|---|---|
| Programming language | Python 3.11 |
| Environment | `venv` + `requirements.txt` (works on Windows/macOS/Linux and on Kaggle) |
| Deep learning framework | **PyTorch** + torchvision + **timm** (pretrained ImageNet weights) |
| Dataset | Kaggle **"Chest X-Ray Images (Pneumonia)"** (Kermany et al., 2018), binary NORMAL / PNEUMONIA |
| Training platform | **Kaggle Notebooks** (free GPU, dataset already attached) |
| Local development | CPU, on a small subset of images, for writing and debugging code |
| Code organization | Shared Python code (`.py`) + one config per model; a few notebooks for exploration, launching training on Kaggle, and results |
| Experiment tracking | Each run saves `metrics.json` + `history.csv` + TensorBoard logs; a script builds the comparison table |
| Git workflow | Personal branch → Pull Request → review → merge into `main` |
| Git tool for non-command-line users | **GitHub Desktop** — see [COMMIT_GUIDE.md](COMMIT_GUIDE.md) |
| Documentation language | English |

## 3. Libraries

| Library | Purpose |
|---|---|
| `torch`, `torchvision` | Core deep learning, image transforms, datasets |
| `timm` | One-line access to pretrained CNNs (ResNet, DenseNet, EfficientNet, …) |
| `scikit-learn` | Train/val split, metrics (precision, recall, F1, ROC-AUC, confusion matrix) |
| `numpy`, `pandas` | Data handling, result tables |
| `matplotlib`, `seaborn` | Plots: training curves, confusion matrices, ROC curves |
| `albumentations` (or `torchvision.transforms.v2`) | Data augmentation |
| `pytorch-grad-cam` | Grad-CAM heatmaps — shows which image regions a model focuses on |
| `tensorboard` | Live training curves |
| `PyYAML` | Reading experiment config files |
| `tqdm` | Progress bars |
| `kaggle` | Kaggle API — download the dataset for local work |

Minimum versions are listed in [`requirements.txt`](../requirements.txt). On Kaggle, torch/torchvision are preinstalled and must not be reinstalled.

## 4. Data

**Source:** [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
— Kermany, D. et al. (2018), *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell.

- ~5,863 pediatric chest X-ray images (JPEG, grayscale, varying sizes).
- Two classes: `NORMAL`, `PNEUMONIA` (pneumonia images are bacterial or viral; we treat them as one class).
- Provided folders: `train/`, `val/`, `test/`.

**Known issues and how we handle them**

| Issue | Handling |
|---|---|
| Official `val/` has only **16 images** — useless for model selection | Re-split official `train/` into train / val (≈85 / 15), stratified by class, fixed random seed. Official `test/` stays untouched as the final held-out test set. |
| Several images per patient (pneumonia file names contain `personN_…`) | Split by patient ID where possible, so the same patient never appears in both train and val (avoids data leakage). |
| Class imbalance (~3 pneumonia : 1 normal) | Class-weighted loss (or weighted sampler). Report metrics beyond accuracy. |
| Images differ in size, are grayscale | Resize to **224×224**, convert to 3 channels, normalize with ImageNet statistics (models are ImageNet-pretrained). |

**Data is never committed to git.** Each person downloads it locally or attaches it in Kaggle.

## 5. Models to compare

All pretrained models are loaded from timm/torchvision with ImageNet weights and fine-tuned with the **same** pipeline.

| Model | Why included |
|---|---|
| Custom baseline CNN | Simple from-scratch reference point |
| VGG16 | Classic deep plain CNN |
| ResNet50 | Residual connections; standard baseline |
| DenseNet121 | Dense connections; backbone of CheXNet (well-known chest X-ray model) |
| EfficientNet-B0 | Accuracy/efficiency trade-off via compound scaling |
| MobileNetV3-Large | Lightweight; relevant for deployment on weak hardware |
| *(optional)* ConvNeXt-Tiny | Modern CNN design, if time permits |

## 6. Training protocol (identical for every model)

- Same split files, same seed, same image size (224), same augmentation (light: small rotation, shift/zoom, brightness/contrast; **no** horizontal flip by default, since anatomy is asymmetric — to be confirmed).
- Same loss (class-weighted cross-entropy), optimizer (AdamW), learning-rate schedule, batch size, max epochs, early stopping on validation metric.
- Fixed random seeds; if time allows, repeat each model with **3 seeds** and report mean ± std.
- Code must run on both CPU and GPU without changes (device chosen automatically).

## 7. Evaluation

Measured on the untouched official **test** set:

- Accuracy, Precision, **Recall / Sensitivity** (most important clinically — missing pneumonia is worse), **Specificity**, F1-score, ROC-AUC
- Confusion matrix and ROC curve per model
- Number of parameters, training time, inference time per image
- **Grad-CAM** heatmaps for qualitative comparison (does the model look at the lungs?)

Final output: one comparison table + plots used in the report/presentation.

## 8. Experiment tracking

Each training run writes to its own folder:

- `config.yaml` — exact settings used (copy of the config)
- `history.csv` — loss/metrics per epoch
- `metrics.json` — final test metrics
- TensorBoard logs

A script collects all `metrics.json` files into a single comparison table (CSV + Markdown). Model checkpoints (`*.pt`) are **not** committed to git — they are kept locally or as Kaggle notebook outputs.

## 9. Why these choices (plain-language rationale)

### Why do we need a GPU (Kaggle) at all?
Training a CNN means billions of multiplications per epoch. A GPU does them in parallel and is about 20–50× faster than a
laptop CPU. One fine-tuning run of ResNet50 here takes roughly **5–10 minutes on a GPU** vs **3–6 hours on a CPU**. With
6 models and repeated runs, CPU-only would take days.

| Option | Benefits | Drawbacks |
|---|---|---|
| **Kaggle Notebooks** *(chosen)* | Free GPU (~30 h/week), dataset already hosted — no download, stable sessions up to 12 h | Weekly quota; git integration is clumsy; phone-verified account needed |
| Google Colab | Free GPU, popular, easy `git clone` | Disconnects when idle; GPU not always available; dataset must be downloaded each session |
| Local NVIDIA GPU | No limits, easy debugging | Requires hardware + CUDA setup |
| CPU only | Works everywhere | Only suitable for debugging on a few images |

Workflow: write and debug code locally on a small subset → run full training on Kaggle.

### What is a notebook, and why not use only notebooks?
A **Jupyter notebook** (`.ipynb`) is a document made of **cells**. Each cell contains either Python code or text. You run
cells one by one and the output (numbers, tables, charts, images) appears right under the cell — like a lab journal.
A **script** (`.py`) is plain code that runs top to bottom with no built-in place for charts or notes.

We use notebooks because:
1. **Kaggle runs notebooks** — a notebook is the "start button" for training on the free GPU.
2. They are ideal for **looking at data and results** (sample X-rays, charts, confusion matrices, Grad-CAM) — good material for the report.

We keep the training logic in shared `.py` files because the project is a **comparison**: if every model had its own
notebook with copied code, small differences (image size, forgotten class weights, …) would sneak in and make the
comparison unfair. With shared code, a bug fix applies to all models at once and only the model name in the config differs.

> In short: **`.py` files are the recipe; notebooks are the kitchen where you press "cook" and look at the dish.**

## 10. Project structure

```
pneumonia_image_classification/
├── README.md, AGENTS.md, CLAUDE.md
├── requirements.txt
├── .claude/skills/commit-workflow/   AI commit/PR procedure
├── docs/                 Plan.md (this file), COMMIT_GUIDE.md
├── configs/              base.yaml (shared settings) + models/<model>.yaml
├── data/
│   ├── raw/              downloaded dataset — git-ignored
│   └── splits/           train/val/test CSVs — committed
├── pneumonia/            shared Python code ("the recipe")
├── scripts/              make_splits.py, train.py, compare_results.py
├── notebooks/            01_eda, 02_kaggle_train, 03_results ("the kitchen")
├── results/              per-run metrics/history/config + comparison table + figures — committed
├── runs/                 checkpoints, TensorBoard logs — git-ignored
├── tests/                quick pytest checks
└── report/               university report and presentation drafts
```

Every folder has its own `README.md` describing what goes there and the planned file names.

| Decision | Choice | Why |
|---|---|---|
| Shared code location | Top-level `pneumonia/` package | Imports work from the repo root with no install step — simplest for beginners and Kaggle |
| How code gets to Kaggle | Public GitHub repo, `git clone` in the Kaggle notebook | One line, always the latest committed code; no tokens needed. Only code/docs are public — never data or secrets |
| Experiment differences | Only in `configs/models/*.yaml` | Visible proof that every model got identical treatment |
| Split files | Committed in `data/splits/` | Tiny files that guarantee everyone uses exactly the same images |
| Results in git | Per-run `config.yaml`, `history.csv`, `metrics.json`, comparison table, PNG figures | Small and needed by everyone for the report |
| Not in git | Dataset, checkpoints (`*.pt`), TensorBoard logs, `kaggle.json` | Large, re-creatable, or secret — enforced by `.gitignore` |
| Guides | In `docs/`; `AGENTS.md` and `CLAUDE.md` stay in the root | AI tools look for those two files in the root |

**Kaggle training flow:** open `notebooks/02_kaggle_train.ipynb` on Kaggle → attach the dataset (*Add Input*) → enable GPU
and Internet → the notebook clones the repo, installs missing requirements and runs `scripts/train.py` → download the
`results/<run>/` folder from the notebook output and commit it via a Pull Request.

## 11. Collaboration

- Everyone works on their own branch and opens a Pull Request into `main`; at least one teammate reviews before merging.
- Commit messages follow the format in [COMMIT_GUIDE.md](COMMIT_GUIDE.md).
- AI assistants follow [AGENTS.md](../AGENTS.md) (and [CLAUDE.md](../CLAUDE.md) for Claude Code) and the commit skill at
  [.claude/skills/commit-workflow/SKILL.md](../.claude/skills/commit-workflow/SKILL.md).

## 12. Open items (next discussions)

- [x] Project folder structure (section 10)
- [ ] Make the GitHub repository public (repo owner: Settings → General → Danger Zone → Change visibility)
- [ ] Exact hyperparameters (learning rate, batch size, epochs, augmentation details)
- [ ] Optional: external test set to measure generalization (e.g. RSNA Pneumonia Detection Challenge subset)
- [ ] Optional: 3-class variant (NORMAL / BACTERIAL / VIRAL)
- [ ] Division of work between team members
