# Project Plan — Comparison of CNN Models in Pneumonia Image Classification

> Status: **all tasks ID-1 … ID-18 done** — 18 training runs (6 models × 3 seeds) compared; summary in
> [results/SUMMARY.md](../results/SUMMARY.md). Remaining items are optional extensions (section 12).

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
| `torchvision.transforms.v2` | Data augmentation (part of torchvision — no extra dependency; `albumentations` was dropped to avoid OpenCV version conflicts on Kaggle) |
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
| Official `val/` has only **16 images** — useless for model selection | Pool official `train/` + `val/` and re-split into train / val (85 / 15), stratified by class, seed 42. Official `test/` stays untouched as the final held-out test set. |
| Several images per patient | Split by patient ID so the same patient never appears in both train and val (avoids data leakage). Patient ID: `personN` for PNEUMONIA files; `IM-XXXX` for NORMAL files (`NORMAL2-IM-XXXX` is a separate group). Train↔test patient overlap is only reported in the EDA notebook — the official test set is not changed. |
| Class imbalance (~3 pneumonia : 1 normal) | **Weighted random sampler** (inverse class frequency, with replacement, one epoch = number of training images) with plain cross-entropy loss. Report metrics beyond accuracy. |
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

Agreed hyperparameters (stored in `configs/base.yaml`; **no per-model overrides**, the baseline CNN included):

| Setting | Value |
|---|---|
| Image size | 224×224, grayscale → 3 channels, ImageNet normalization |
| Augmentation (train only) | rotation ±10°, shift / scale ±10%, brightness / contrast ±0.1; **no** horizontal flip (anatomy is asymmetric) |
| Batch size | 32 |
| Optimizer | AdamW, learning rate 1e-4, weight decay 1e-4 |
| LR schedule | cosine decay with 1 warm-up epoch |
| Epochs | max 25, early stopping on **validation ROC-AUC** with patience 5 (ties broken by validation loss); the best epoch is kept |
| Fine-tuning | all layers trained (full fine-tune) |
| Class imbalance | weighted random sampler + plain cross-entropy |
| Precision | mixed precision (AMP) on GPU |
| Decision threshold | 0.5 on the PNEUMONIA probability |
| Seeds | 42 for the first run of every model; then 43 and 44 → report mean ± std over 3 seeds |
| Determinism | Python / NumPy / PyTorch seeded, `cudnn.deterministic = True`, `cudnn.benchmark = False` |
| Hardware | every Kaggle run on **one T4 GPU**, so training and inference times are comparable |

Models: baseline CNN = 4 × [Conv 3×3 → BatchNorm → ReLU → MaxPool] with 32 → 64 → 128 → 256 channels, global average
pooling, dropout 0.3, linear layer → 2 classes. Pretrained models come from timm with their default ImageNet weights
(`vgg16`, `resnet50`, `densenet121`, `efficientnet_b0`, `mobilenetv3_large_100`) and a new 2-class head.

Code must run on both CPU and GPU without changes (device chosen automatically).

## 7. Evaluation

Measured on the untouched official **test** set:

- Accuracy, Precision, **Recall / Sensitivity** (most important clinically — missing pneumonia is worse), **Specificity**, F1-score, ROC-AUC
- Confusion matrix and ROC curve per model
- Number of parameters, training time, inference time per image (GPU, batch size 1, median over the test set after 10 warm-up passes)
- **Grad-CAM** heatmaps for qualitative comparison (does the model look at the lungs?). Computed on Kaggle right after
  training from the best checkpoint, on the same fixed set of test images for every model (including misclassified ones);
  only the PNG figures are committed.

Final output: one comparison table + plots used in the report/presentation (`scripts/compare_results.py` →
`results/comparison.md` and `results/figures/comparison_*.png`).

**Primary metric (team decision after the 18 runs):** models are compared by **test ROC-AUC, mean ± std over seeds
42–44**. ROC-AUC does not depend on the decision threshold and was stable across seeds (std ≤ 0.008), while metrics at
the fixed 0.5 threshold varied strongly (e.g. baseline specificity 0.11–0.77). Threshold-0.5 metrics are still reported,
with that caveat. Choosing a per-model threshold on the validation set is noted in the report as future work; the
threshold is never tuned on the test set.

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
- [x] Make the GitHub repository public
- [x] Exact hyperparameters (section 6)
- [ ] Optional: external test set to measure generalization (e.g. RSNA Pneumonia Detection Challenge subset)
- [ ] Optional: 3-class variant (NORMAL / BACTERIAL / VIRAL)
- [ ] Optional: per-model decision threshold chosen on the validation set (section 7)
- [x] Division of work between team members — task spreadsheet ID-1 … ID-18
