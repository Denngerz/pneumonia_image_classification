# notebooks/ — the "kitchen"

Notebooks are for **looking** at data and results and for **launching** training on Kaggle.
They import from `pneumonia/` and must not contain their own copies of training or metric code.

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Explore the dataset: class counts, sample images, image sizes, patients per class |
| `02_kaggle_train.ipynb` | Kaggle launcher: `git clone` this repo, install the missing libraries, run `scripts/train.py` for the chosen models and seeds, zip the results. How to use it: [docs/KAGGLE_GUIDE.md](../docs/KAGGLE_GUIDE.md) |
| `03_results.ipynb` | Comparison table, metric bars, ROC curves, training curves, confusion matrices, quality vs speed, cross-model Grad-CAM — with short explanations; figures for the report |

Naming: `NN_short_name.ipynb`. Clear outputs of training notebooks before committing; results notebooks may keep their charts.
