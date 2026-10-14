# notebooks/ — the "kitchen"

Notebooks are for **looking** at data and results and for **launching** training on Kaggle.
They import from `pneumonia/` and must not contain their own copies of training or metric code.

| Planned notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Explore the dataset: class counts, sample images, image sizes, patients per class |
| `02_kaggle_train.ipynb` | Kaggle launcher: `git clone` this repo, install requirements, run `scripts/train.py` for the chosen configs, save outputs |
| `03_results.ipynb` | Comparison table, training curves, confusion matrices, ROC curves, Grad-CAM — figures for the report |

Naming: `NN_short_name.ipynb`. Clear outputs of training notebooks before committing; results notebooks may keep their charts.
