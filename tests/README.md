# tests/ — quick automated checks (pytest)

Run from the repository root: `pytest` (about 1 minute on a laptop CPU). Tests must be fast and must not need the full
dataset or a GPU — they build tiny synthetic images in a temporary folder (`conftest.py`).

| File | Checks |
|---|---|
| `test_splits.py` | Patient IDs parsed from file names; no patient in both train and val; class ratio kept; splits reproducible; committed split files have no leakage |
| `test_data.py` | Transforms give 3×224×224 tensors; the image cache does not change images; val/test order fixed; weighted sampler balances classes reproducibly; debug subset |
| `test_models.py` | Every registered model outputs `(batch, 2)`; baseline size; model configs only set the model name (fair-comparison rule) |
| `test_train.py` | Warm-up + cosine LR schedule; early-stopping rule; tiny end-to-end training run; two identical runs give identical history |
| `test_evaluate.py` | Metrics on a hand-made example; inference timing; Grad-CAM works for every model; `metrics.json` end to end |
| `test_compare.py` | Collecting runs, mean ± std, Markdown table, ROC averaging, all report figures, Grad-CAM border measure |
