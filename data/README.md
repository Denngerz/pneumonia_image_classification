# data/

| Folder | In git? | Contents |
|---|---|---|
| `raw/chest_xray/` | ❌ ignored | The Kaggle dataset as downloaded (`train/`, `val/`, `test/` with `NORMAL/` and `PNEUMONIA/` subfolders) |
| `splits/` | ✅ committed | `train.csv`, `val.csv`, `test.csv` — list of image paths (relative to `chest_xray/`), label and patient ID |

The split files are tiny and are committed so that **every teammate and every Kaggle run uses exactly the same images**.
They are generated once by `scripts/make_splits.py` (patient-level, stratified, fixed seed) and only changed by team agreement.

## Getting the dataset

- **On Kaggle:** in the notebook, *Add Input* → search "Chest X-Ray Images (Pneumonia)" (paultimothymooney) → it appears under
  `/kaggle/input/chest-xray-pneumonia/chest_xray`. Pass that path to the scripts; nothing is copied into this folder.
- **Locally:** download from <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia> (or with the Kaggle API:
  `kaggle datasets download -d paultimothymooney/chest-xray-pneumonia`) and unzip so that `data/raw/chest_xray/train` exists.
  Keep `kaggle.json` outside the repo.
