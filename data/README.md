# data/

| Folder | In git? | Contents |
|---|---|---|
| `raw/chest_xray/` | ❌ ignored | The Kaggle dataset as downloaded (`train/`, `val/`, `test/` with `NORMAL/` and `PNEUMONIA/` subfolders) |
| `splits/` | ✅ committed | `train.csv`, `val.csv`, `test.csv` — one row per image |

The split files are tiny and are committed so that **every teammate and every Kaggle run uses exactly the same images**.
They are generated once by `scripts/make_splits.py` (patient-level, stratified, seed 42) and only changed by team agreement.

| Split | Source | Images | Patients | NORMAL | PNEUMONIA |
|---|---|---|---|---|---|
| `train.csv` | official `train/` + `val/`, 85 % | 4447 | 2441 | 1147 | 3300 |
| `val.csv` | official `train/` + `val/`, 15 % | 785 | 420 | 202 | 583 |
| `test.csv` | official `test/`, unchanged | 624 | 427 | 234 | 390 |

CSV columns: `path` (relative to `chest_xray/`, e.g. `train/NORMAL/IM-0115-0001.jpeg`), `label` (0 = NORMAL,
1 = PNEUMONIA), `class_name`, `patient_id` (`PNEUMONIA/person1000`, `NORMAL/IM-0115`, `NORMAL/NORMAL2-IM-0427`),
`source_split` (official folder the image came from).

No patient is in both train and val. 170 PNEUMONIA patient numbers also occur in the official test folder; the
dataset does not say whether these are the same people, so the test set is left as published (see `notebooks/01_eda.ipynb`).

## Getting the dataset

- **On Kaggle:** in the notebook, *Add Input* → search "Chest X-Ray Images (Pneumonia)" (paultimothymooney) → it appears under
  `/kaggle/input/chest-xray-pneumonia/chest_xray`. Pass that path to the scripts; nothing is copied into this folder.
- **Locally:** download from <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia> (or with the Kaggle API:
  `kaggle datasets download -d paultimothymooney/chest-xray-pneumonia`) and unzip so that `data/raw/chest_xray/train` exists.
  Keep `kaggle.json` outside the repo. The zip also contains a duplicate `chest_xray/chest_xray/` folder and a
  `__MACOSX/` folder — skip them, only `chest_xray/{train,val,test}` are used.
