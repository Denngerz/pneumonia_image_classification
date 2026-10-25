# How to Train a Model on Kaggle

This guide shows how to train one of our models on Kaggle's free GPU and bring the results back into the project.
No installation on your computer is needed — everything happens in the browser.

> **Why Kaggle?** Training on a laptop CPU would take hours per model; Kaggle gives a free NVIDIA T4 GPU
> (30 hours per week) and already hosts our dataset. See [Plan.md](Plan.md) section 9.

---

## 0. One-time setup

1. Create an account at <https://www.kaggle.com> and **verify your phone number**
   (*Settings → Phone verification*). Without it, GPU and Internet are not available in notebooks.
2. That's all — the project code is downloaded from GitHub automatically by the notebook.

## 1. Create the notebook (once)

1. Go to <https://www.kaggle.com/code> → **+ New Notebook**.
2. **File → Import Notebook → Link** and paste:

   ```
   https://raw.githubusercontent.com/Denngerz/pneumonia_image_classification/main/notebooks/02_kaggle_train.ipynb
   ```

   → **Import**.
3. Click the notebook title (top left) and rename it to `02_kaggle_train`.
4. Right panel → **Input → + Add Input** → paste
   `https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia` in the search box → press the **+** next to
   *Chest X-Ray Images (Pneumonia)*. It appears under *Input → Datasets*.
5. Right panel → **Session options**:
   - **Accelerator → GPU T4 x2** → *Turn on GPU T4 x2*. (Always T4: our training/inference times must be comparable.
     The code uses one of the two GPUs.)
   - **Internet → on** (needed to download the code and the pretrained weights).

The notebook keeps these settings, so next time skip straight to step 2.

## 2. Choose what to train

Only edit the **first code cell** (the "Settings" cell):

```python
MODELS = ["resnet50"]   # one or more of: baseline_cnn, vgg16, resnet50, densenet121, efficientnet_b0, mobilenetv3_large
SEEDS = [42]            # 42 for the first run; 43 and 44 for the repeats
DEBUG = False           # True = 1-minute check on 32 images; False = the real training
BRANCH = "main"
REPO_URL = "https://github.com/Denngerz/pneumonia_image_classification.git"
```

Do **not** change anything else: all training settings come from `configs/base.yaml`, identical for every model.

## 3. Run it in the background

1. Top right **Save Version** → Version name e.g. `resnet50 seed42` → Version type **Save & Run All (Commit)** → **Save**.
2. The run continues on Kaggle even if you close the browser. Follow it via the bell icon / *View active events*,
   or open the notebook page → **Versions** / **Logs**.
3. Durations on one T4 (25 epochs at most, usually less because of early stopping):

   | Run | Total time |
   |---|---|
   | `DEBUG = True`, all 6 models | ~3.5 min (checked in task ID-7) |
   | `baseline_cnn` (25 epochs, ~9 s each) | ~6 min |
   | `vgg16` (~43 s per epoch) | ~14 min |
   | `resnet50` (~19 s per epoch) | ~6 min |
   | `densenet121` (~23 s per epoch) | ~8 min |
   | `efficientnet_b0` (~14 s per epoch, first epoch ~40 s) | ~9 min |
   | `mobilenetv3_large` (~10 s per epoch, first epoch ~33 s) | ~7 min |

   Measured on seed 42 (tasks ID-8 … ID-13); early stopping makes the number of epochs differ between runs.
   All 6 models in one version take about 50 minutes of the 30-hour weekly GPU quota.

   About 1.5 minutes of every run is setup: cloning, installing and loading all images into memory once.

## 4. Download the results

1. When the version shows **Successful**, open the notebook page → **Output** tab.
2. Download `results_<model>_seed<seed>.zip` (a few hundred KB). Ignore the `runs/` folder (checkpoints, ~100 MB–1 GB,
   never committed).
3. Unzip it into the project's `results/` folder, so that you get e.g. `results/resnet50_seed42/metrics.json`.
4. Commit it on a new branch (`exp/resnet50-seed42`) with the message `Exp: add resnet50 seed42 results` and open a
   Pull Request — see [COMMIT_GUIDE.md](COMMIT_GUIDE.md). In the PR description write the main test numbers from
   `metrics.json` and anything strange you noticed in `history.csv` (e.g. validation loss going up while training loss
   goes down = overfitting).

## 5. If something goes wrong

| Problem | Fix |
|---|---|
| `Dataset not found` | Step 1.4 — attach *Chest X-Ray Images (Pneumonia)* |
| `git clone` / `pip install` fails, *"Temporary failure in name resolution"* | Internet is off (step 1.5) or the phone is not verified |
| `cuda False` in the first output lines | Accelerator is *None* — choose GPU T4 x2 |
| `CUDA out of memory` | Make sure nothing else runs in the same notebook; tell the team (do **not** change the batch size for one model only) |
| Run stopped after 12 hours | Kaggle's limit per run — train fewer models per version |
| Weekly GPU quota used up | Wait for the reset (shown under *Session options → Accelerator*) or ask a teammate to run it |
