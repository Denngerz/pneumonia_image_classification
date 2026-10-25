"""Grad-CAM heat maps: which parts of an X-ray pushed the model towards its prediction.

Run right after training (on Kaggle) because checkpoints are not committed. The same fixed test images are used for
every model so the pictures can be compared side by side; each model's most confident mistakes are added.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402
from torch import nn  # noqa: E402

from pneumonia.data import ChestXrayDataset, load_image  # noqa: E402
from pneumonia.models import gradcam_target_layer  # noqa: E402
from pneumonia.splits import CLASS_NAMES  # noqa: E402

# Fixed on purpose (not the run seed): every model and every seed gets the same test images.
GRADCAM_IMAGE_SEED = 0


def fixed_image_indices(test_df: pd.DataFrame, per_class: int = 4) -> list[int]:
    """Row numbers of the test images shown for every model: `per_class` images of each class."""
    chosen = test_df.groupby("label", sort=True).sample(per_class, random_state=GRADCAM_IMAGE_SEED)
    return sorted(chosen.index.tolist())


def mistake_indices(predictions: pd.DataFrame, count: int = 4) -> list[int]:
    """Row numbers of this model's most confident wrong predictions."""
    wrong = predictions[predictions["pred"] != predictions["label"]]
    confidence = (wrong["prob_pneumonia"] - 0.5).abs()
    return confidence.sort_values(ascending=False).index[:count].tolist()


def compute_cams(model: nn.Module, model_name: str, dataset: ChestXrayDataset, indices: list[int],
                 classes: list[int], device: torch.device) -> np.ndarray:
    """Heat maps in [0, 1] of shape (len(indices), H, W), one per image, for the given target classes."""
    model.eval()
    images = torch.stack([dataset[i][0] for i in indices]).to(device)
    targets = [ClassifierOutputTarget(c) for c in classes]
    with GradCAM(model=model, target_layers=[gradcam_target_layer(model, model_name)]) as cam:
        return cam(input_tensor=images, targets=targets)


def save_gradcam_figure(gray_images: list[np.ndarray], cams: np.ndarray, titles: list[str], title: str,
                        path: Path, columns: int = 4) -> None:
    """Grid of X-rays with the heat map laid over them."""
    rows = int(np.ceil(len(gray_images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, gray, cam, subtitle in zip(axes.ravel(), gray_images, cams, titles):
        rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32) / 255.0
        ax.imshow(show_cam_on_image(rgb, cam, use_rgb=True))
        ax.set_title(subtitle, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    # JPEG: these are photo-like images, ~5× smaller than PNG — they are committed for every run.
    fig.savefig(path, dpi=100, pil_kwargs={"quality": 85})
    plt.close(fig)


def run_gradcam(run: dict, config: dict, test_df: pd.DataFrame) -> None:
    """Write `gradcam.jpg` (fixed images + mistakes) and `gradcam_fixed.npz` (heat maps of the fixed images).

    Needs `test_predictions.csv` from `pneumonia.evaluate.evaluate_run`.
    """
    model_name, device = config["model"]["name"], run["device"]
    dataset = run["loaders"]["test"].dataset
    predictions = pd.read_csv(run["results_dir"] / "test_predictions.csv")
    fixed = fixed_image_indices(test_df, per_class=min(4, int(test_df["label"].value_counts().min())))
    mistakes = [i for i in mistake_indices(predictions) if i not in fixed]
    indices = fixed + mistakes
    predicted = predictions.loc[indices, "pred"].astype(int).tolist()

    cams = compute_cams(run["model"], model_name, dataset, indices, predicted, device)
    size = config["data"]["image_size"]
    gray_images = [np.asarray(load_image(dataset.paths[i], size)) for i in indices]
    titles = []
    for i in indices:
        row = predictions.loc[i]
        mark = "" if row["pred"] == row["label"] else "  ✗ WRONG"
        titles.append(f"true {CLASS_NAMES[row['label']]} → {CLASS_NAMES[row['pred']]}\n"
                      f"p(pneumonia)={row['prob_pneumonia']:.2f}{mark}")
    save_gradcam_figure(gray_images, cams, titles, f"Grad-CAM — {run['run_name']}",
                        run["results_dir"] / "gradcam.jpg")
    np.savez_compressed(
        run["results_dir"] / "gradcam_fixed.npz",
        cams=(cams[:len(fixed)] * 255).round().astype(np.uint8),
        paths=np.array(predictions.loc[fixed, "path"].tolist()),
        labels=predictions.loc[fixed, "label"].to_numpy(),
        preds=predictions.loc[fixed, "pred"].to_numpy(),
        probs=predictions.loc[fixed, "prob_pneumonia"].to_numpy(),
    )
