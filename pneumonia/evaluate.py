"""One evaluation function for every model: test-set metrics, model size and speed → metrics.json."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import Dataset

from pneumonia.models import count_parameters
from pneumonia.train import predict


def compute_metrics(labels: list[int] | torch.Tensor, probs: list[float] | torch.Tensor,
                    threshold: float = 0.5) -> dict:
    """Classification metrics with PNEUMONIA (label 1) as the positive class.

    recall = sensitivity = share of sick patients found; specificity = share of healthy patients called healthy.
    Ratios with an empty denominator are reported as 0.0.
    """
    labels = torch.as_tensor(labels).long()
    probs = torch.as_tensor(probs).float()
    preds = (probs >= threshold).long()
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(labels.numpy(), preds.numpy(), labels=[0, 1]).ravel())

    def ratio(num: int, den: int) -> float:
        return num / den if den else 0.0

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {
        "accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "specificity": ratio(tn, tn + fp),
        "f1": ratio(2 * precision * recall, precision + recall),
        "roc_auc": float(roc_auc_score(labels.numpy(), probs.numpy())) if labels.unique().numel() == 2 else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


@torch.no_grad()
def measure_inference_time(model: nn.Module, dataset: Dataset, device: torch.device, warmup: int,
                           use_amp: bool = False) -> dict:
    """Milliseconds per image at batch size 1: median over the dataset after `warmup` untimed passes."""
    model.eval()
    times_ms = []
    for i in range(warmup + len(dataset)):
        image = dataset[i % len(dataset)][0].unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if i >= warmup:
            times_ms.append((time.perf_counter() - start) * 1000)
    return {"inference_ms_median": statistics.median(times_ms), "inference_ms_mean": statistics.fmean(times_ms)}


def evaluate_run(run: dict, config: dict, test_df: pd.DataFrame) -> dict:
    """Score the best checkpoint on the test set and write `metrics.json` + `test_predictions.csv` to the run folder.

    `run` is the dictionary returned by `pneumonia.train.fit`; `test_df` is the test split table (same row order as
    the test loader).
    """
    model, device, loader = run["model"], run["device"], run["loaders"]["test"]
    labels, logits = predict(model, loader, device, run["use_amp"])
    probs = torch.softmax(logits, dim=1)[:, 1]
    threshold = config["evaluation"]["threshold"]

    predictions = test_df[["path", "label", "patient_id"]].copy()
    predictions["prob_pneumonia"] = probs.numpy()
    predictions["pred"] = (probs >= threshold).long().numpy()
    predictions.to_csv(run["results_dir"] / "test_predictions.csv", index=False, lineterminator="\n")

    history = pd.read_csv(run["results_dir"] / "history.csv")
    best_row = history.loc[history["epoch"] == run["best_epoch"]].iloc[0]
    metrics = {
        "run": run["run_name"],
        "model": config["model"]["name"],
        "seed": config["seed"],
        "threshold": threshold,
        "test_images": len(labels),
        "test": compute_metrics(labels, probs, threshold),
        "parameters": count_parameters(model),
        "best_epoch": run["best_epoch"],
        "epochs_trained": run["epochs_trained"],
        "best_val_roc_auc": float(best_row["val_roc_auc"]),
        "best_val_loss": float(best_row["val_loss"]),
        "training_seconds": run["training_seconds"],
        **measure_inference_time(model, loader.dataset, device, config["evaluation"]["inference_warmup"],
                                 run["use_amp"]),
        "device": device.type,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    write_json(metrics, run["results_dir"] / "metrics.json")
    return metrics


def write_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
