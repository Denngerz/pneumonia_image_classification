"""Evaluation checks: metrics on a hand-made example, Grad-CAM for every model, end-to-end metrics.json."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from pneumonia.data import ChestXrayDataset, build_transforms, load_split_tables
from pneumonia.evaluate import compute_metrics, evaluate_run, measure_inference_time
from pneumonia.gradcam import compute_cams, fixed_image_indices, mistake_indices, run_gradcam
from pneumonia.models import MODEL_NAMES, create_model
from pneumonia.train import fit
from pneumonia.utils import deep_merge

# `config` and `fake_data` fixtures come from conftest.py


def test_compute_metrics_hand_made_example():
    # 4 sick (1) and 4 healthy (0) patients; threshold 0.5 gives TP=3, FN=1, TN=2, FP=2
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    probs = [0.9, 0.8, 0.6, 0.4, 0.7, 0.55, 0.3, 0.1]
    m = compute_metrics(labels, probs, threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 2, "fp": 2, "fn": 1, "tp": 3}
    assert m["accuracy"] == pytest.approx(5 / 8)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["specificity"] == pytest.approx(2 / 4)
    assert m["precision"] == pytest.approx(3 / 5)
    assert m["f1"] == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))
    # ROC-AUC: share of (sick, healthy) pairs ranked correctly: 0.9 → 4, 0.8 → 4, 0.6 → 3, 0.4 → 2, i.e. 13 / 16
    assert m["roc_auc"] == pytest.approx(13 / 16)


def test_compute_metrics_threshold_and_empty_class():
    m = compute_metrics([1, 1, 1], [0.9, 0.2, 0.6], threshold=0.7)
    assert m["confusion_matrix"] == {"tn": 0, "fp": 0, "fn": 2, "tp": 1}
    assert m["specificity"] == 0.0
    assert m["roc_auc"] is None


def test_inference_time_is_positive():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))
    dataset = [(torch.randn(3, 8, 8), 0) for _ in range(5)]
    timing = measure_inference_time(model, dataset, torch.device("cpu"), warmup=2)
    assert timing["inference_ms_median"] > 0


def test_gradcam_image_selection():
    test_df = pd.DataFrame({"label": [0] * 10 + [1] * 10})
    fixed = fixed_image_indices(test_df, per_class=4)
    assert len(fixed) == 8 and fixed == fixed_image_indices(test_df, per_class=4)
    predictions = pd.DataFrame({"label": [0, 0, 1, 1], "pred": [1, 0, 0, 1], "prob_pneumonia": [0.6, 0.1, 0.05, 0.9]})
    assert mistake_indices(predictions) == [2, 0]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_gradcam_works_for_every_model(name, config, fake_data):
    test_df = load_split_tables(config)["test"]
    dataset = ChestXrayDataset(test_df, fake_data, 224, build_transforms(config, train=False))
    model = create_model(name, pretrained=False)
    cams = compute_cams(model, name, dataset, [0, 5], [1, 0], torch.device("cpu"))
    assert cams.shape == (2, 224, 224)
    assert np.isfinite(cams).all() and cams.min() >= 0 and cams.max() <= 1


def test_metrics_json_end_to_end(config, fake_data, tmp_path):
    config = deep_merge(config, {"model": {"name": "baseline_cnn", "pretrained": False},
                                 "training": {"epochs": 1, "batch_size": 4}})
    run = fit(config, fake_data, tmp_path / "results", tmp_path / "runs")
    test_df = load_split_tables(config)["test"]
    evaluate_run(run, config, test_df)
    run_gradcam(run, config, test_df)
    folder = tmp_path / "results" / "baseline_cnn_seed42"
    metrics = json.loads((folder / "metrics.json").read_text())
    cm = metrics["test"]["confusion_matrix"]
    assert sum(cm.values()) == metrics["test_images"] == len(test_df)
    assert metrics["parameters"] > 0 and metrics["inference_ms_median"] > 0
    assert len(pd.read_csv(folder / "test_predictions.csv")) == len(test_df)
    assert (folder / "gradcam.jpg").exists()
    assert np.load(folder / "gradcam_fixed.npz")["cams"].shape == (6, 224, 224)
