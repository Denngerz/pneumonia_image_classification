"""Training loop checks: LR schedule, early-stopping rule, and a tiny end-to-end run on synthetic images."""

import pandas as pd
import pytest
import torch
import yaml

from pneumonia.train import build_scheduler, fit, is_improvement, validation_scores
from pneumonia.utils import deep_merge

# `config` and `fake_data` fixtures come from conftest.py


def test_scheduler_warms_up_then_decays_to_zero(config):
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=10, epochs=5)
    lrs = []
    for _ in range(50):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(0.1)          # warm-up starts low
    assert max(lrs) == pytest.approx(1.0)         # reaches the base LR after 1 epoch
    assert lrs[10:] == sorted(lrs[10:], reverse=True)
    assert lrs[-1] < 0.01


def test_improvement_rule():
    best = {"val_roc_auc": 0.90, "val_loss": 0.30}
    assert is_improvement({"val_roc_auc": 0.91, "val_loss": 0.50}, best)
    assert is_improvement({"val_roc_auc": 0.90, "val_loss": 0.20}, best)
    assert not is_improvement({"val_roc_auc": 0.90, "val_loss": 0.40}, best)
    assert not is_improvement({"val_roc_auc": 0.89, "val_loss": 0.10}, best)
    assert is_improvement({"val_roc_auc": 0.5, "val_loss": 9.0}, None)


def test_validation_scores_perfect_predictions():
    labels = torch.tensor([0, 1, 1, 0])
    logits = torch.tensor([[3.0, -3.0], [-3.0, 3.0], [-2.0, 2.0], [2.0, -2.0]])
    scores = validation_scores(labels, logits)
    assert scores["val_roc_auc"] == 1.0
    assert scores["val_accuracy"] == 1.0


def test_fit_runs_end_to_end_and_writes_outputs(config, fake_data, tmp_path):
    config = deep_merge(config, {
        "model": {"name": "baseline_cnn", "pretrained": False},
        "training": {"epochs": 3, "batch_size": 4, "early_stopping_patience": 1},
    })
    run = fit(config, fake_data, tmp_path / "results", tmp_path / "runs")
    results, runs = tmp_path / "results" / "baseline_cnn_seed42", tmp_path / "runs" / "baseline_cnn_seed42"
    history = pd.read_csv(results / "history.csv")
    assert 1 <= len(history) <= 3
    assert {"train_loss", "val_loss", "val_roc_auc", "best"} <= set(history.columns)
    assert history.loc[history["best"], "epoch"].iloc[-1] == run["best_epoch"]
    saved = yaml.safe_load((results / "config.yaml").read_text())
    assert saved["seed"] == 42 and saved["model"]["name"] == "baseline_cnn"
    assert (runs / "best.pt").exists() and (runs / "last.pt").exists()


def test_fit_is_reproducible(config, fake_data, tmp_path):
    config = deep_merge(config, {"model": {"name": "baseline_cnn", "pretrained": False},
                                 "training": {"epochs": 1, "batch_size": 4}})
    fit(config, fake_data, tmp_path / "a", tmp_path / "runs_a")
    fit(config, fake_data, tmp_path / "b", tmp_path / "runs_b")
    columns = ["train_loss", "val_loss", "val_roc_auc"]
    first = pd.read_csv(tmp_path / "a" / "baseline_cnn_seed42" / "history.csv")[columns]
    second = pd.read_csv(tmp_path / "b" / "baseline_cnn_seed42" / "history.csv")[columns]
    pd.testing.assert_frame_equal(first, second)
