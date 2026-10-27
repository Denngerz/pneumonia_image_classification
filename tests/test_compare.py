"""Comparison checks on a few fake runs: aggregation, table format, and that every figure is written."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from PIL import Image

from pneumonia.compare import (border_share, build_gradcam_report, build_report, collect_runs, mean_roc_curve,
                               summarize, summary_markdown)


def _fake_run(results_dir: Path, model: str, seed: int, auc: float, rng: np.random.Generator) -> None:
    folder = results_dir / f"{model}_seed{seed}"
    folder.mkdir(parents=True)
    test = {"accuracy": 0.9, "precision": 0.9, "recall": 0.95, "specificity": 0.8, "f1": 0.92, "roc_auc": auc,
            "confusion_matrix": {"tn": 8, "fp": 2, "fn": 1, "tp": 19}}
    metrics = {"run": folder.name, "model": model, "seed": seed, "test": test, "parameters": 1000,
               "best_epoch": 3, "epochs_trained": 5, "training_seconds": 120.0, "inference_ms_median": 2.0,
               "gpu": None}
    (folder / "metrics.json").write_text(json.dumps(metrics))
    labels = np.array([0] * 10 + [1] * 20)
    probs = np.clip(labels * 0.6 + rng.random(30) * 0.5, 0, 1)
    pd.DataFrame({"path": [f"x{i}" for i in range(30)], "label": labels, "patient_id": "p",
                  "prob_pneumonia": probs, "pred": (probs >= 0.5).astype(int)}).to_csv(
        folder / "test_predictions.csv", index=False)
    pd.DataFrame({"epoch": [1, 2, 3], "train_loss": [0.6, 0.4, 0.3], "val_loss": [0.5, 0.4, 0.45],
                  "val_roc_auc": [0.9, 0.95, 0.94]}).to_csv(folder / "history.csv", index=False)
    np.savez_compressed(folder / "gradcam_fixed.npz", cams=rng.integers(0, 255, (2, 32, 32), dtype=np.uint8),
                        paths=np.array(["test/NORMAL/a.jpeg", "test/PNEUMONIA/b.jpeg"]), labels=np.array([0, 1]),
                        preds=np.array([1, 1]), probs=np.array([0.7, 0.9]))


@pytest.fixture()
def fake_results(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    for model, aucs in (("resnet50", (0.97, 0.98, 0.99)), ("baseline_cnn", (0.90, 0.92, 0.94))):
        for seed, auc in zip((42, 43, 44), aucs):
            _fake_run(tmp_path, model, seed, auc, rng)
    (tmp_path / "figures").mkdir()
    return tmp_path


def test_collect_runs_in_registry_order(fake_results):
    runs = collect_runs(fake_results)
    assert len(runs) == 6
    assert list(runs["model"].unique()) == ["baseline_cnn", "resnet50"]  # registry order, not alphabetical
    assert list(runs["seed"][:3]) == [42, 43, 44]


def test_summary_mean_and_std(fake_results):
    summary = summarize(collect_runs(fake_results)).set_index("model")
    assert summary.loc["resnet50", "roc_auc_mean"] == pytest.approx(0.98)
    assert summary.loc["resnet50", "roc_auc_std"] == pytest.approx(0.01)  # sample std of 0.97, 0.98, 0.99
    assert summary.loc["baseline_cnn", "seeds"] == 3


def test_markdown_table(fake_results):
    text = summary_markdown(summarize(collect_runs(fake_results)))
    assert "| ResNet50 | 0.0 | 3 | 0.980 ± 0.010 |" in text
    assert text.count("\n") == 4  # header, separator, 2 models


def test_mean_roc_curve_is_monotonic():
    df = pd.DataFrame({"label": [0, 0, 1, 1], "prob_pneumonia": [0.1, 0.6, 0.4, 0.9]})
    grid = np.linspace(0, 1, 11)
    curve = mean_roc_curve([df, df], grid)
    assert curve[0] == 0 and curve[-1] == 1
    assert np.all(np.diff(curve) >= 0)


def test_border_share():
    uniform = np.ones((1, 100, 100))
    centre = np.zeros((1, 100, 100))
    centre[0, 40:60, 40:60] = 1
    frame = np.ones((1, 100, 100))
    frame[0, 10:90, 10:90] = 0
    shares = border_share(np.concatenate([uniform, centre, frame, np.zeros((1, 100, 100))]), border=0.1)
    assert shares[0] == pytest.approx(0.36)
    assert shares[1] == pytest.approx(0.0)
    assert shares[2] == pytest.approx(1.0)
    assert np.isnan(shares[3])  # an empty heat map has no "heat" to place


def test_gradcam_report(fake_results, tmp_path):
    data_dir = tmp_path / "chest_xray"
    for rel in ("test/NORMAL/a.jpeg", "test/PNEUMONIA/b.jpeg"):
        (data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (64, 48), color=120).save(data_dir / rel)
    border = build_gradcam_report(fake_results, fake_results / "figures", data_dir, seed=42)
    assert list(border["model"]) == ["baseline_cnn", "resnet50"]
    assert border["border_share_mean"].between(0, 1).all()
    assert (fake_results / "gradcam_border_share.csv").exists()
    assert (fake_results / "figures" / "gradcam_border_share.png").exists()
    assert (fake_results / "figures" / "gradcam_comparison_seed42.jpg").exists()


def test_build_report_writes_tables_and_figures(fake_results):
    build_report(fake_results, fake_results / "figures")
    for name in ("comparison.md", "comparison.csv", "comparison_runs.csv"):
        assert (fake_results / name).exists()
    figures = sorted(p.name for p in (fake_results / "figures").glob("comparison_*.png"))
    assert figures == ["comparison_confusion_matrices.png", "comparison_efficiency.png", "comparison_metrics.png",
                       "comparison_roc.png", "comparison_training_curves.png"]
