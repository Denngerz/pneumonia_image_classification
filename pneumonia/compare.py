"""Collect all runs from results/, build the comparison table and the report figures.

Primary comparison metric is the test ROC-AUC (threshold-independent, stable across seeds). Metrics at the fixed 0.5
threshold are reported too, with the caveat that they vary much more between seeds (team decision, see Plan.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from pneumonia.models import MODEL_NAMES  # noqa: E402

METRICS = ["roc_auc", "accuracy", "recall", "specificity", "precision", "f1"]
METRIC_LABELS = {"roc_auc": "ROC-AUC", "accuracy": "Accuracy", "recall": "Recall (sensitivity)",
                 "specificity": "Specificity", "precision": "Precision", "f1": "F1"}
MODEL_LABELS = {"baseline_cnn": "Baseline CNN", "vgg16": "VGG16", "resnet50": "ResNet50",
                "densenet121": "DenseNet121", "efficientnet_b0": "EfficientNet-B0",
                "mobilenetv3_large": "MobileNetV3-L"}

# Fixed categorical order (one colour per model in every figure, never reassigned by rank).
MODEL_COLORS = dict(zip(MODEL_NAMES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def collect_runs(results_dir: Path) -> pd.DataFrame:
    """One row per run folder that has a metrics.json (debug runs never land in results/)."""
    rows = []
    for metrics_path in sorted(Path(results_dir).glob("*_seed*/metrics.json")):
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        cm = m["test"]["confusion_matrix"]
        rows.append({
            "run": m["run"], "model": m["model"], "seed": m["seed"],
            **{k: m["test"][k] for k in METRICS},
            "tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"], "tp": cm["tp"],
            "parameters": m["parameters"], "best_epoch": m["best_epoch"], "epochs_trained": m["epochs_trained"],
            "training_minutes": m["training_seconds"] / 60, "inference_ms": m["inference_ms_median"],
            "gpu": m["gpu"],
        })
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs
    order = {name: i for i, name in enumerate(MODEL_NAMES)}
    return runs.sort_values(["model", "seed"], key=lambda s: s.map(order) if s.name == "model" else s,
                            ignore_index=True)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample standard deviation per model over its seeds (in registry order)."""
    columns = METRICS + ["training_minutes", "inference_ms"]
    grouped = runs.groupby("model", sort=False)
    summary = grouped[columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "seeds", grouped["seed"].count())
    summary.insert(1, "parameters", grouped["parameters"].first())
    return summary.reset_index()


def summary_markdown(summary: pd.DataFrame) -> str:
    """Markdown table 'mean ± std' — ready to paste into the report."""
    header = ["Model", "Params (M)", "Seeds"] + [METRIC_LABELS[m] for m in METRICS] + ["Train (min)", "ms / image"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, row in summary.iterrows():
        cells = [MODEL_LABELS.get(row["model"], row["model"]), f"{row['parameters'] / 1e6:.1f}", str(row["seeds"])]
        cells += [_mean_std(row, m, 3) for m in METRICS]
        cells += [_mean_std(row, "training_minutes", 1), _mean_std(row, "inference_ms", 1)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _mean_std(row: pd.Series, metric: str, digits: int) -> str:
    std = row[f"{metric}_std"]
    std_text = "–" if pd.isna(std) else f"{std:.{digits}f}"
    return f"{row[f'{metric}_mean']:.{digits}f} ± {std_text}"


def write_tables(runs: pd.DataFrame, summary: pd.DataFrame, results_dir: Path) -> None:
    runs.to_csv(results_dir / "comparison_runs.csv", index=False, lineterminator="\n", float_format="%.6g")
    summary.to_csv(results_dir / "comparison.csv", index=False, lineterminator="\n", float_format="%.6g")
    note = ("Test set, mean ± standard deviation over seeds 42, 43, 44. ROC-AUC is the primary metric; the other "
            "metrics use the fixed 0.5 threshold and vary much more between seeds.\n\n")
    (results_dir / "comparison.md").write_text("# Model comparison\n\n" + note + summary_markdown(summary),
                                               encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------------------------- figures

def _style(ax: plt.Axes) -> None:
    """Recessive grid and axes, ink-coloured text."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=9)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _figure(*args, **kwargs) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(*args, squeeze=False, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def plot_metric_bars(runs: pd.DataFrame, path: Path) -> None:
    """Small multiples: one panel per metric, bar = mean over seeds, dots = individual seeds."""
    models = list(dict.fromkeys(runs["model"]))
    fig, axes = _figure(2, 3, figsize=(13, 7))
    x = np.arange(len(models))
    for ax, metric in zip(axes.ravel(), METRICS):
        means = [runs.loc[runs["model"] == m, metric].mean() for m in models]
        ax.bar(x, means, width=0.62, color=[MODEL_COLORS[m] for m in models], edgecolor=SURFACE, linewidth=2)
        for i, m in enumerate(models):
            values = runs.loc[runs["model"] == m, metric]
            ax.scatter(np.full(len(values), i), values, s=22, color=INK, zorder=3, edgecolors=SURFACE, linewidths=1)
        low = min(runs[metric].min(), min(means))
        ax.set_ylim(max(0.0, low - 0.05), 1.0)
        ax.set_xticks(x, [MODEL_LABELS[m] for m in models], rotation=35, ha="right")
        ax.set_title(METRIC_LABELS[metric] + (" — primary" if metric == "roc_auc" else " @ 0.5"), fontsize=11)
        _style(ax)
    fig.suptitle("Test metrics per model (bar = mean of 3 seeds, dots = single seeds)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def mean_roc_curve(predictions: list[pd.DataFrame], grid: np.ndarray) -> np.ndarray:
    """Average the TPR of several runs over a common FPR grid (vertical averaging)."""
    tprs = []
    for df in predictions:
        fpr, tpr, _ = roc_curve(df["label"], df["prob_pneumonia"])
        tprs.append(np.interp(grid, fpr, tpr))
    mean = np.mean(tprs, axis=0)
    mean[0] = 0.0
    return mean


def plot_roc_curves(runs: pd.DataFrame, results_dir: Path, path: Path) -> None:
    """ROC curve per model, averaged over its seeds; zoomed panel on the top-left corner."""
    grid = np.linspace(0, 1, 501)
    fig, axes = _figure(1, 2, figsize=(12, 5.2))
    for ax, (xmax, ymin) in zip(axes[0], [(1.0, 0.0), (0.4, 0.8)]):
        for model in dict.fromkeys(runs["model"]):
            preds = [pd.read_csv(results_dir / run / "test_predictions.csv")
                     for run in runs.loc[runs["model"] == model, "run"]]
            auc = runs.loc[runs["model"] == model, "roc_auc"].mean()
            ax.plot(grid, mean_roc_curve(preds, grid), color=MODEL_COLORS[model], linewidth=2,
                    label=f"{MODEL_LABELS[model]} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
        ax.set_xlim(0, xmax)
        ax.set_ylim(ymin, 1.0 if ymin == 0 else 1.005)
        ax.set_xlabel("False positive rate (1 − specificity)")
        ax.set_ylabel("True positive rate (recall)")
        _style(ax)
    axes[0][0].set_title("ROC curves (mean over 3 seeds)", fontsize=11)
    axes[0][1].set_title("Zoom: top-left corner", fontsize=11)
    axes[0][0].legend(loc="lower right", fontsize=9, frameon=False, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_training_curves(runs: pd.DataFrame, results_dir: Path, path: Path) -> None:
    """One panel per model: training loss (solid) and validation loss (dashed) of every seed."""
    models = list(dict.fromkeys(runs["model"]))
    fig, axes = _figure(2, 3, figsize=(13, 7), sharey=True)
    for ax, model in zip(axes.ravel(), models):
        for run in runs.loc[runs["model"] == model, "run"]:
            history = pd.read_csv(results_dir / run / "history.csv")
            ax.plot(history["epoch"], history["train_loss"], color=MODEL_COLORS[model], linewidth=1.6)
            ax.plot(history["epoch"], history["val_loss"], color=MODEL_COLORS[model], linewidth=1.2,
                    linestyle="--", alpha=0.8)
        ax.set_title(MODEL_LABELS[model], fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, 25)
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cross-entropy loss")
    axes[0][0].plot([], [], color=INK_SECONDARY, linewidth=1.6, label="training loss")
    axes[0][0].plot([], [], color=INK_SECONDARY, linewidth=1.2, linestyle="--", label="validation loss")
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.suptitle("Training curves, 3 seeds per model (lines end where early stopping fired)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion_matrices(runs: pd.DataFrame, path: Path) -> None:
    """Confusion matrix per model, averaged over seeds (counts out of 624 test images)."""
    models = list(dict.fromkeys(runs["model"]))
    fig, axes = _figure(2, 3, figsize=(11, 7.4))
    for ax, model in zip(axes.ravel(), models):
        r = runs[runs["model"] == model]
        cm = np.array([[r["tn"].mean(), r["fp"].mean()], [r["fn"].mean(), r["tp"].mean()]])
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for (i, j), count in np.ndenumerate(cm):
            ax.text(j, i, f"{count:.0f}\n({share[i, j]:.0%})", ha="center", va="center", fontsize=10,
                    color="white" if share[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1], ["NORMAL", "PNEUMONIA"])
        ax.set_yticks([0, 1], ["NORMAL", "PNEUMONIA"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(MODEL_LABELS[model], fontsize=11)
        ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=9)
    fig.suptitle("Confusion matrices at threshold 0.5 (mean over 3 seeds; % of the true class)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# Label offsets in points, so the three models clustered around 8–10 ms do not overlap.
EFFICIENCY_LABEL_OFFSETS = {"resnet50": (-12, 18), "vgg16": (12, 14), "mobilenetv3_large": (-14, -34),
                            "efficientnet_b0": (10, -6), "densenet121": (10, -6), "baseline_cnn": (10, -6)}


def plot_efficiency(summary: pd.DataFrame, path: Path) -> None:
    """ROC-AUC against inference time; marker area grows with the parameter count."""
    fig, axes = _figure(1, 1, figsize=(7.5, 5))
    ax = axes[0][0]
    for _, row in summary.iterrows():
        model = row["model"]
        size = 40 + 260 * np.sqrt(row["parameters"] / summary["parameters"].max())
        ax.errorbar(row["inference_ms_mean"], row["roc_auc_mean"], yerr=row["roc_auc_std"], fmt="none",
                    ecolor=INK_MUTED, elinewidth=1, capsize=3, zorder=2)
        ax.scatter(row["inference_ms_mean"], row["roc_auc_mean"], s=size, color=MODEL_COLORS[model],
                   edgecolors=SURFACE, linewidths=2, zorder=3)
        dx, dy = EFFICIENCY_LABEL_OFFSETS.get(model, (10, -6))
        ax.annotate(f"{MODEL_LABELS[model]}\n{row['parameters'] / 1e6:.1f} M params",
                    (row["inference_ms_mean"], row["roc_auc_mean"]), textcoords="offset points", xytext=(dx, dy),
                    ha="right" if dx < 0 else "left", va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlabel("Inference time, ms per image (Tesla T4, batch 1)")
    ax.set_ylabel("Test ROC-AUC (mean ± std over 3 seeds)")
    ax.set_title("Quality vs speed (marker size ~ number of parameters)", fontsize=11)
    ax.set_xlim(0, summary["inference_ms_mean"].max() * 1.35)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def build_report(results_dir: Path, figures_dir: Path) -> pd.DataFrame:
    """Write the comparison tables and all figures; return the per-model summary."""
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(results_dir)
    if runs.empty:
        raise FileNotFoundError(f"No */metrics.json found in {results_dir}")
    summary = summarize(runs)
    write_tables(runs, summary, results_dir)
    plot_metric_bars(runs, figures_dir / "comparison_metrics.png")
    plot_roc_curves(runs, results_dir, figures_dir / "comparison_roc.png")
    plot_training_curves(runs, results_dir, figures_dir / "comparison_training_curves.png")
    plot_confusion_matrices(runs, figures_dir / "comparison_confusion_matrices.png")
    plot_efficiency(summary, figures_dir / "comparison_efficiency.png")
    return summary
