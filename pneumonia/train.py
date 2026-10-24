"""One training loop for every model: AdamW, warm-up + cosine LR, AMP on GPU, early stopping on validation ROC-AUC."""

from __future__ import annotations

import math
import platform
from pathlib import Path

import pandas as pd
import timm
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pneumonia.data import build_dataloaders
from pneumonia.models import count_parameters, create_model
from pneumonia.utils import Timer, get_device, save_config, set_seed


def run_name(config: dict, debug: bool = False) -> str:
    """`<model>_seed<seed>`, e.g. `resnet50_seed42` (with a `_debug` suffix for debug runs)."""
    name = f"{config['model']['name']}_seed{config['seed']}"
    return f"{name}_debug" if debug else name


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    if train_cfg["optimizer"] != "adamw":
        raise ValueError(f"Unsupported optimizer: {train_cfg['optimizer']}")
    return torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict, steps_per_epoch: int,
                    epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-step schedule: linear warm-up for `warmup_epochs`, then cosine decay to 0 at the last epoch."""
    train_cfg = config["training"]
    if train_cfg["scheduler"] != "cosine":
        raise ValueError(f"Unsupported scheduler: {train_cfg['scheduler']}")
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = min(steps_per_epoch * train_cfg["warmup_epochs"], total_steps - 1)

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler, scaler: torch.amp.GradScaler,
                    device: torch.device, use_amp: bool) -> float:
    """One pass over the (class-balanced) training batches. Returns the mean training loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_images = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * len(labels)
        total_images += len(labels)
    return total_loss / total_images


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            use_amp: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(labels, logits)` for every image of the loader, in loader order (on the CPU)."""
    model.eval()
    all_labels, all_logits = [], []
    for images, labels in loader:
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images.to(device, non_blocking=True))
        all_labels.append(labels)
        all_logits.append(logits.float().cpu())
    return torch.cat(all_labels), torch.cat(all_logits)


def validation_scores(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    """Loss, ROC-AUC and accuracy on the validation set — the signals used for early stopping."""
    probs = torch.softmax(logits, dim=1)[:, 1]
    return {
        "val_loss": nn.functional.cross_entropy(logits, labels).item(),
        "val_roc_auc": float(roc_auc_score(labels.numpy(), probs.numpy())),
        "val_accuracy": (logits.argmax(dim=1) == labels).float().mean().item(),
    }


def is_improvement(scores: dict[str, float], best: dict[str, float] | None) -> bool:
    """Higher validation ROC-AUC wins; on a tie the lower validation loss wins."""
    if best is None:
        return True
    if scores["val_roc_auc"] != best["val_roc_auc"]:
        return scores["val_roc_auc"] > best["val_roc_auc"]
    return scores["val_loss"] < best["val_loss"]


def environment_info(device: torch.device) -> dict:
    """Recorded with each run so results can be traced to the hardware and library versions used."""
    return {
        "device": device.type,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": str(torch.__version__),  # TorchVersion is a str subclass that YAML cannot write
        "timm": str(timm.__version__),
    }


def fit(config: dict, data_dir: Path, results_dir: Path, runs_dir: Path, debug: bool = False) -> dict:
    """Train one model with the shared protocol and keep the best epoch.

    Writes `config.yaml` and `history.csv` (after every epoch) to `results_dir/<run>/`, and `best.pt`, `last.pt` and
    TensorBoard logs to `runs_dir/<run>/`. Returns the model with the best weights loaded plus run information.
    """
    set_seed(config["seed"])
    device = get_device()
    name = run_name(config, debug)
    run_results, run_files = Path(results_dir) / name, Path(runs_dir) / name
    run_results.mkdir(parents=True, exist_ok=True)
    run_files.mkdir(parents=True, exist_ok=True)

    train_cfg = config["training"]
    epochs = config["debug"]["epochs"] if debug else train_cfg["epochs"]
    use_amp = train_cfg["amp"] and device.type == "cuda"
    save_config({**config, "debug_run": debug, "environment": environment_info(device)}, run_results / "config.yaml")

    loaders = build_dataloaders(config, data_dir, debug=debug)
    model = create_model(config["model"]["name"], pretrained=config["model"]["pretrained"]).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(loaders["train"]), epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    writer = SummaryWriter(str(run_files / "tensorboard"))

    history: list[dict] = []
    best: dict | None = None
    epochs_without_improvement = 0
    print(f"{name}: {count_parameters(model):,} parameters, device={device}, epochs<={epochs}")
    with Timer() as training_timer:
        for epoch in range(1, epochs + 1):
            with Timer() as epoch_timer:
                lr = optimizer.param_groups[0]["lr"]
                train_loss = train_one_epoch(model, loaders["train"], optimizer, scheduler, scaler, device, use_amp)
                scores = validation_scores(*predict(model, loaders["val"], device, use_amp))
            row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, **scores, "epoch_seconds": epoch_timer.seconds}
            improved = is_improvement(scores, best)
            if improved:
                best = {**row}
                epochs_without_improvement = 0
                torch.save(model.state_dict(), run_files / "best.pt")
            else:
                epochs_without_improvement += 1
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict()}, run_files / "last.pt")
            history.append({**row, "best": improved})
            pd.DataFrame(history).to_csv(run_results / "history.csv", index=False, lineterminator="\n")
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
            print(f"epoch {epoch:2d} | train_loss {train_loss:.4f} | val_loss {scores['val_loss']:.4f} | "
                  f"val_auc {scores['val_roc_auc']:.4f} | val_acc {scores['val_accuracy']:.4f} | "
                  f"{epoch_timer.seconds:.0f}s{' *' if improved else ''}")
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(f"Early stopping: no validation ROC-AUC improvement for {epochs_without_improvement} epochs")
                break
    writer.close()

    model.load_state_dict(torch.load(run_files / "best.pt", map_location=device, weights_only=True))
    return {
        "model": model,
        "loaders": loaders,
        "device": device,
        "run_name": name,
        "results_dir": run_results,
        "runs_dir": run_files,
        "best_epoch": best["epoch"],
        "epochs_trained": len(history),
        "training_seconds": training_timer.seconds,
        "use_amp": use_amp,
    }
