"""Helpers shared by every script: seeding, device choice, config loading, timing."""

from __future__ import annotations

import copy
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch so runs can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """DataLoader `worker_init_fn`: give every worker its own reproducible random state."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    """Use the GPU when there is one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of `base` with `override` applied recursively (nested dicts are merged, not replaced)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(model_config: Path | None = None, base_config: Path = BASE_CONFIG) -> dict:
    """Load `base.yaml` and apply a model config on top of it."""
    config = load_yaml(base_config)
    if model_config is not None:
        config = deep_merge(config, load_yaml(model_config))
    return config


def save_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def resolve_path(path: str | Path) -> Path:
    """Interpret a relative config path relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


class Timer:
    """Context manager that measures wall-clock seconds: `with Timer() as t: ...; t.seconds`."""

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc: Any) -> None:
        self.seconds = time.perf_counter() - self.start
