"""Shared test setup: import path and a tiny synthetic dataset (no real data or GPU needed)."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pneumonia.splits import save_splits  # noqa: E402
from pneumonia.utils import deep_merge, load_config  # noqa: E402


@pytest.fixture()
def config(tmp_path: Path) -> dict:
    """base.yaml with the split folder pointing to a temporary directory."""
    return deep_merge(load_config(), {"data": {"splits_dir": str(tmp_path / "splits"), "num_workers": 0}})


@pytest.fixture()
def fake_data(tmp_path: Path, config: dict) -> Path:
    """Images of different sizes and colour modes (like the real dataset), 3 NORMAL : 9 PNEUMONIA per split."""
    data_dir = tmp_path / "chest_xray"
    tables = {}
    for split in ("train", "val", "test"):
        rows = []
        for label, class_name, count in ((0, "NORMAL", 3), (1, "PNEUMONIA", 9)):
            for i in range(count):
                rel = f"{split}/{class_name}/img{i}.jpeg"
                (data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                mode, size = ("RGB", (300, 200)) if i % 2 else ("L", (180, 260))
                Image.new(mode, size, color=40 * i % 255 if mode == "L" else (10, 20, 30)).save(data_dir / rel)
                rows.append({"path": rel, "label": label, "class_name": class_name,
                             "patient_id": f"{class_name}/p{i}", "source_split": split})
        tables[split] = pd.DataFrame(rows)
    save_splits(tables, Path(config["data"]["splits_dir"]))
    return data_dir
