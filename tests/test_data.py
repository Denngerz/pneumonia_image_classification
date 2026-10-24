"""Data pipeline checks on a few synthetic images (no dataset needed)."""

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from pneumonia.data import build_dataloaders, build_transforms, debug_subset, load_image, make_weighted_sampler
from pneumonia.splits import save_splits
from pneumonia.utils import deep_merge, load_config


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


def test_config_merge_keeps_shared_settings():
    config = deep_merge(load_config(), {"model": {"name": "resnet50"}})
    assert config["model"] == {"name": "resnet50", "pretrained": True}
    assert config["data"]["image_size"] == 224
    assert config["training"]["batch_size"] == 32


@pytest.mark.parametrize("train", [True, False])
def test_transforms_output_shape(config, train, tmp_path):
    Image.new("RGB", (517, 333)).save(tmp_path / "x.jpeg")
    image = load_image(tmp_path / "x.jpeg", config["data"]["image_size"])
    tensor = build_transforms(config, train=train)(image.convert("RGB"))
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_cache_does_not_change_images(config, fake_data):
    cached = build_dataloaders(deep_merge(config, {"data": {"cache_images": True}}), fake_data)["val"]
    uncached = build_dataloaders(deep_merge(config, {"data": {"cache_images": False}}), fake_data)["val"]
    for (a, _), (b, _) in zip(cached, uncached):
        assert torch.equal(a, b)


def test_dataloaders_give_normalized_batches(config, fake_data):
    loaders = build_dataloaders(config, fake_data)
    images, labels = next(iter(loaders["val"]))
    assert images.shape[1:] == (3, 224, 224)
    assert set(labels.tolist()) <= {0, 1}
    assert len(loaders["test"].dataset) == 12


def test_val_and_test_order_is_fixed(config, fake_data):
    first = [labels.tolist() for _, labels in build_dataloaders(config, fake_data)["test"]]
    second = [labels.tolist() for _, labels in build_dataloaders(config, fake_data)["test"]]
    assert first == second


def test_weighted_sampler_balances_classes():
    labels = [0] * 100 + [1] * 300
    sampler = make_weighted_sampler(labels, seed=42)
    drawn = Counter(labels[i] for _ in range(20) for i in sampler)
    assert len(list(sampler)) == len(labels)
    assert 0.45 <= drawn[0] / sum(drawn.values()) <= 0.55


def test_weighted_sampler_is_reproducible():
    labels = [0] * 10 + [1] * 30
    assert list(make_weighted_sampler(labels, 7)) == list(make_weighted_sampler(labels, 7))


def test_debug_subset_takes_few_images_per_class():
    df = pd.DataFrame({"path": [f"x{i}" for i in range(50)], "label": [0] * 10 + [1] * 40})
    subset = debug_subset(df, images_per_class=4, seed=42)
    assert subset["label"].value_counts().to_dict() == {0: 4, 1: 4}
    assert subset.equals(debug_subset(df, images_per_class=4, seed=42))
