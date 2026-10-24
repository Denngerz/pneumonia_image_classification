"""Split checks: patient IDs, no leakage, class ratio, determinism. Uses a fake dataset of empty files."""

from pathlib import Path

import pandas as pd
import pytest

from pneumonia.splits import make_splits, patient_id_from_filename, save_splits

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename, expected", [
    ("person1000_bacteria_2931.jpeg", "person1000"),
    ("person1_virus_6.jpeg", "person1"),
    ("IM-0115-0001.jpeg", "IM-0115"),
    ("NORMAL2-IM-0427-0001.jpeg", "NORMAL2-IM-0427"),
    ("unknown.jpeg", "unknown"),
])
def test_patient_id_from_filename(filename, expected):
    assert patient_id_from_filename(filename) == expected


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


@pytest.fixture()
def fake_dataset(tmp_path: Path) -> Path:
    """60 NORMAL patients x 2 images, 90 PNEUMONIA patients x 2-3 images, plus tiny val/test folders."""
    root = tmp_path / "chest_xray"
    for i in range(60):
        for k in range(2):
            _touch(root / "train" / "NORMAL" / f"IM-{i:04d}-{k + 1:04d}.jpeg")
    for i in range(90):
        for k in range(2 + i % 2):
            kind = "bacteria" if i % 3 else "virus"
            _touch(root / "train" / "PNEUMONIA" / f"person{i}_{kind}_{k}.jpeg")
    _touch(root / "val" / "NORMAL" / "NORMAL2-IM-1427-0001.jpeg")
    _touch(root / "val" / "PNEUMONIA" / "person500_bacteria_1.jpeg")
    _touch(root / "test" / "NORMAL" / "IM-0001-0001.jpeg")
    _touch(root / "test" / "PNEUMONIA" / "person1_virus_6.jpeg")
    _touch(root / "train" / "NORMAL" / ".DS_Store")
    return root


def test_no_patient_in_train_and_val(fake_dataset):
    splits = make_splits(fake_dataset)
    overlap = set(splits["train"]["patient_id"]) & set(splits["val"]["patient_id"])
    assert not overlap


def test_val_fraction_and_class_ratio(fake_dataset):
    splits = make_splits(fake_dataset, val_fraction=0.15)
    pool_size = len(splits["train"]) + len(splits["val"])
    assert pool_size == 60 * 2 + 225 + 2
    for label in (0, 1):
        class_total = (splits["train"]["label"] == label).sum() + (splits["val"]["label"] == label).sum()
        val_share = (splits["val"]["label"] == label).sum() / class_total
        assert 0.12 <= val_share <= 0.20


def test_test_split_is_official_test_folder(fake_dataset):
    test_df = make_splits(fake_dataset)["test"]
    assert set(test_df["source_split"]) == {"test"}
    assert len(test_df) == 2


def test_splits_are_deterministic(fake_dataset, tmp_path):
    save_splits(make_splits(fake_dataset, seed=42), tmp_path / "a")
    save_splits(make_splits(fake_dataset, seed=42), tmp_path / "b")
    for name in ("train", "val", "test"):
        assert (tmp_path / "a" / f"{name}.csv").read_bytes() == (tmp_path / "b" / f"{name}.csv").read_bytes()


def test_committed_splits_have_no_leakage():
    splits_dir = REPO_ROOT / "data" / "splits"
    if not (splits_dir / "train.csv").exists():
        pytest.skip("split files not generated yet")
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    assert not set(train_df["patient_id"]) & set(val_df["patient_id"])
    assert not set(train_df["path"]) & set(test_df["path"])
    assert set(test_df["source_split"]) == {"test"}
