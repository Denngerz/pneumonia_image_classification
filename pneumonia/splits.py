"""Patient-level, class-stratified train/val split of the Kermany chest X-ray dataset.

The official `val/` folder has only 16 images, so official `train/` + `val/` are pooled and re-split.
All images of one patient go to the same split (no leakage). The official `test/` folder is kept as is.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

CLASS_NAMES = ("NORMAL", "PNEUMONIA")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
SPLIT_COLUMNS = ["path", "label", "class_name", "patient_id", "source_split"]

# PNEUMONIA: person1000_bacteria_2931.jpeg -> person1000
_PERSON_RE = re.compile(r"^(person\d+)_", re.IGNORECASE)
# NORMAL: IM-0115-0001.jpeg -> IM-0115, NORMAL2-IM-0427-0001.jpeg -> NORMAL2-IM-0427
_NORMAL_RE = re.compile(r"^((?:NORMAL2-)?IM-\d+)-\d+", re.IGNORECASE)


def patient_id_from_filename(filename: str) -> str:
    """Return the patient ID encoded in a file name, or the file stem if no pattern matches."""
    stem = Path(filename).stem
    for pattern in (_PERSON_RE, _NORMAL_RE):
        match = pattern.match(stem)
        if match:
            return match.group(1)
    return stem


def scan_folder(data_dir: Path, source_split: str) -> pd.DataFrame:
    """List all images of one official folder (`train`, `val` or `test`) as split rows."""
    rows = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / source_split / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing folder: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES or image_path.name.startswith("."):
                continue
            rows.append({
                "path": image_path.relative_to(data_dir).as_posix(),
                "label": label,
                "class_name": class_name,
                "patient_id": f"{class_name}/{patient_id_from_filename(image_path.name)}",
                "source_split": source_split,
            })
    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


def split_by_patient(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into train/val so that no patient is in both and each class keeps ~`val_fraction` in val.

    Per class, patient groups are shuffled with `seed` and moved to val until the class reaches its
    target image count. Deterministic for a given input and seed.
    """
    rng = random.Random(seed)
    val_patients: set[str] = set()
    for _, class_df in df.groupby("label", sort=True):
        counts = class_df.groupby("patient_id").size().sort_index()
        patients = list(counts.index)
        rng.shuffle(patients)
        target = round(len(class_df) * val_fraction)
        taken = 0
        for patient in patients:
            if taken >= target:
                break
            val_patients.add(patient)
            taken += int(counts[patient])
    is_val = df["patient_id"].isin(val_patients)
    return _sorted(df[~is_val]), _sorted(df[is_val])


def make_splits(data_dir: Path, val_fraction: float = 0.15, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Build the train/val/test tables from the official dataset folders."""
    pool = pd.concat([scan_folder(data_dir, "train"), scan_folder(data_dir, "val")], ignore_index=True)
    train_df, val_df = split_by_patient(pool, val_fraction, seed)
    return {"train": train_df, "val": val_df, "test": _sorted(scan_folder(data_dir, "test"))}


def save_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write `<name>.csv` for every split (LF line endings so files are identical on every OS)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(output_dir / f"{name}.csv", index=False, lineterminator="\n")


def load_split(splits_dir: Path, name: str) -> pd.DataFrame:
    """Read one split CSV (`train`, `val` or `test`)."""
    return pd.read_csv(splits_dir / f"{name}.csv")


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("path").reset_index(drop=True)
