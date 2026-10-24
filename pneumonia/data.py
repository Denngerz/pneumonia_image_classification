"""Images → tensors → batches. One pipeline for every model, driven only by the config."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

from pneumonia.splits import load_split
from pneumonia.utils import resolve_path, seed_worker

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_NAMES = ("train", "val", "test")


def load_image(path: Path, size: int) -> Image.Image:
    """Open an X-ray as grayscale and resize it to `size`×`size` (the deterministic first step for every image)."""
    with Image.open(path) as img:
        # Some files are stored as RGB although they are grayscale; make all of them single-channel.
        return img.convert("L").resize((size, size), Image.Resampling.BILINEAR)


class ChestXrayDataset(Dataset):
    """Reads the images listed in a split table. Returns `(image_tensor, label)`.

    With `cache=True` every image is decoded and resized once, up front, and kept in memory (~50 KB each):
    decoding the original JPEGs (up to ~2900 px) every epoch would leave the GPU waiting.
    """

    def __init__(self, df: pd.DataFrame, data_dir: Path, image_size: int, transform: v2.Transform,
                 cache: bool = False) -> None:
        self.paths = [Path(data_dir) / p for p in df["path"]]
        self.labels = df["label"].astype(int).tolist()
        self.image_size = image_size
        self.transform = transform
        self.cache = None
        if cache:
            with ThreadPoolExecutor() as pool:
                self.cache = list(pool.map(lambda p: load_image(p, image_size), self.paths))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.cache[index] if self.cache is not None else load_image(self.paths[index], self.image_size)
        # Copy the gray channel into 3 channels for ImageNet-pretrained models.
        return self.transform(image.convert("RGB")), self.labels[index]


def build_transforms(config: dict, train: bool) -> v2.Compose:
    """Augment (training only), scale to [0, 1] and normalize with ImageNet statistics.

    Images arrive already resized to `image_size` by `load_image`.
    """
    steps: list[v2.Transform] = []
    if train:
        aug = config["augmentation"]
        steps += [
            v2.RandomAffine(degrees=aug["rotation_degrees"], translate=(aug["translate"], aug["translate"]),
                            scale=tuple(aug["scale"])),
            v2.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]),
        ]
        if aug.get("horizontal_flip"):
            steps.append(v2.RandomHorizontalFlip())
    steps += [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(steps)


def make_weighted_sampler(labels: list[int], seed: int) -> WeightedRandomSampler:
    """Draw each class equally often (weight = 1 / class size); one epoch still has len(labels) images."""
    labels_tensor = torch.as_tensor(labels)
    class_counts = torch.bincount(labels_tensor)
    weights = (1.0 / class_counts.double())[labels_tensor]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True, generator=generator)


def debug_subset(df: pd.DataFrame, images_per_class: int, seed: int) -> pd.DataFrame:
    """A few images per class, chosen reproducibly, for quick CPU checks."""
    parts = [g.sample(min(len(g), images_per_class), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sort_values("path").reset_index(drop=True)


def load_split_tables(config: dict, debug: bool = False) -> dict[str, pd.DataFrame]:
    splits_dir = resolve_path(config["data"]["splits_dir"])
    tables = {name: load_split(splits_dir, name) for name in SPLIT_NAMES}
    if debug:
        tables = {name: debug_subset(df, config["debug"]["images_per_class"], config["seed"])
                  for name, df in tables.items()}
    return tables


def build_dataloaders(config: dict, data_dir: Path, debug: bool = False) -> dict[str, DataLoader]:
    """Train / val / test DataLoaders for any model.

    Training batches come from a class-balancing weighted sampler; val and test are read in a fixed order.
    """
    tables = load_split_tables(config, debug)
    seed = config["seed"]
    batch_size = config["debug"]["batch_size"] if debug else config["training"]["batch_size"]
    num_workers = 0 if debug else config["data"]["num_workers"]
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    loaders = {}
    for name, df in tables.items():
        is_train = name == "train"
        dataset = ChestXrayDataset(df, data_dir, config["data"]["image_size"],
                                   build_transforms(config, train=is_train), cache=config["data"]["cache_images"])
        if is_train:
            sampler = make_weighted_sampler(dataset.labels, seed)
            loaders[name] = DataLoader(dataset, sampler=sampler, generator=torch.Generator().manual_seed(seed),
                                       **common)
        else:
            loaders[name] = DataLoader(dataset, shuffle=False, **common)
    return loaders
