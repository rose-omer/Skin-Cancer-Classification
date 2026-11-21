"""Veri listeleme, train/val/test bölünmesi ve dönüşümler.

Klasör düzeni (Kaggle fanconic seti):
    data/train/{benign,malignant}/*.jpg
    data/test/{benign,malignant}/*.jpg

Bölünme (önce `skinlesion.dedup` koşulursa kopyalar train/val'den çıkarılır):
    python -m skinlesion.data --data-root data --splits-dir splits --exclude-dups reports/dedup.csv
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision.transforms import v2

from skinlesion.errors import TestSetAccessError

CLASSES = ("benign", "malignant")  # etiket 1 = malignant
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = ("train", "val", "test")

Sample = tuple[Path, int]


def list_images(folder: str | Path) -> list[Sample]:
    """`folder/<sınıf>/*` altındaki görüntüleri sıralı ve etiketli döner."""
    folder = Path(folder)
    samples: list[Sample] = []
    for label, name in enumerate(CLASSES):
        class_dir = folder / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Sınıf klasörü yok: {class_dir}")
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        samples.extend((p, label) for p in files)
    return samples


def stratified_val_split(
    labels: Sequence[int], val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sınıf oranını koruyarak train/val indekslerini döner."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    labels_arr = np.asarray(labels)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels_arr)), labels_arr))
    return np.sort(train_idx), np.sort(val_idx)


def grouped_val_split(
    labels: Sequence[int], groups: Sequence[str], n_splits: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hasta düzeyinde bölünme (ISIC 2020 `patient_id`). Bir hasta yalnız bir kümede olur."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels_arr = np.asarray(labels)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels_arr)), labels_arr, groups))
    return np.sort(train_idx), np.sort(val_idx)


def _write_split(path: Path, rows: list[Sample], data_root: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "label"])
        for file, label in rows:
            writer.writerow([file.relative_to(data_root).as_posix(), label])


def read_duplicates(dedup_csv: str | Path) -> set[Path]:
    """`skinlesion.dedup` raporundaki train dosyaları (test kopyası olanlar)."""
    with Path(dedup_csv).open(encoding="utf-8") as fh:
        return {Path(row["train_path"]).resolve() for row in csv.DictReader(fh)}


def make_splits(
    data_root: str | Path,
    splits_dir: str | Path,
    val_fraction: float,
    seed: int,
    exclude: set[Path] | None = None,
) -> dict[str, int]:
    """`splits/{train,val,test}.csv` yazar. Resmî test klasörü olduğu gibi kalır.

    `exclude`: test kümesinde kopyası olan train dosyaları. Train ve val'den çıkarılır.
    """
    data_root = Path(data_root)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_all = list_images(data_root / "train")
    if exclude:
        train_all = [(p, y) for p, y in train_all if p.resolve() not in exclude]
    test = list_images(data_root / "test")
    train_idx, val_idx = stratified_val_split([y for _, y in train_all], val_fraction, seed)
    split_rows = {
        "train": [train_all[i] for i in train_idx],
        "val": [train_all[i] for i in val_idx],
        "test": test,
    }
    for name, rows in split_rows.items():
        _write_split(splits_dir / f"{name}.csv", rows, data_root)
    return {name: len(rows) for name, rows in split_rows.items()}


def read_split(
    splits_dir: str | Path, name: str, data_root: str | Path, *, allow_test: bool = False
) -> list[Sample]:
    """Bölünme dosyasını okur. Test kümesi yalnız `allow_test=True` ile açılır."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"Bilinmeyen bölünme: {name}")
    if name == "test" and not allow_test:
        raise TestSetAccessError(
            "Test kümesi yalnız `skinlesion.evaluate --final` ve `skinlesion.embed --final` "
            "ile, her biri bir kez kullanılır."
        )
    data_root = Path(data_root)
    with (Path(splits_dir) / f"{name}.csv").open(encoding="utf-8") as fh:
        return [(data_root / row["path"], int(row["label"])) for row in csv.DictReader(fh)]


def build_transforms(
    img_size: int, mean: Sequence[float], std: Sequence[float], *, train: bool
) -> v2.Compose:
    """Omurganın kendi normalizasyonu (timm data config) ile dönüşüm zinciri."""
    tail = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    if not train:
        return v2.Compose([v2.Resize(img_size), v2.CenterCrop(img_size), *tail])
    return v2.Compose(
        [
            v2.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(20),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            *tail,
        ]
    )


class ImageDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform: v2.Compose) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB")), label


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sınıf oranını koruyan train/val bölünmesi")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-dups",
        default=None,
        help="skinlesion.dedup raporu; testte kopyası olan train görüntüleri çıkarılır",
    )
    args = parser.parse_args(argv)
    exclude = read_duplicates(args.exclude_dups) if args.exclude_dups else None
    counts = make_splits(args.data_root, args.splits_dir, args.val_fraction, args.seed, exclude)
    print(", ".join(f"{k}={v}" for k, v in counts.items()))
    if exclude:
        print(f"Çıkarılan kopya: {len(exclude)} train görüntüsü")


if __name__ == "__main__":
    main()
