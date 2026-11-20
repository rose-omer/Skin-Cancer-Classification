"""Train–test kopya denetimi: sha256 (birebir kopya) ve 64 bit dHash (yakın kopya).

Kullanım:
    python -m skinlesion.dedup --data-root data --out reports/dedup.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from skinlesion.data import list_images

DEFAULT_MAX_HAMMING = 4


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Fark hash'i: gri görüntüyü (hash_size+1) x hash_size boyuta indirir.

    Her satırda yatay komşu pikselleri karşılaştırır; 8x8 = 64 bit üretir.
    """
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def dhash_file(path: str | Path) -> int:
    with Image.open(path) as img:
        return dhash(img)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass(frozen=True)
class DuplicatePair:
    train_path: str
    test_path: str
    hamming: int
    exact: bool


def find_cross_duplicates(
    train_paths: list[str],
    test_paths: list[str],
    max_hamming: int = DEFAULT_MAX_HAMMING,
) -> list[DuplicatePair]:
    """Train ve test arasında Hamming ≤ max_hamming olan bütün çiftleri döner."""
    train_hash = np.array([dhash_file(p) for p in train_paths], dtype=np.uint64)
    test_hash = np.array([dhash_file(p) for p in test_paths], dtype=np.uint64)
    train_sha = [sha256_file(p) for p in train_paths]
    test_sha = [sha256_file(p) for p in test_paths]

    pairs: list[DuplicatePair] = []
    for j, test_value in enumerate(test_hash):
        distances = np.bitwise_count(train_hash ^ test_value)
        for i in np.flatnonzero(distances <= max_hamming):
            pairs.append(
                DuplicatePair(
                    train_path=train_paths[i],
                    test_path=test_paths[j],
                    hamming=int(distances[i]),
                    exact=train_sha[i] == test_sha[j],
                )
            )
    return sorted(pairs, key=lambda p: (p.hamming, p.test_path))


def write_report(pairs: list[DuplicatePair], out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["train_path", "test_path", "hamming", "exact"])
        for p in pairs:
            writer.writerow([p.train_path, p.test_path, p.hamming, int(p.exact)])
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train–test kopya denetimi")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="reports/dedup.csv")
    parser.add_argument("--max-hamming", type=int, default=DEFAULT_MAX_HAMMING)
    args = parser.parse_args(argv)

    root = Path(args.data_root)
    train = [str(p) for p, _ in list_images(root / "train")]
    test = [str(p) for p, _ in list_images(root / "test")]
    pairs = find_cross_duplicates(train, test, args.max_hamming)
    out = write_report(pairs, args.out)
    exact = sum(p.exact for p in pairs)
    print(f"{len(pairs)} yakın kopya çifti ({exact} birebir) -> {out}")


if __name__ == "__main__":
    main()
