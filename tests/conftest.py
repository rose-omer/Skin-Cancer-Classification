"""Ortak fixture'lar: küçük sahte görüntüler ve rastgele ağırlıklı küçük checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from skinlesion.data import CLASSES
from skinlesion.model import (
    OOD_FILE,
    CheckpointConfig,
    create_model,
    normalization,
    save_checkpoint,
)
from skinlesion.ood import fit_ood

TINY_MODEL = "test_efficientnet"  # timm test modeli; ağ indirmez
TINY_SIZE = 64


def fake_image(seed: int, label: int, size: int = 48) -> Image.Image:
    """Sınıfa bağlı renk tonu + gürültü. Malign örnekler daha koyu."""
    rng = np.random.default_rng(seed)
    base = np.array([170, 120, 100]) if label == 0 else np.array([90, 55, 45])
    pixels = base + rng.normal(0, 25, size=(size, size, 3))
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def write_fake_dataset(root: Path, per_class: dict[str, int]) -> Path:
    """`root/{train,test}/{benign,malignant}/*.png` yazar."""
    seed = 0
    for split, n in per_class.items():
        for label, name in enumerate(CLASSES):
            folder = root / split / name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                fake_image(seed, label).save(folder / f"{name}_{i:03d}.png")
                seed += 1
    return root


@pytest.fixture
def fake_data(tmp_path: Path) -> Path:
    return write_fake_dataset(tmp_path / "data", {"train": 20, "test": 6})


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    model = create_model(TINY_MODEL, pretrained=False).eval()
    mean, std = normalization(model)
    cfg = CheckpointConfig(
        model_name=TINY_MODEL,
        img_size=TINY_SIZE,
        mean=mean,
        std=std,
        threshold=0.42,
        temperature=1.5,
        info={"experiment_id": "tiny"},
    )
    out = save_checkpoint(model, cfg, tmp_path_factory.mktemp("ckpt"))
    with torch.inference_mode():
        x = torch.randn(64, 3, TINY_SIZE, TINY_SIZE)
        feats = model.forward_head(model.forward_features(x), pre_logits=True).numpy()
    fit_ood(feats, 0.99).save(out / OOD_FILE)
    return out
