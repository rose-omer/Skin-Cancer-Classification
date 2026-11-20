"""Temel model tabanı: dondurulmuş embedding + LogisticRegression.

Varsayılan omurga DINOv2-base (timm, Apache-2.0). Embedding'ler önbelleğe yazılır.
Gated modeller (MedSigLIP, Derm Foundation) HF_TOKEN ve lisans onayı ister; bu modül onları
indirmez.

Kullanım:
    python -m skinlesion.embed --config configs/effv2s_fanconic.yaml \
        --backbone vit_base_patch14_dinov2.lvd142m
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from skinlesion.config import load_config
from skinlesion.data import ImageDataset, Sample, build_transforms, read_split

DEFAULT_BACKBONE = "vit_base_patch14_dinov2.lvd142m"
C_GRID = (0.01, 0.1, 1.0, 10.0)


def create_backbone(name: str, img_size: int, *, pretrained: bool = True) -> torch.nn.Module:
    kwargs = {"img_size": img_size} if "vit" in name else {}
    return timm.create_model(name, pretrained=pretrained, num_classes=0, **kwargs).eval()


@torch.inference_mode()
def embed(
    backbone: torch.nn.Module, samples: Sequence[Sample], img_size: int, batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    cfg = timm.data.resolve_data_config({}, model=backbone)
    tf = build_transforms(img_size, cfg["mean"], cfg["std"], train=False)
    loader = DataLoader(ImageDataset(samples, tf), batch_size=batch_size, shuffle=False)
    feats, labels = [], []
    for x, y in loader:
        feats.append(backbone(x).float().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def fit_probe(
    train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray
) -> tuple[object, float, float]:
    """C değerini val AUROC'a göre seçer. (model, C, val_auroc) döner."""
    best = (None, 0.0, -1.0)
    for c in C_GRID:
        probe = make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
        )
        probe.fit(train_x, train_y)
        auroc = float(roc_auc_score(val_y, probe.predict_proba(val_x)[:, 1]))
        if auroc > best[2]:
            best = (probe, c, auroc)
    return best


def cache_path(cache_dir: Path, splits_dir: str | Path, name: str, img_size: int) -> Path:
    """Önbellek anahtarı bölünme dosyasının içeriğine bağlıdır.

    Bölünme yeniden üretilirse eski embedding'ler sessizce kullanılmaz.
    """
    digest = hashlib.sha256((Path(splits_dir) / f"{name}.csv").read_bytes()).hexdigest()[:12]
    return cache_dir / f"{name}_{img_size}px_{digest}.npz"


def _read_rows(out: Path) -> list[dict]:
    return json.loads(out.read_text(encoding="utf-8")) if out.is_file() else []


def final_runs(out: Path, backbone: str) -> int:
    """Bu omurga için test kümesinin kaç kez ölçüldüğü."""
    row = next((r for r in _read_rows(out) if r["backbone"] == backbone), {})
    return int(row.get("final_runs", 0))


def guard_final(out: Path, backbone: str, rerun: bool) -> int:
    """`evaluate --final` ile aynı tek seferlik koruma. Önceki test ölçümü sayısını döner."""
    runs = final_runs(out, backbone)
    if runs and not rerun:
        raise SystemExit(
            f"{out} içinde {backbone} için test ölçümü zaten var: test kümesi bir kez ölçüldü. "
            "Bilerek yeniden ölçmek için --rerun-final verin; bu, raporda kayıt altına alınır."
        )
    return runs


def _cached(cache: Path, backbone, samples, img_size, batch_size) -> tuple[np.ndarray, np.ndarray]:
    if cache.is_file():
        with np.load(cache) as data:
            return data["x"], data["y"]
    x, y = embed(backbone, samples, img_size, batch_size)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, y=y)
    return x, y


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Embedding + LogisticRegression tabanı")
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--cache", default="cache/embeddings")
    parser.add_argument("--out", default="reports/probes.json")
    parser.add_argument("--final", action="store_true", help="Test AUROC'u bir kez ölç")
    parser.add_argument("--rerun-final", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if args.final:
        previous_runs = guard_final(out, args.backbone, args.rerun_final)
    else:
        previous_runs = final_runs(out, args.backbone)
    cfg = load_config(args.config)
    backbone = create_backbone(args.backbone, cfg.img_size)
    cache = Path(args.cache) / args.backbone.replace("/", "_")
    split = {
        name: _cached(
            cache_path(cache, cfg.splits_dir, name, cfg.img_size),
            backbone,
            read_split(cfg.splits_dir, name, cfg.data_root, allow_test=args.final),
            cfg.img_size,
            cfg.batch_size,
        )
        for name in (("train", "val", "test") if args.final else ("train", "val"))
    }
    probe, c, val_auroc = fit_probe(*split["train"], *split["val"])
    row = {"backbone": args.backbone, "C": c, "val_auroc": val_auroc}
    if args.final:
        test_x, test_y = split["test"]
        row["test_auroc"] = float(roc_auc_score(test_y, probe.predict_proba(test_x)[:, 1]))
        previous_runs += 1
    if previous_runs:
        row["final_runs"] = previous_runs  # val koşusu test kaydını silmez

    rows = [r for r in _read_rows(out) if r["backbone"] != args.backbone] + [row]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(row)


if __name__ == "__main__":
    main()
