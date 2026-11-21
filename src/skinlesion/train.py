"""İki aşamalı ince ayar, temperature scaling, eşik seçimi ve OOD istatistiği.

Kullanım:
    python -m skinlesion.train --config configs/effv2s_fanconic.yaml

Test kümesine dokunmaz. Test yalnız `python -m skinlesion.evaluate --final` ile ölçülür.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from skinlesion.calibrate import apply_temperature, expected_calibration_error, fit_temperature
from skinlesion.config import TrainConfig, load_config
from skinlesion.data import ImageDataset, build_transforms, make_splits, read_split
from skinlesion.metrics import select_threshold, sensitivity, specificity
from skinlesion.model import (
    OOD_FILE,
    CheckpointConfig,
    create_model,
    normalization,
    predict_logits,
    save_checkpoint,
    set_backbone_trainable,
)
from skinlesion.ood import false_alarm_rate, fit_ood


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(dataset: ImageDataset, cfg: TrainConfig, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        generator=torch.Generator().manual_seed(cfg.seed),
    )


def _cosine_optimizer(
    params: list[nn.Parameter], lr: float, weight_decay: float, total_steps: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    max_batches: int | None,
) -> float:
    model.train()
    total, count = 0.0, 0
    for step, (x, y) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            loss = loss_fn(model(x).squeeze(-1), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += float(loss.item()) * len(y)
        count += len(y)
    return total / max(count, 1)


def _val_auroc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    logits, labels, _ = predict_logits(model, loader, device)
    return float(roc_auc_score(labels, logits))


def _save_curves(history: list[dict[str, float]], out: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(9, 3.2))
    ax_loss.plot(epochs, [h["train_loss"] for h in history])
    ax_loss.set(title="Train loss", xlabel="epoch")
    ax_auc.plot(epochs, [h["val_auroc"] for h in history])
    ax_auc.set(title="Val AUROC", xlabel="epoch")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fit(cfg: TrainConfig) -> Path:
    """Eğitir ve checkpoint klasörünü döner."""
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir = Path(cfg.splits_dir)
    if not (splits_dir / "train.csv").is_file():
        make_splits(cfg.data_root, splits_dir, cfg.val_fraction, cfg.seed)
    train_samples = read_split(splits_dir, "train", cfg.data_root)
    val_samples = read_split(splits_dir, "val", cfg.data_root)

    model = create_model(cfg.model_name, pretrained=cfg.pretrained).to(device)
    mean, std = normalization(model)
    train_tf = build_transforms(cfg.img_size, mean, std, train=True)
    eval_tf = build_transforms(cfg.img_size, mean, std, train=False)
    train_loader = _loader(ImageDataset(train_samples, train_tf), cfg, shuffle=True)
    val_loader = _loader(ImageDataset(val_samples, eval_tf), cfg, shuffle=False)

    labels = np.array([y for _, y in train_samples])
    n_pos = max(int(labels.sum()), 1)
    pos_weight = torch.tensor((len(labels) - n_pos) / n_pos, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    steps = len(train_loader) if cfg.max_train_batches is None else cfg.max_train_batches
    steps = min(steps, len(train_loader))

    history: list[dict[str, float]] = []
    best_auroc, best_state, stale = -math.inf, None, 0
    stages = [
        ("head", cfg.head_epochs, cfg.head_lr, False),
        ("full", cfg.finetune_epochs, cfg.finetune_lr, True),
    ]
    epoch = 0
    for stage, n_epochs, lr, full in stages:
        set_backbone_trainable(model, full)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer, scheduler = _cosine_optimizer(params, lr, cfg.weight_decay, n_epochs * steps)
        stale = 0  # erken durdurma sayacı her aşamada sıfırdan başlar
        for _ in range(n_epochs):
            epoch += 1
            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                loss_fn,
                device,
                scaler,
                cfg.max_train_batches,
            )
            auroc = _val_auroc(model, val_loader, device)
            history.append({"epoch": epoch, "stage": stage, "train_loss": loss, "val_auroc": auroc})
            print(f"[{stage}] epoch {epoch}: loss={loss:.4f} val_auroc={auroc:.4f}")
            if auroc > best_auroc:
                best_auroc, stale = auroc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if full and stale >= cfg.patience:
                print(f"Erken durdurma: {cfg.patience} epoch val AUROC iyileşmedi.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Kalibrasyon ve eşik yalnız val kümesinde seçilir.
    val_logits, val_labels, _ = predict_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    val_probs = apply_temperature(val_logits, temperature)
    threshold = select_threshold(val_probs, val_labels, cfg.target_sensitivity)

    # Kapsam dışı girdi: dağılım train özelliklerinden (artırmasız), eşik val uzaklıklarından.
    train_eval_loader = _loader(ImageDataset(train_samples, eval_tf), cfg, shuffle=False)
    _, _, train_feats = predict_logits(model, train_eval_loader, device, with_features=True)
    _, _, val_feats = predict_logits(model, val_loader, device, with_features=True)
    ood = fit_ood(train_feats, cfg.ood_quantile, calib_feats=val_feats)

    run_dir = cfg.run_dir
    val_report = {
        "auroc": float(roc_auc_score(val_labels, val_probs)),
        "ece_before": expected_calibration_error(apply_temperature(val_logits, 1.0), val_labels),
        "ece_after": expected_calibration_error(val_probs, val_labels),
        "sensitivity": sensitivity(val_probs, val_labels, threshold),
        "specificity": specificity(val_probs, val_labels, threshold),
        "n": int(len(val_labels)),
        # Eşik val'den seçildiği için val oranı ~1 - ood_quantile olur; train oranı karşılaştırma.
        "ood_threshold": ood.threshold,
        "ood_false_alarm_val": false_alarm_rate(ood, val_feats),
        "ood_false_alarm_train": false_alarm_rate(ood, train_feats),
    }
    ckpt_cfg = CheckpointConfig(
        model_name=cfg.model_name,
        img_size=cfg.img_size,
        mean=mean,
        std=std,
        threshold=threshold,
        temperature=temperature,
        target_sensitivity=cfg.target_sensitivity,
        band_low=cfg.band_low,
        band_high=cfg.band_high,
        info={
            "experiment_id": cfg.experiment_id,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "seed": cfg.seed,
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "timm": timm.__version__,
            },
            "train_config": cfg.to_dict(),
            "val": val_report,
        },
    )
    save_checkpoint(model.cpu(), ckpt_cfg, run_dir)
    ood.save(run_dir / OOD_FILE)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _save_curves(history, run_dir / f"{cfg.experiment_id}_curves.png")
    print(
        f"Checkpoint: {run_dir} | val AUROC={val_report['auroc']:.4f} "
        f"T={temperature:.3f} eşik={threshold:.3f}"
    )
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="EfficientNetV2-S ince ayar")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    fit(load_config(args.config))


if __name__ == "__main__":
    main()
