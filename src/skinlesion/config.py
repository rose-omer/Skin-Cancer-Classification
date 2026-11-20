"""YAML deney config'i."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from skinlesion.errors import ConfigError


@dataclass
class TrainConfig:
    experiment_id: str = "effv2s_fanconic"
    data_root: str = "data"
    splits_dir: str = "splits"
    output_dir: str = "runs"
    model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k"
    pretrained: bool = True
    img_size: int = 224
    val_fraction: float = 0.15
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 0
    head_epochs: int = 3
    finetune_epochs: int = 20
    head_lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 0.05
    patience: int = 5
    target_sensitivity: float = 0.90
    band_low: float = 0.3
    band_high: float = 0.7
    ood_quantile: float = 0.99
    max_train_batches: int | None = None  # yalnız duman testi için

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 0.5:
            raise ConfigError(f"val_fraction 0 ile 0.5 arasında olmalı: {self.val_fraction}")
        if not 0.0 < self.target_sensitivity <= 1.0:
            raise ConfigError(
                f"target_sensitivity (0, 1] aralığında olmalı: {self.target_sensitivity}"
            )
        if not 0.0 <= self.band_low < self.band_high <= 1.0:
            raise ConfigError("band_low < band_high olmalı ve ikisi de [0, 1] aralığında olmalı.")

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path, **overrides: Any) -> TrainConfig:
    """YAML dosyasını okur. Bilinmeyen anahtar hata verir; yazım hatası sessiz geçmez."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Config dosyası yok: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config YAML olarak okunamadı: {path}") from exc
    raw.update(overrides)
    known = {f.name for f in fields(TrainConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"Bilinmeyen config anahtarı: {', '.join(unknown)}")
    return TrainConfig(**raw)
