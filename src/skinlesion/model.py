"""timm modeli, toplu çıkarım, checkpoint ve ONNX dışa aktarımı."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader

from skinlesion.errors import CheckpointNotFoundError

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
OOD_FILE = "ood.npz"
ONNX_FILE = "model.onnx"
METRICS_FILE = "metrics.json"


def create_model(name: str, *, pretrained: bool) -> nn.Module:
    """Tek logit çıktılı (malign) timm modeli."""
    return timm.create_model(name, pretrained=pretrained, num_classes=1)


def normalization(model: nn.Module) -> tuple[list[float], list[float]]:
    """Omurganın ön eğitimde kullandığı mean/std değeri (timm data config)."""
    cfg = timm.data.resolve_data_config({}, model=model)
    return list(cfg["mean"]), list(cfg["std"])


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Aşama 1'de yalnız sınıflandırma başı, aşama 2'de bütün ağ eğitilir."""
    for param in model.parameters():
        param.requires_grad = trainable
    for param in model.get_classifier().parameters():
        param.requires_grad = True


def features_and_logits(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Havuzlanmış özellik (OOD için) ve logit."""
    feats = model.forward_head(model.forward_features(x), pre_logits=True)
    return feats, model.get_classifier()(feats).squeeze(-1)


@torch.inference_mode()
def predict_logits(
    model: nn.Module, loader: DataLoader, device: torch.device, *, with_features: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Toplu çıkarım. Görüntü başına ayrı çağrı yapılmaz."""
    model.eval()
    logits, labels, feats = [], [], []
    for x, y in loader:
        f, z = features_and_logits(model, x.to(device))
        logits.append(z.float().cpu())
        labels.append(y)
        if with_features:
            feats.append(f.float().cpu())
    feat_arr = torch.cat(feats).numpy() if with_features else None
    return torch.cat(logits).numpy(), torch.cat(labels).numpy(), feat_arr


@dataclass
class CheckpointConfig:
    """Checkpoint ile gelen çıkarım ayarları. `config.json` olarak saklanır."""

    model_name: str
    img_size: int
    mean: list[float]
    std: list[float]
    threshold: float = 0.5
    temperature: float = 1.0
    target_sensitivity: float = 0.90
    band_low: float = 0.3
    band_high: float = 0.7
    info: dict[str, Any] = field(default_factory=dict)


def save_checkpoint(model: nn.Module, cfg: CheckpointConfig, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, out_dir / WEIGHTS_FILE)
    write_checkpoint_config(cfg, out_dir)
    return out_dir


def write_checkpoint_config(cfg: CheckpointConfig, out_dir: str | Path) -> None:
    path = Path(out_dir) / CONFIG_FILE
    path.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")


def read_checkpoint_config(ckpt_dir: str | Path) -> CheckpointConfig:
    path = Path(ckpt_dir) / CONFIG_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CheckpointNotFoundError(f"Checkpoint config'i yok: {path}") from exc
    return CheckpointConfig(**raw)


def load_checkpoint(ckpt_dir: str | Path) -> tuple[nn.Module, CheckpointConfig]:
    """Ağırlıkları ağ indirmeden yükler (pretrained=False)."""
    ckpt_dir = Path(ckpt_dir)
    cfg = read_checkpoint_config(ckpt_dir)
    weights = ckpt_dir / WEIGHTS_FILE
    if not weights.is_file():
        raise CheckpointNotFoundError(f"Ağırlık dosyası yok: {weights}")
    model = create_model(cfg.model_name, pretrained=False)
    model.load_state_dict(load_file(weights))
    return model.eval(), cfg


class _OnnxWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats, logits = features_and_logits(self.model, x)
        return logits, feats


def export_onnx(model: nn.Module, img_size: int, out_path: str | Path) -> Path:
    """İki çıktılı ONNX: `logit` ve OOD için `features`. Yığın boyutu dinamik."""
    out_path = Path(out_path)
    dummy = torch.randn(2, 3, img_size, img_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            _OnnxWrapper(
                model
            ).eval(),  # sarmalayıcı da eval olmalı; yoksa BN train modunda izlenir
            (dummy,),
            str(out_path),
            input_names=["image"],
            output_names=["logit", "features"],
            dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}, "features": {0: "batch"}},
            dynamo=False,
        )
    return out_path


def export_checkpoint_onnx(ckpt_dir: str | Path) -> Path:
    """Checkpoint klasörüne `model.onnx` yazar; demo `SKINLESION_BACKEND=onnx` ile okur."""
    model, cfg = load_checkpoint(ckpt_dir)
    return export_onnx(model, cfg.img_size, Path(ckpt_dir) / ONNX_FILE)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Checkpoint'i ONNX'e aktarır")
    parser.add_argument("--export-onnx", metavar="CKPT_DIR", required=True)
    args = parser.parse_args(argv)
    print(f"ONNX: {export_checkpoint_onnx(args.export_onnx)}")


if __name__ == "__main__":
    main()
