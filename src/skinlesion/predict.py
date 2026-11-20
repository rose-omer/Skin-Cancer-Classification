"""Çıkarım modülü. Demo ve testler yalnız bu modülü çağırır.

Checkpoint kaynağı (öncelik sırasıyla):
1. `MODEL_REPO` (+ isteğe bağlı `MODEL_REVISION`): HF Hub model reposundan indirilir.
2. `SKINLESION_CHECKPOINT` ya da `Predictor(ckpt_dir=...)`: yerel klasör.
3. Varsayılan: `checkpoints/effv2s_fanconic`.

Arka uç: `SKINLESION_BACKEND=torch` (varsayılan, Grad-CAM var) ya da `onnx`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Self, TypedDict

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import v2

from skinlesion.calibrate import apply_temperature
from skinlesion.data import build_transforms
from skinlesion.errors import CheckpointNotFoundError, InvalidImageError
from skinlesion.explain import grad_cam
from skinlesion.model import (
    CONFIG_FILE,
    METRICS_FILE,
    ONNX_FILE,
    OOD_FILE,
    WEIGHTS_FILE,
    CheckpointConfig,
    features_and_logits,
    load_checkpoint,
    read_checkpoint_config,
)
from skinlesion.ood import OodStats

DEFAULT_CHECKPOINT = Path("checkpoints/effv2s_fanconic")
MIN_SIDE = 32

Band = Literal["dusuk", "belirsiz", "yuksek"]
Backend = Literal["torch", "onnx"]


class Prediction(TypedDict):
    prob: float  # kalibre malign olasılığı
    prob_raw: float  # kalibrasyon öncesi
    band: Band
    threshold: float
    above_threshold: bool
    cam: np.ndarray | None  # (H, W), [0, 1]
    ood: bool | None
    ood_distance: float | None


def band_limits(threshold: float, band_low: float, band_high: float) -> tuple[float, float]:
    """Belirsiz bandın sınırları. Bant eşiği her zaman kapsar.

    Böylece "düşük" her zaman eşiğin altında, "yüksek" her zaman eşiğin üstünde kalır.
    Eşik %18 ve yapılandırma 0,30–0,70 ise bant 0,18–0,70 olur.
    """
    return min(band_low, threshold), max(band_high, threshold)


def band_for(prob: float, threshold: float, band_low: float, band_high: float) -> Band:
    low, high = band_limits(threshold, band_low, band_high)
    if prob < low:
        return "dusuk"
    if prob > high:
        return "yuksek"
    return "belirsiz"


def resolve_checkpoint_dir(ckpt_dir: str | Path | None = None) -> Path:
    repo = os.environ.get("MODEL_REPO")
    if repo:
        from huggingface_hub import snapshot_download

        try:
            local = snapshot_download(
                repo_id=repo,
                revision=os.environ.get("MODEL_REVISION"),
                allow_patterns=[CONFIG_FILE, WEIGHTS_FILE, OOD_FILE, ONNX_FILE, METRICS_FILE],
            )
        except Exception as exc:  # ağ, yetki ya da repo hatası: kullanıcıya tek tür hata
            raise CheckpointNotFoundError(f"HF Hub reposu indirilemedi: {repo}") from exc
        return Path(local)
    return Path(ckpt_dir or os.environ.get("SKINLESION_CHECKPOINT") or DEFAULT_CHECKPOINT)


def to_rgb_image(image: object) -> Image.Image:
    """PIL, numpy dizisi ya da dosya yolunu doğrulanmış RGB görüntüye çevirir."""
    if image is None:
        raise InvalidImageError("Görüntü verilmedi.")
    try:
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, str | Path):
            with Image.open(image) as opened:
                img = opened.copy()
        else:
            raise InvalidImageError(f"Desteklenmeyen girdi türü: {type(image).__name__}")
    except (UnidentifiedImageError, OSError, TypeError, ValueError) as exc:
        raise InvalidImageError("Dosya bir görüntü olarak okunamadı.") from exc
    if min(img.size) < MIN_SIDE:
        raise InvalidImageError(f"Görüntü çok küçük: en az {MIN_SIDE} piksel olmalı.")
    return img.convert("RGB")


class Predictor:
    def __init__(self, ckpt_dir: str | Path | None = None, backend: Backend | None = None) -> None:
        self.ckpt_dir = resolve_checkpoint_dir(ckpt_dir)
        self.backend: Backend = backend or os.environ.get("SKINLESION_BACKEND", "torch")  # type: ignore[assignment]
        if self.backend not in ("torch", "onnx"):
            raise ValueError(f"Bilinmeyen arka uç: {self.backend}")
        self.config: CheckpointConfig | None = None
        self.model: torch.nn.Module | None = None
        self.session = None
        self.ood: OodStats | None = None
        self.metrics: dict | None = None

    def load(self) -> Self:
        if self.backend == "torch":
            self.model, self.config = load_checkpoint(self.ckpt_dir)
        else:
            self.config = read_checkpoint_config(self.ckpt_dir)
            onnx_path = self.ckpt_dir / ONNX_FILE
            if not onnx_path.is_file():
                raise CheckpointNotFoundError(f"ONNX dosyası yok: {onnx_path}")
            import onnxruntime as ort

            self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if (self.ckpt_dir / OOD_FILE).is_file():
            self.ood = OodStats.load(self.ckpt_dir / OOD_FILE)
        metrics_path = self.ckpt_dir / METRICS_FILE
        if metrics_path.is_file():
            self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        size = self.config.img_size
        self._transform = build_transforms(size, self.config.mean, self.config.std, train=False)
        self._view = v2.Compose([v2.Resize(size), v2.CenterCrop(size)])
        return self

    def view(self, image: object) -> Image.Image:
        """Modelin gördüğü kırpım. Grad-CAM bu görüntünün üstüne bindirilir."""
        self._require_config()
        return self._view(to_rgb_image(image))

    @property
    def supports_cam(self) -> bool:
        return self.model is not None

    def band(self, prob: float) -> Band:
        cfg = self._require_config()
        return band_for(prob, cfg.threshold, cfg.band_low, cfg.band_high)

    def predict(self, image: object, *, with_cam: bool = True) -> Prediction:
        cfg = self._require_config()
        img = to_rgb_image(image)
        x = self._transform(img).unsqueeze(0)
        logit, feats = self._forward(x)
        prob = float(apply_temperature(np.array([logit]), cfg.temperature)[0])
        cam = grad_cam(self.model, x) if with_cam and self.model is not None else None
        ood_distance = float(self.ood.distance(feats)[0]) if self.ood is not None else None
        return Prediction(
            prob=prob,
            prob_raw=float(apply_temperature(np.array([logit]), 1.0)[0]),
            band=self.band(prob),
            threshold=cfg.threshold,
            above_threshold=prob >= cfg.threshold,
            cam=cam,
            ood=None if ood_distance is None else ood_distance > self.ood.threshold,
            ood_distance=ood_distance,
        )

    def _forward(self, x: torch.Tensor) -> tuple[float, np.ndarray]:
        if self.session is not None:
            logit, feats = self.session.run(None, {"image": x.numpy()})
            return float(logit[0]), feats
        with torch.inference_mode():
            feats, logit = features_and_logits(self.model, x)
        return float(logit[0]), feats.numpy()

    def _require_config(self) -> CheckpointConfig:
        if self.config is None:
            raise RuntimeError("Önce `load()` çağrılmalı.")
        return self.config
