"""Grad-CAM: son evrişim aşamasının çıktısını hedefler.

Isı haritası modelin karar verirken ağırlık verdiği bölgeyi gösterir. Tanı kanıtı değildir.
timm CNN'lerinde `forward_features` son aşamanın NCHW çıktısını verir; ek kanca gerekmez.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402, N812
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402

DEFAULT_ALPHA = 0.4
COLORMAP = "inferno"  # algısal olarak düzgün; renk körlüğünde jet'ten iyi okunur


def grad_cam(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Tek görüntü için (H, W) boyutlu, [0, 1] aralığında ısı haritası."""
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"Girdi (1, 3, H, W) olmalı: {tuple(x.shape)}")
    model.eval()
    with torch.enable_grad():
        feats = model.forward_features(x)
        if feats.ndim != 4:
            raise ValueError("Grad-CAM yalnız evrişimli (NCHW) omurgalarda çalışır.")
        logit = model.forward_head(feats).squeeze()
        # Yalnız özellik haritasının gradyanı alınır; parametre .grad hiç yazılmaz.
        # Böylece eşzamanlı istekler paylaşılan modelde birbirini bozmaz.
        (grad,) = torch.autograd.grad(logit, feats)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    span = cam.max() - cam.min()
    return (cam - cam.min()) / span if span > 1e-12 else np.zeros_like(cam)


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = DEFAULT_ALPHA) -> Image.Image:
    """Isı haritasını görüntünün üstüne `alpha` saydamlıkla bindirir."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha [0, 1] aralığında olmalı: {alpha}")
    base = image.convert("RGB")
    heat = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8)).resize(base.size)
    rgba = matplotlib.colormaps[COLORMAP](np.asarray(heat) / 255.0)
    heat_rgb = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
    return Image.blend(base, heat_rgb, alpha)


def save_gallery(
    images: Sequence[Image.Image], titles: Sequence[str], out: str | Path, cols: int = 4
) -> Path:
    """Rapor için ızgara: Grad-CAM galerisi ve hata galerisi aynı fonksiyonu kullanır."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = max(1, -(-len(images) // cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, title in zip(axes.flat, images, titles, strict=False):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
