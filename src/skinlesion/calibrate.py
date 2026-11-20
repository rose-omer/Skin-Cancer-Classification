"""Temperature scaling (Guo vd., 2017) ve ECE.

Tek parametre T doğrulama kümesinde BCE en aza indirilerek bulunur: p = sigmoid(logit / T).
"""

from __future__ import annotations

import numpy as np
import torch


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"Sıcaklık pozitif olmalı: {temperature}")
    return sigmoid(np.asarray(logits, dtype=np.float64) / temperature)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Val logit'lerinde BCE'yi en aza indiren T > 0 değerini döner."""
    z = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # T = exp(log_t) > 0
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn(z / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def reliability_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eşit genişlikli kovalar: (ortalama olasılık, gözlenen malign oranı, örnek sayısı)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    sum_p = np.bincount(idx, weights=probs, minlength=n_bins)
    sum_y = np.bincount(idx, weights=labels, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = np.where(counts > 0, sum_p / counts, np.nan)
        freq = np.where(counts > 0, sum_y / counts, np.nan)
    return conf, freq, counts


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Malign olasılığı için ECE: Σ (n_b / N) · |ort(p) − ort(y)|."""
    conf, freq, counts = reliability_bins(probs, labels, n_bins)
    mask = counts > 0
    return float(np.sum(counts[mask] / counts.sum() * np.abs(conf[mask] - freq[mask])))
