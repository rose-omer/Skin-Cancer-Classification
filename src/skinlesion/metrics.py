"""Tarama ölçütleri: eşik seçimi, duyarlılık/özgüllük, pAUC ve bootstrap güven aralığı."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ISIC_2020_PREVALENCE = 0.018


def select_threshold(probs: np.ndarray, labels: np.ndarray, target_sensitivity: float) -> float:
    """Duyarlılığı en az `target_sensitivity` olan en yüksek eşiği döner (karar: p ≥ eşik)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    positives = np.sort(probs[labels == 1])[::-1]
    if positives.size == 0:
        raise ValueError("Eşik seçimi için en az bir malign örnek gerekir.")
    needed = math.ceil(target_sensitivity * positives.size - 1e-9)
    return float(positives[max(needed, 1) - 1])


def confusion_at(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, int]:
    pred = np.asarray(probs) >= threshold
    labels = np.asarray(labels).astype(bool)
    return {
        "tp": int(np.sum(pred & labels)),
        "fp": int(np.sum(pred & ~labels)),
        "tn": int(np.sum(~pred & ~labels)),
        "fn": int(np.sum(~pred & labels)),
    }


def _safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def sensitivity(probs: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    c = confusion_at(probs, labels, threshold)
    return _safe_div(c["tp"], c["tp"] + c["fn"])


def specificity(probs: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    c = confusion_at(probs, labels, threshold)
    return _safe_div(c["tn"], c["tn"] + c["fp"])


def ppv_at_prevalence(sens: float, spec: float, prevalence: float) -> float:
    """Verilen yaygınlıkta PPV. Dengeli test kümesinin precision değeri gerçekte geçmez."""
    return _safe_div(sens * prevalence, sens * prevalence + (1 - spec) * (1 - prevalence))


def partial_auc(labels: np.ndarray, probs: np.ndarray, min_tpr: float = 0.80) -> float:
    """ISIC 2024 pAUC: ROC eğrisinin TPR ≥ min_tpr bölgesindeki alanı. Aralık [0, 1 − min_tpr]."""
    v_gt = np.abs(np.asarray(labels) - 1)
    v_pred = -np.asarray(probs, dtype=np.float64)
    max_fpr = 1.0 - min_tpr
    scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    return float(0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (scaled - 0.5))


Metric = Callable[[np.ndarray, np.ndarray], float]


def bootstrap_ci(
    metric: Metric,
    probs: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Yüzdelik bootstrap güven aralığı. Tek sınıflı örneklemler atlanır."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, labels.size, labels.size)
        if labels[idx].min() == labels[idx].max():
            continue
        values.append(metric(probs[idx], labels[idx]))
    low, high = np.nanpercentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def summarize(
    probs: np.ndarray, labels: np.ndarray, threshold: float, n_boot: int = 1000, seed: int = 0
) -> dict[str, object]:
    """Bir eşikte bütün ölçütleri %95 güven aralığıyla döner."""
    metrics: dict[str, Metric] = {
        "auroc": lambda p, y: float(roc_auc_score(y, p)),
        "auprc": lambda p, y: float(average_precision_score(y, p)),
        "pauc_tpr80": lambda p, y: partial_auc(y, p),
        "sensitivity": lambda p, y: sensitivity(p, y, threshold),
        "specificity": lambda p, y: specificity(p, y, threshold),
    }
    out: dict[str, object] = {"threshold": float(threshold), "n": int(len(labels))}
    for name, fn in metrics.items():
        low, high = bootstrap_ci(fn, probs, labels, n_boot=n_boot, seed=seed)
        out[name] = {"value": fn(np.asarray(probs), np.asarray(labels)), "ci95": [low, high]}
    out["confusion"] = confusion_at(probs, labels, threshold)
    out["ppv_at_isic2020_prevalence"] = ppv_at_prevalence(
        out["sensitivity"]["value"], out["specificity"]["value"], ISIC_2020_PREVALENCE
    )
    return out
