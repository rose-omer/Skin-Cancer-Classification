"""Kapsam dışı girdi denetimi: train özelliklerine Mahalanobis uzaklığı."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class OodStats:
    mean: np.ndarray
    precision: np.ndarray
    threshold: float

    def distance(self, feats: np.ndarray) -> np.ndarray:
        diff = np.atleast_2d(feats) - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self.precision, diff))

    def is_out_of_distribution(self, feats: np.ndarray) -> np.ndarray:
        return self.distance(feats) > self.threshold

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean, precision=self.precision, threshold=self.threshold)

    @classmethod
    def load(cls, path: str | Path) -> OodStats:
        with np.load(path) as data:
            return cls(data["mean"], data["precision"], float(data["threshold"]))


def fit_ood(
    train_feats: np.ndarray, quantile: float = 0.99, calib_feats: np.ndarray | None = None
) -> OodStats:
    """Ledoit-Wolf kovaryansı train özelliklerinden kurulur.

    Eşik `calib_feats` (val) uzaklıklarının `quantile` yüzdeliğidir. Model train görüntülerini
    görmüştür; train uzaklıkları küçük kalır ve yalnız onlardan seçilen eşik gerçek
    görüntülerde fazla uyarı verir. `calib_feats` verilmezse train kullanılır (yalnız testler).
    """
    lw = LedoitWolf().fit(train_feats)
    stats = OodStats(lw.location_, lw.precision_, threshold=0.0)
    calib = train_feats if calib_feats is None else calib_feats
    threshold = float(np.quantile(stats.distance(calib), quantile))
    return OodStats(stats.mean, stats.precision, threshold)


def false_alarm_rate(stats: OodStats, feats: np.ndarray) -> float:
    """Kapsam içi örneklerin uyarı alan oranı."""
    return float(np.mean(stats.is_out_of_distribution(feats)))
