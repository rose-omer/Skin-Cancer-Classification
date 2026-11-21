import numpy as np
import pytest

from skinlesion.ood import false_alarm_rate, fit_ood


def test_threshold_comes_from_calibration_features():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(400, 6))
    # Görülmüş train örnekleri merkeze yakın; val daha dağınık (gerçek ince ayardaki gibi).
    val = rng.normal(scale=1.5, size=(400, 6))
    on_train = fit_ood(train, 0.99)
    on_val = fit_ood(train, 0.99, calib_feats=val)
    assert on_val.threshold > on_train.threshold
    assert false_alarm_rate(on_val, val) == pytest.approx(0.01, abs=0.01)
    # Yalnız train'den seçilen eşik val'de çok daha sık uyarı verir.
    assert false_alarm_rate(on_train, val) > 0.1


def test_far_point_is_flagged():
    rng = np.random.default_rng(1)
    stats = fit_ood(rng.normal(size=(300, 4)), 0.99, calib_feats=rng.normal(size=(100, 4)))
    assert stats.is_out_of_distribution(np.full((1, 4), 20.0))[0]
