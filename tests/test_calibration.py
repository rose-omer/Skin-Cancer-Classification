import numpy as np
import pytest

from skinlesion.calibrate import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    sigmoid,
)


def test_temperature_one_keeps_probabilities():
    logits = np.array([-3.0, -0.5, 0.0, 0.7, 4.0])
    np.testing.assert_allclose(apply_temperature(logits, 1.0), sigmoid(logits))


def test_non_positive_temperature_is_rejected():
    with pytest.raises(ValueError):
        apply_temperature(np.zeros(2), 0.0)


def test_ece_on_known_example():
    # Kova 1 (0.2): 2 örnek, gözlenen 0.5 -> |0.2-0.5| = 0.3
    # Kova 2 (0.9): 2 örnek, gözlenen 1.0 -> |0.9-1.0| = 0.1
    probs = np.array([0.2, 0.2, 0.9, 0.9])
    labels = np.array([0, 1, 1, 1])
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.2)


def test_perfectly_calibrated_bins_have_zero_ece():
    probs = np.array([0.25] * 4 + [0.75] * 4)
    labels = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert expected_calibration_error(probs, labels, n_bins=4) == pytest.approx(0.0)


def test_fit_temperature_recovers_overconfidence_and_lowers_ece():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 1.5, 4000)
    labels = (rng.random(4000) < sigmoid(true_logits)).astype(int)
    overconfident = true_logits * 3.0  # gerçek T = 3

    temperature = fit_temperature(overconfident, labels)

    assert temperature == pytest.approx(3.0, rel=0.15)
    before = expected_calibration_error(apply_temperature(overconfident, 1.0), labels)
    after = expected_calibration_error(apply_temperature(overconfident, temperature), labels)
    assert after < before
