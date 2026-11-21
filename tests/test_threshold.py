import numpy as np
import pytest

from skinlesion.metrics import (
    bootstrap_ci,
    partial_auc,
    ppv_at_prevalence,
    select_threshold,
    sensitivity,
    specificity,
    summarize,
)


def test_threshold_is_highest_value_meeting_target_sensitivity():
    pos = np.linspace(0.05, 0.95, 10)  # 10 malign
    neg = np.linspace(0.0, 0.5, 10)
    probs = np.concatenate([pos, neg])
    labels = np.array([1] * 10 + [0] * 10)

    threshold = select_threshold(probs, labels, 0.90)

    assert threshold == pytest.approx(pos[1])  # 9/10 pozitif eşikte ya da üstünde
    assert sensitivity(probs, labels, threshold) >= 0.90
    assert sensitivity(probs, labels, threshold + 1e-6) < 0.90


@pytest.mark.parametrize("target", [0.5, 0.8, 0.95, 1.0])
def test_threshold_meets_target_on_random_scores(target):
    rng = np.random.default_rng(1)
    labels = (rng.random(500) < 0.3).astype(int)
    probs = np.clip(rng.normal(0.35 + 0.3 * labels, 0.15), 0, 1)
    threshold = select_threshold(probs, labels, target)
    assert sensitivity(probs, labels, threshold) >= target


def test_threshold_needs_a_positive():
    with pytest.raises(ValueError):
        select_threshold(np.array([0.1, 0.2]), np.array([0, 0]), 0.9)


def test_partial_auc_bounds():
    labels = np.array([0, 0, 1, 1])
    assert partial_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(0.2)
    assert partial_auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0, abs=1e-9)


def test_ppv_at_prevalence_matches_plan_example():
    # Plan §2 madde 8: duyarlılık 0.763, özgüllük 0.864, yaygınlık %1.8 -> PPV ~ %9
    assert ppv_at_prevalence(0.763, 0.864, 0.018) == pytest.approx(0.093, abs=0.001)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(2)
    labels = (rng.random(300) < 0.4).astype(int)
    probs = np.clip(rng.normal(0.4 + 0.2 * labels, 0.2), 0, 1)
    low, high = bootstrap_ci(lambda p, y: specificity(p, y, 0.5), probs, labels, n_boot=200)
    assert low <= specificity(probs, labels, 0.5) <= high


def test_summarize_has_required_fields():
    rng = np.random.default_rng(3)
    labels = (rng.random(120) < 0.5).astype(int)
    probs = np.clip(rng.normal(0.4 + 0.2 * labels, 0.2), 0, 1)
    out = summarize(probs, labels, 0.5, n_boot=50)
    for key in ("threshold", "sensitivity", "specificity", "auroc", "auprc", "pauc_tpr80"):
        assert key in out
    assert len(out["sensitivity"]["ci95"]) == 2
    assert sum(out["confusion"].values()) == 120
