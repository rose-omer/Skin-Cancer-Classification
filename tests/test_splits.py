import numpy as np
import pytest

from skinlesion.data import (
    grouped_val_split,
    make_splits,
    read_duplicates,
    read_split,
    stratified_val_split,
)
from skinlesion.errors import TestSetAccessError


def test_splits_have_no_common_files(fake_data, tmp_path):
    counts = make_splits(fake_data, tmp_path / "splits", val_fraction=0.15, seed=42)
    sets = {
        name: {p for p, _ in read_split(tmp_path / "splits", name, fake_data, allow_test=True)}
        for name in ("train", "val", "test")
    }
    assert counts == {name: len(s) for name, s in sets.items()}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]


def test_duplicates_are_removed_from_train_and_val(fake_data, tmp_path):
    dup = fake_data / "train" / "benign" / "benign_000.png"
    report = tmp_path / "dedup.csv"
    report.write_text(f"train_path,test_path,hamming,exact\n{dup},x.png,0,1\n", encoding="utf-8")
    counts = make_splits(fake_data, tmp_path / "splits", 0.15, 42, read_duplicates(report))
    kept = {
        p.resolve()
        for name in ("train", "val")
        for p, _ in read_split(tmp_path / "splits", name, fake_data)
    }
    assert dup.resolve() not in kept
    assert counts["train"] + counts["val"] == 39  # 40 train görüntüsünden biri çıktı


def test_val_split_keeps_class_ratio_and_is_seeded():
    labels = [0] * 80 + [1] * 20
    _, val_a = stratified_val_split(labels, 0.15, seed=7)
    _, val_b = stratified_val_split(labels, 0.15, seed=7)
    assert np.array_equal(val_a, val_b)
    assert np.asarray(labels)[val_a].mean() == pytest.approx(0.2, abs=0.05)


def test_test_split_requires_explicit_access(fake_data, tmp_path):
    make_splits(fake_data, tmp_path / "splits", val_fraction=0.15, seed=0)
    with pytest.raises(TestSetAccessError):
        read_split(tmp_path / "splits", "test", fake_data)


def test_grouped_split_has_no_common_patient():
    rng = np.random.default_rng(0)
    groups = [f"p{i // 3}" for i in range(120)]
    labels = [int(rng.random() < 0.3) for _ in groups]
    train_idx, val_idx = grouped_val_split(labels, groups, n_splits=5, seed=0)
    train_groups = {groups[i] for i in train_idx}
    val_groups = {groups[i] for i in val_idx}
    assert not train_groups & val_groups
    assert len(train_idx) + len(val_idx) == len(groups)
