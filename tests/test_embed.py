import json

import numpy as np
import pytest

from skinlesion.data import list_images
from skinlesion.embed import cache_path, create_backbone, embed, fit_probe, guard_final
from tests.conftest import TINY_MODEL


def test_cache_key_follows_split_file(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train.csv").write_text("path,label\na.png,0\n", encoding="utf-8")
    first = cache_path(tmp_path / "cache", splits, "train", 224)
    (splits / "train.csv").write_text("path,label\nb.png,1\n", encoding="utf-8")
    second = cache_path(tmp_path / "cache", splits, "train", 224)
    assert first != second  # yeniden üretilen bölünme eski önbelleği kullanmaz
    assert cache_path(tmp_path / "cache", splits, "train", 448) != second


def test_final_guard_allows_once(tmp_path):
    out = tmp_path / "probes.json"
    assert guard_final(out, "dino", rerun=False) == 0
    out.write_text(json.dumps([{"backbone": "dino", "val_auroc": 0.8}]), encoding="utf-8")
    assert guard_final(out, "dino", rerun=False) == 0  # yalnız val koşusu: test henüz açılmadı
    out.write_text(
        json.dumps([{"backbone": "dino", "test_auroc": 0.8, "final_runs": 1}]), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="bir kez"):
        guard_final(out, "dino", rerun=False)
    assert guard_final(out, "dino", rerun=True) == 1
    assert guard_final(out, "baska", rerun=False) == 0


def test_fit_probe_separates_linear_features():
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.4).astype(int)
    x = rng.normal(size=(200, 8)) + y[:, None] * 2.0
    _, c, val_auroc = fit_probe(x[:150], y[:150], x[150:], y[150:])
    assert val_auroc > 0.95
    assert c > 0


def test_embed_returns_one_vector_per_image(fake_data):
    samples = list_images(fake_data / "test")
    backbone = create_backbone(TINY_MODEL, 48, pretrained=False)
    feats, labels = embed(backbone, samples, img_size=48, batch_size=4)
    assert feats.shape[0] == len(samples) == labels.shape[0]
    assert feats.ndim == 2
