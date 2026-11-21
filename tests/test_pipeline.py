"""Uçtan uca duman testi: sahte görüntülerle train -> evaluate --final -> Predictor.

Model rastgele başlatılır ve birkaç adım koşar; amaç kodun çalıştığını göstermektir.
"""

import json

import pytest
import yaml

from skinlesion.config import load_config
from skinlesion.errors import ConfigError
from skinlesion.evaluate import render_table, run_final
from skinlesion.model import CONFIG_FILE, OOD_FILE, WEIGHTS_FILE
from skinlesion.predict import Predictor
from skinlesion.train import fit
from tests.conftest import TINY_MODEL, fake_image


@pytest.fixture
def smoke_config(fake_data, tmp_path):
    cfg = {
        "experiment_id": "smoke",
        "data_root": str(fake_data),
        "splits_dir": str(tmp_path / "splits"),
        "output_dir": str(tmp_path / "runs"),
        "model_name": TINY_MODEL,
        "pretrained": False,
        "img_size": 48,
        "batch_size": 8,
        "head_epochs": 1,
        "finetune_epochs": 1,
        "patience": 1,
        "max_train_batches": 2,
    }
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("lerning_rate: 0.1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="lerning_rate"):
        load_config(path)


def test_train_evaluate_predict_end_to_end(smoke_config, tmp_path):
    run_dir = fit(load_config(smoke_config))
    for name in (WEIGHTS_FILE, CONFIG_FILE, OOD_FILE, "history.json", "smoke_curves.png"):
        assert (run_dir / name).is_file(), name
    ckpt = json.loads((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    assert ckpt["temperature"] > 0
    assert 0 <= ckpt["threshold"] <= 1
    assert ckpt["info"]["val"]["sensitivity"] >= 0.9

    reports = tmp_path / "reports"
    metrics = run_final(str(smoke_config), None, reports, n_boot=20, rerun=False)
    for key in ("threshold", "sensitivity", "specificity"):
        assert key in metrics["test"]["at_selected"]
    assert metrics["final_runs"] == 1
    for fig in ("roc_pr.png", "reliability.png", "confusion.png", "errors.png", "gradcam.png"):
        assert (reports / fig).is_file(), fig
    assert "Xception" in render_table(metrics)

    with pytest.raises(SystemExit, match="bir kez"):
        run_final(str(smoke_config), None, reports, n_boot=20, rerun=False)

    predictor = Predictor(run_dir).load()
    assert predictor.metrics is not None
    out = predictor.predict(fake_image(0, 1, size=60))
    assert out["ood"] in (True, False)
