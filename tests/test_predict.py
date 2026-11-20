import numpy as np
import pytest

from skinlesion.errors import CheckpointNotFoundError, InvalidImageError
from skinlesion.model import ONNX_FILE, export_onnx, load_checkpoint
from skinlesion.predict import Predictor, band_for, band_limits, resolve_checkpoint_dir
from tests.conftest import TINY_SIZE, fake_image

SCHEMA = {
    "prob": float,
    "prob_raw": float,
    "band": str,
    "threshold": float,
    "above_threshold": bool,
    "ood": bool,
    "ood_distance": float,
}


@pytest.fixture(scope="module")
def predictor(tiny_checkpoint):
    return Predictor(tiny_checkpoint, backend="torch").load()


def test_predict_returns_schema_dict(predictor):
    out = predictor.predict(fake_image(0, 1, size=80))
    for key, kind in SCHEMA.items():
        assert isinstance(out[key], kind), key
    assert 0.0 <= out["prob"] <= 1.0
    assert out["band"] in {"dusuk", "belirsiz", "yuksek"}
    assert out["threshold"] == pytest.approx(0.42)
    assert out["above_threshold"] == (out["prob"] >= 0.42)
    assert out["cam"].shape == (TINY_SIZE, TINY_SIZE)


def test_temperature_is_applied(predictor):
    out = predictor.predict(fake_image(1, 0, size=80), with_cam=False)
    logit = np.log(out["prob_raw"] / (1 - out["prob_raw"]))
    assert out["prob"] == pytest.approx(1 / (1 + np.exp(-logit / 1.5)), abs=1e-6)
    assert out["cam"] is None


@pytest.mark.parametrize(
    ("prob", "band"), [(0.1, "dusuk"), (0.3, "belirsiz"), (0.55, "belirsiz"), (0.71, "yuksek")]
)
def test_band_edges(predictor, prob, band):
    assert predictor.band(prob) == band


@pytest.mark.parametrize("threshold", [0.05, 0.18, 0.3, 0.5, 0.7, 0.85])
def test_band_never_contradicts_threshold(threshold):
    """'Düşük' hep eşiğin altında, 'yüksek' hep eşiğin üstünde kalır."""
    for prob in np.linspace(0.0, 1.0, 201):
        band = band_for(float(prob), threshold, 0.3, 0.7)
        if band == "dusuk":
            assert prob < threshold
        if band == "yuksek":
            assert prob >= threshold


def test_band_limits_cover_threshold():
    assert band_limits(0.18, 0.3, 0.7) == (0.18, 0.7)
    assert band_limits(0.5, 0.3, 0.7) == (0.3, 0.7)
    assert band_limits(0.8, 0.3, 0.7) == (0.3, 0.8)
    # İnceleme örneği: eşik %18, olasılık %25 artık 'düşük' değil.
    assert band_for(0.25, 0.18, 0.3, 0.7) == "belirsiz"


def test_accepts_numpy_and_path(predictor, tmp_path):
    img = fake_image(2, 0, size=80)
    path = tmp_path / "x.png"
    img.save(path)
    a = predictor.predict(np.asarray(img), with_cam=False)["prob"]
    b = predictor.predict(str(path), with_cam=False)["prob"]
    assert a == pytest.approx(b, abs=1e-5)


@pytest.mark.parametrize("bad", [None, 12, "yok.png"])
def test_invalid_inputs_raise_invalid_image(predictor, bad):
    with pytest.raises(InvalidImageError):
        predictor.predict(bad)


def test_non_image_file_raises_invalid_image(predictor, tmp_path):
    path = tmp_path / "not_image.jpg"
    path.write_text("bu bir görüntü değil", encoding="utf-8")
    with pytest.raises(InvalidImageError):
        predictor.predict(str(path))


def test_tiny_image_is_rejected(predictor):
    with pytest.raises(InvalidImageError, match="çok küçük"):
        predictor.predict(fake_image(3, 0, size=16))


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(CheckpointNotFoundError):
        Predictor(tmp_path / "yok").load()


def test_env_checkpoint_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("MODEL_REPO", raising=False)
    monkeypatch.setenv("SKINLESION_CHECKPOINT", str(tmp_path))
    assert resolve_checkpoint_dir() == tmp_path


def test_hub_failure_becomes_checkpoint_error(monkeypatch):
    import huggingface_hub

    def fail(**_kwargs):
        raise OSError("ağ yok")

    monkeypatch.setenv("MODEL_REPO", "kullanici/yok")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    with pytest.raises(CheckpointNotFoundError):
        resolve_checkpoint_dir()


def test_onnx_backend_matches_torch(tiny_checkpoint, predictor):
    pytest.importorskip("onnxruntime")
    model, cfg = load_checkpoint(tiny_checkpoint)
    export_onnx(model, cfg.img_size, tiny_checkpoint / ONNX_FILE)
    onnx_predictor = Predictor(tiny_checkpoint, backend="onnx").load()
    img = fake_image(4, 1, size=80)
    a = predictor.predict(img, with_cam=False)
    b = onnx_predictor.predict(img)
    assert b["prob"] == pytest.approx(a["prob"], abs=1e-4)
    assert b["ood_distance"] == pytest.approx(a["ood_distance"], rel=1e-3)
    assert b["cam"] is None and not onnx_predictor.supports_cam
