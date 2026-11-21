"""Gradio arayüzü sunucu açılmadan kurulur."""

import os

import gradio as gr
import pytest

from app.app import (
    DELETE_CACHE,
    DISCLAIMER_EN,
    DISCLAIMER_TR,
    FONTS_DIR,
    PRIVACY,
    SERIF_FILES,
    LoadResult,
    about_html,
    build_demo,
    caption_html,
    empty_html,
    error_html,
    figure_html,
    gate_html,
    input_intro_html,
    loading_html,
    page_css,
    pct,
    result_html,
    theme,
    verdict_label,
)
from skinlesion.predict import Prediction, Predictor, band_for
from tests.conftest import fake_image


@pytest.fixture(scope="module")
def predictor(tiny_checkpoint):
    return Predictor(tiny_checkpoint).load()


def _components(demo: gr.Blocks) -> list:
    return list(demo.blocks.values())


def test_demo_builds_with_checkpoint(predictor, tiny_checkpoint):
    demo = build_demo(LoadResult(predictor, tiny_checkpoint))
    assert isinstance(demo, gr.Blocks)
    images = [c for c in _components(demo) if isinstance(c, gr.Image)]
    upload = next(c for c in images if c.label == "Dermoskopi görüntüsü")
    assert upload.interactive is False  # onay olmadan yükleme kapalı
    assert upload.visible is False  # yerinde kilit notu durur
    assert upload.buttons == []  # Gradio araç çubuğu yok
    assert upload.sources == ["upload"]  # kamera yok
    slider = next(c for c in _components(demo) if isinstance(c, gr.Slider))
    assert slider.buttons == []  # sıfırlama düğmesi yok
    button = next(c for c in _components(demo) if isinstance(c, gr.Button))
    assert button.interactive is False
    theme()  # tema nesnesi kurulabiliyor


def test_missing_checkpoint_shows_warning_screen(tmp_path):
    demo = build_demo(LoadResult(None, tmp_path / "yok", "Ağırlık dosyası yok"))
    html_blocks = [c.value for c in _components(demo) if isinstance(c, gr.HTML)]
    assert any("Model dosyası bulunamadı" in v for v in html_blocks)
    assert not any(isinstance(c, gr.Image) for c in _components(demo))


def test_result_html_always_carries_disclaimer_and_threshold(predictor):
    pred = predictor.predict(fake_image(0, 1, size=80), with_cam=False)
    out = result_html(pred, band_low=0.3, band_high=0.7, target=0.9)
    assert DISCLAIMER_TR in out
    assert "%90 duyarlılık" in out
    assert pct(pred["prob"]) in out


def test_ood_warning_appears_only_when_flagged(predictor):
    pred = predictor.predict(fake_image(1, 0, size=80), with_cam=False)
    pred["ood"] = True
    assert "dermoskopi görüntüsüne benzemiyor" in result_html(
        pred, band_low=0.3, band_high=0.7, target=0.9
    )
    pred["ood"] = False
    assert "benzemiyor" not in result_html(pred, band_low=0.3, band_high=0.7, target=0.9)


def _pred(prob: float, threshold: float) -> Prediction:
    return Prediction(
        prob=prob,
        prob_raw=prob,
        band=band_for(prob, threshold, 0.3, 0.7),
        threshold=threshold,
        above_threshold=prob >= threshold,
        cam=None,
        ood=False,
        ood_distance=0.0,
    )


@pytest.mark.parametrize(
    ("prob", "threshold", "label"),
    [
        (0.507, 0.18, "Belirsiz, eşiğin üstünde"),  # incelemede görülen durum
        (0.25, 0.18, "Belirsiz, eşiğin üstünde"),  # eskiden 'Düşük' ama eşiğin üstündeydi
        (0.10, 0.18, "Düşük, eşiğin altında"),
        (0.75, 0.18, "Yüksek, eşiğin üstünde"),
        (0.75, 0.80, "Belirsiz, eşiğin altında"),  # eskiden 'Yüksek' ama eşiğin altındaydı
    ],
)
def test_band_and_threshold_shown_as_one_consistent_label(prob, threshold, label):
    pred = _pred(prob, threshold)
    assert verdict_label(pred) == label
    out = result_html(pred, band_low=0.3, band_high=0.7, target=0.9)
    band = label.split(",")[0]
    assert f"{band.lower()} bantta" in out
    assert ("eşiğin <strong>üstünde</strong>" in out) == (prob >= threshold)
    fig = figure_html(pred, band_low=0.3, band_high=0.7)
    assert f"<td>{band}</td>" in fig  # panel d: bant etiketi
    assert ("İncelenmeli" in fig) == (prob >= threshold)
    assert ("Düşük" in fig and "İncelenmeli" in fig) is False


def test_scale_band_is_widened_to_cover_threshold():
    pred = _pred(0.5, 0.18)
    cap = caption_html(pred, band_low=0.3, band_high=0.7)
    assert "%18–70" in cap
    assert "genişletildi" in cap
    fig = figure_html(pred, band_low=0.3, band_high=0.7)
    # Eksen y=10 (1,0) ile y=210 (0) arası: bant 0,70'ten (y=70) eşiğe 0,18'e (y=174) iner.
    assert 'class="sl-axis__band" x="40.5" y="70.0"' in fig
    assert 'height="104.0"' in fig
    assert "0,18–0,70" in fig
    plain = caption_html(_pred(0.5, 0.5), band_low=0.3, band_high=0.7)
    assert "genişletildi" not in plain


def test_caption_and_figure_states():
    assert "değerlendirmeden sonra" in caption_html()
    assert "model yüklendiğinde" in caption_html(model_ready=False)
    assert "ONNX" in caption_html(_pred(0.5, 0.18), has_cam=False)
    assert 'aria-busy="true"' in loading_html()


def test_gate_note_only_before_consent():
    assert "onayladığınızda" in gate_html(False)
    assert "onayladığınızda" not in gate_html(True)


def test_privacy_text_matches_cache_cleanup():
    interval, max_age = DELETE_CACHE
    assert interval + max_age <= 20 * 60
    assert "20 dakika" in PRIVACY


def test_theme_loads_no_external_fonts():
    t = theme()
    assert "googleapis" not in str(t._stylesheets)  # ziyaretçi IP'si Google'a gitmez
    assert "Source Sans 3" in t.font
    css = page_css()
    assert "googleapis" not in css
    assert css.count("@font-face") == len(SERIF_FILES)
    assert all((FONTS_DIR / name).is_file() for name, _, _ in SERIF_FILES)


def test_gradio_analytics_off_by_default():
    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"


def test_states_and_formatting():
    assert pct(0.1234) == "%12,3"
    assert "işaretleyin" in empty_html(False)
    assert "Görüntüyü değerlendir" in empty_html(True)
    assert 'role="alert"' in error_html("<b>x</b>")
    assert "<b>" not in error_html("<b>x</b>")


def test_about_without_metrics_says_so(predictor):
    text = about_html(predictor)
    assert "test ölçümü yok" in text
    assert "Cilt tonu" in text
    assert "0,763" in text  # ilk sürümün README'deki ölçümü
    assert "test ölçümü yok" in about_html(None)
    assert "<ul>" not in text  # Sınırlar tek paragraf


def test_english_disclaimer_only_in_closing_note():
    assert 'lang="en"' not in input_intro_html()
    assert DISCLAIMER_TR in input_intro_html()
    assert about_html(None).count(DISCLAIMER_EN) == 1
