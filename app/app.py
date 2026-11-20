"""Gradio demosu. Eğitim ve araştırma amaçlıdır; tıbbi teşhis değildir.

Çalıştırma:
    python app/app.py
Ortam değişkenleri: SKINLESION_CHECKPOINT, MODEL_REPO, MODEL_REVISION, SKINLESION_BACKEND,
GRADIO_SERVER_NAME (varsayılan 127.0.0.1), GRADIO_SERVER_PORT (varsayılan 7860).
"""

from __future__ import annotations

import html
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))  # Space'te paket kurulu olmayabilir

from skinlesion.errors import InvalidImageError, SkinLesionError  # noqa: E402
from skinlesion.explain import DEFAULT_ALPHA, overlay  # noqa: E402
from skinlesion.predict import (  # noqa: E402
    Prediction,
    Predictor,
    band_limits,
    resolve_checkpoint_dir,
)

logger = logging.getLogger("skinlesion.app")

EXAMPLES_DIR = ROOT / "examples"
CSS_PATH = Path(__file__).with_name("style.css")
MODEL_CARD_URL = "https://github.com/rose-omer/Skin-Cancer-Classification/blob/main/MODEL_CARD.md"

BAND_LABELS = {"dusuk": "Düşük", "belirsiz": "Belirsiz", "yuksek": "Yüksek"}
BAND_MARKER = {"dusuk": "var(--sl-low)", "belirsiz": "var(--sl-mid)", "yuksek": "var(--sl-high)"}

DISCLAIMER_TR = (
    "Bu bir eğitim ve araştırma demosudur. Tıbbi teşhis değildir. "
    "Şüpheli bir ben ya da lezyon için dermatoloğa başvurun."
)
DISCLAIMER_EN = (
    "This is an educational and research demo. It is not a medical diagnosis. "
    "See a dermatologist about any suspicious mole or lesion."
)
PRIVACY = (
    "Yüklenen görüntü kalıcı olarak saklanmaz ve işaretlenmez. "
    "Sunucudaki geçici kopya yaklaşık 20 dakika içinde silinir."
)
# Temizlik 10 dakikada bir koşar ve 10 dakikadan eski dosyaları siler: üst sınır ~20 dakika.
DELETE_CACHE = (600, 600)
EXAMPLES_LABEL = "Örnek görüntüler (ISIC Archive kökenli; görüntü kimliği doğrulanmadı)"
GATE_NOTE = "Yükleme alanı, üstteki uyarıyı onayladığınızda açılır."
UPLOAD_PLACEHOLDER = "Görüntüyü buraya bırakın ya da tıklayıp seçin. JPEG ya da PNG."
CAM_NOTE = (
    "Isı haritası, modelin karar verirken ağırlık verdiği bölgeyi gösterir. Tanı kanıtı değildir."
)

ICON_INFO = (
    '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">'
    '<circle cx="10" cy="10" r="8.25" stroke="currentColor" stroke-width="1.5"/>'
    '<path d="M10 9v5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>'
    '<circle cx="10" cy="6.25" r="1" fill="currentColor"/></svg>'
)
ICON_WARN = (
    '<svg width="18" height="18" viewBox="0 0 20 20" fill="none" aria-hidden="true">'
    '<path d="M10 2.5 18 17H2L10 2.5Z" stroke="currentColor" stroke-width="1.5" '
    'stroke-linejoin="round"/><path d="M10 8v4" stroke="currentColor" stroke-width="1.5" '
    'stroke-linecap="round"/><circle cx="10" cy="14.4" r="0.9" fill="currentColor"/></svg>'
)

ICON_LOCK = (
    '<svg width="16" height="16" viewBox="0 0 20 20" fill="none" aria-hidden="true">'
    '<rect x="4" y="9" width="12" height="8.5" rx="1.5" stroke="currentColor" '
    'stroke-width="1.5"/><path d="M7 9V6.5a3 3 0 0 1 6 0V9" stroke="currentColor" '
    'stroke-width="1.5"/></svg>'
)


def dec(value: float, digits: int = 3) -> str:
    """Türkçe ondalık biçimi: 0,270"""
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 1) -> str:
    """Türkçe yüzde biçimi: %12,4"""
    return "%" + dec(value * 100, digits)


# --- HTML parçaları ---------------------------------------------------------------------------


def header_html() -> str:
    return (
        '<header class="sl-header">'
        "<h1>Dermoskopi lezyon sınıflandırma demosu</h1>"
        "<p>Bir dermoskopi görüntüsü için kalibre malign olasılığını, karar eşiğini ve "
        "modelin ağırlık verdiği bölgeyi gösterir.</p>"
        "</header>"
    )


def notice_html() -> str:
    return (
        '<div class="sl-notice" role="note" aria-label="Uyarı">'
        f"{ICON_INFO}<div>"
        f"<p><strong>{html.escape(DISCLAIMER_TR)}</strong></p>"
        f'<p class="sl-en" lang="en">{html.escape(DISCLAIMER_EN)}</p>'
        f'<p class="sl-privacy">{html.escape(PRIVACY)}</p>'
        "</div></div>"
    )


def empty_html(consented: bool) -> str:
    step = (
        "Bir dermoskopi görüntüsü yükleyin ya da örneklerden birini seçin, sonra "
        "“Görüntüyü değerlendir” düğmesine basın."
        if consented
        else "Önce üstteki uyarıyı okuyun ve “Bunu okudum ve anladım” kutusunu işaretleyin. "
        "Yükleme alanı ondan sonra açılır."
    )
    return (
        '<section class="sl-panel sl-empty" aria-live="polite">'
        '<h2 class="sl-h2">Sonuç burada görünür</h2>'
        f"<p>{step}</p></section>"
    )


def gate_html(consented: bool) -> str:
    """Onaysız durumda pasif yükleme alanının nedenini söyler."""
    if consented:
        return '<div class="sl-gate" aria-live="polite"></div>'
    return (
        '<div class="sl-gate" aria-live="polite">'
        f'<p class="sl-gate__note">{ICON_LOCK}<span>{GATE_NOTE}</span></p></div>'
    )


def error_html(message: str) -> str:
    return (
        '<section class="sl-panel sl-error" role="alert">'
        '<h2 class="sl-h2">Görüntü değerlendirilemedi</h2>'
        f"<p>{html.escape(message)}</p></section>"
    )


def verdict_label(pred: Prediction) -> str:
    """Bant ve eşik tek etikette: "Belirsiz, eşiğin üstünde"."""
    side = "üstünde" if pred["above_threshold"] else "altında"
    return f"{BAND_LABELS[pred['band']]}, eşiğin {side}"


def _ticks(low: float, high: float, min_gap: float = 0.06) -> list[float]:
    """Ölçek etiketleri; birbirine çok yakın olanlar atlanır."""
    out = [0.0]
    for value in (low, high):
        if value - out[-1] >= min_gap and 1.0 - value >= min_gap:
            out.append(value)
    return [*out, 1.0]


def result_html(pred: Prediction, *, band_low: float, band_high: float, target: float) -> str:
    prob, threshold, band = pred["prob"], pred["threshold"], pred["band"]
    above = pred["above_threshold"]
    side = "üstünde" if above else "altında"
    low, high = band_limits(threshold, band_low, band_high)
    adjusted = (low, high) != (band_low, band_high)
    scale_label = (
        f"Olasılık ölçeği: sonuç {pct(prob)}, belirsiz bant {pct(low, 0)} ile "
        f"{pct(high, 0)} arası, eşik {pct(threshold)}."
    )
    band_note = (
        " Belirsiz bant eşiği kapsayacak biçimde genişletildi; düşük bant hep eşiğin altında kalır."
        if adjusted
        else ""
    )
    advice = (
        "Tarama kuralına göre bu lezyonun bir dermatoloğa gösterilmesi önerilir."
        if above
        else "Bu sonuç tek başına lezyonun zararsız olduğunu göstermez."
    )
    ticks = "".join(
        f'<span style="left:{t * 100:.2f}%">{t * 100:.0f}</span>' for t in _ticks(low, high)
    )
    ood = ""
    if pred["ood"]:
        ood = (
            f'<p class="sl-ood" role="alert">{ICON_WARN}<span>Bu görüntü dermoskopi görüntüsüne '
            "benzemiyor. Model bu tür görüntüyle eğitilmedi; sonuç anlamsız olabilir.</span></p>"
        )
    return f"""
<section class="sl-panel" aria-live="polite" aria-labelledby="sl-result-title">
  <h2 class="sl-h2" id="sl-result-title">Sonuç</h2>
  <div class="sl-readout">
    <span class="sl-prob">{pct(prob)}</span>
    <span class="sl-band sl-band--{band}"><span>{BAND_LABELS[band]}<span
      class="sl-band__rule">, eşiğin {side}</span></span></span>
  </div>
  <p class="sl-caption">Kalibre malign olasılığı. Bant: %0–{pct(low, 0)[1:]} düşük,
    {pct(low, 0)}–{pct(high, 0)[1:]} belirsiz, üstü yüksek.{band_note}</p>
  <div class="sl-scale" role="img" aria-label="{html.escape(scale_label)}">
    <div class="sl-scale__track">
      <div class="sl-scale__band" style="left:{low * 100:.2f}%;
        width:{(high - low) * 100:.2f}%"></div>
      <div class="sl-scale__threshold" style="left:{threshold * 100:.2f}%"></div>
      <div class="sl-scale__marker" style="left:{prob * 100:.2f}%;
        --sl-marker:{BAND_MARKER[band]}"></div>
    </div>
    <div class="sl-scale__ticks" aria-hidden="true">{ticks}</div>
    <ul class="sl-legend" aria-hidden="true">
      <li><span class="sl-key sl-key--marker"></span>Bu görüntü</li>
      <li><span class="sl-key sl-key--threshold"></span>Eşik {pct(threshold)}</li>
      <li><span class="sl-key sl-key--band"></span>Belirsiz bant</li>
    </ul>
  </div>
  <p class="sl-threshold">Eşik, doğrulama kümesinde {pct(target, 0)} duyarlılık için seçildi.
    Bu görüntünün olasılığı eşiğin <strong>{side}</strong>. {advice}</p>
  {ood}
  <p class="sl-footnote">{html.escape(DISCLAIMER_TR)}</p>
</section>"""


def missing_html(ckpt_dir: Path, reason: str) -> str:
    return f"""
<section class="sl-panel sl-missing" role="alert">
  <h2 class="sl-h2">Model dosyası bulunamadı</h2>
  <p>Demo, eğitilmiş bir checkpoint olmadan tahmin yapmaz. Rastgele ağırlıkla sonuç
    göstermek yanıltıcı olurdu.</p>
  <p>Aranan yer: <code>{html.escape(str(ckpt_dir))}</code><br>
    Ayrıntı: {html.escape(reason)}</p>
  <p>Demoyu açmak için iki yol var:</p>
  <ul>
    <li>Modeli eğitin ve checkpoint'i bu klasöre yazın:
      <pre><code>python -m skinlesion.train --config configs/effv2s_fanconic.yaml</code></pre></li>
    <li>Ya da bir Hugging Face model reposu verin:
      <pre><code>MODEL_REPO=&lt;kullanıcı&gt;/&lt;repo&gt; python app/app.py</code></pre></li>
  </ul>
</section>"""


def about_markdown(predictor: Predictor | None) -> str:
    lines = ["## Model hakkında", ""]
    if predictor is not None and predictor.config is not None:
        cfg = predictor.config
        lines += [
            f"- Omurga: `{cfg.model_name}`, giriş {cfg.img_size} px.",
            f"- Sıcaklık (temperature scaling): {dec(cfg.temperature)}. "
            f"Eşik: {dec(cfg.threshold)}, doğrulama kümesinde {pct(cfg.target_sensitivity, 0)} "
            "duyarlılık hedefiyle seçildi.",
        ]
    lines += ["", "### Test sonuçları", ""]
    metrics = predictor.metrics if predictor is not None else None
    if metrics:
        sel = metrics["test"]["at_selected"]

        def ci(m: dict) -> str:
            return f"{m['value']:.3f} (%95 GA {m['ci95'][0]:.3f}–{m['ci95'][1]:.3f})"

        lines += [
            "| Ölçüt | Değer |",
            "|---|---|",
            f"| Duyarlılık | {ci(sel['sensitivity'])} |",
            f"| Özgüllük | {ci(sel['specificity'])} |",
            f"| AUROC | {ci(sel['auroc'])} |",
            f"| Test örneği | {sel['n']} |",
            "",
            "Test kümesi dengelidir. %1,8 yaygınlıkta (ISIC 2020) pozitif öngörü değeri "
            f"yaklaşık {pct(sel['ppv_at_isic2020_prevalence'])} olur.",
        ]
    else:
        lines.append(
            "Bu checkpoint için test ölçümü yok. `python -m skinlesion.evaluate --final` "
            "komutu ölçümü üretir."
        )
    lines += [
        "",
        "### Veri seti ve lisans",
        "",
        "- Kaggle `fanconic/skin-cancer-malignant-vs-benign`, ISIC Archive kaynaklı. Lisans "
        "belirtilmemiş; açıklama ticari kullanımı yasaklıyor.",
        "- Örnek görüntüler aynı Kaggle setinden alındı ve ISIC Archive kökenlidir. Görüntü "
        "başına ISIC kimliği henüz doğrulanmadı. Demo ticari olmayan kullanım içindir.",
        "",
        "### Bilinen sınırlar",
        "",
        "- Cilt tonu (Fitzpatrick) dağılımı ölçülmedi. Koyu tenlerde başarım bilinmiyor.",
        "- Test kümesi yapay olarak dengeli (%45 malign). Gerçek yaygınlık çok daha düşük.",
        "- Tek kaynak: hasta ve lezyon kimliği yok; aynı lezyon farklı kümelerde olabilir.",
        "- Telefon fotoğrafı kapsam dışı. Model yalnız dermoskopi görüntüsüyle eğitildi.",
        "",
        f"[Model kartı]({MODEL_CARD_URL})",
    ]
    return "\n".join(lines)


def cam_alt(pred: Prediction) -> str:
    return (
        f"Grad-CAM ısı haritası. Kalibre malign olasılığı {pct(pred['prob'])}, "
        f"{verdict_label(pred).lower()}."
    )


# --- Arayüz -----------------------------------------------------------------------------------


@dataclass
class LoadResult:
    predictor: Predictor | None
    ckpt_dir: Path
    reason: str = ""


def load_predictor() -> LoadResult:
    try:
        ckpt_dir = resolve_checkpoint_dir()
    except SkinLesionError as exc:
        return LoadResult(None, Path(os.environ.get("MODEL_REPO", "?")), str(exc))
    try:
        return LoadResult(Predictor(ckpt_dir).load(), ckpt_dir)
    except SkinLesionError as exc:
        return LoadResult(None, ckpt_dir, str(exc))


def build_demo(loaded: LoadResult) -> gr.Blocks:
    predictor = loaded.predictor
    with gr.Blocks(title="Dermoskopi sınıflandırma demosu", delete_cache=DELETE_CACHE) as demo:
        gr.HTML(header_html())
        if predictor is None:
            with gr.Group(elem_classes="sl-consent"):
                gr.HTML(notice_html() + "<div style='height:16px'></div>")
            gr.HTML(missing_html(loaded.ckpt_dir, loaded.reason))
            return demo

        cfg = predictor.config
        with gr.Group(elem_classes="sl-consent"):
            gr.HTML(notice_html())
            consent = gr.Checkbox(label="Bunu okudum ve anladım", value=False)

        with gr.Tabs():
            with gr.Tab("Değerlendir"), gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=300):
                    gr.HTML('<h2 class="sl-h2">Görüntü</h2>')
                    gate = gr.HTML(gate_html(False))
                    image_in = gr.Image(
                        label="Dermoskopi görüntüsü",
                        type="pil",
                        sources=["upload"],
                        interactive=False,
                        height=320,
                        placeholder=UPLOAD_PLACEHOLDER,
                    )
                    run_btn = gr.Button(
                        "Görüntüyü değerlendir", variant="primary", interactive=False
                    )
                    examples = sorted(EXAMPLES_DIR.glob("*.jpg"))
                    if examples:
                        gr.Examples(
                            examples=[[str(p)] for p in examples],
                            inputs=image_in,
                            label=EXAMPLES_LABEL,
                        )
                with gr.Column(scale=6, min_width=300):
                    result = gr.HTML(empty_html(False))
                    with gr.Group(visible=False, elem_classes="sl-cam") as cam_group:
                        cam_img = gr.Image(
                            label="Grad-CAM",
                            type="pil",
                            interactive=False,
                            buttons=[],
                            height=320,
                        )
                        alpha = gr.Slider(
                            0.0,
                            1.0,
                            value=DEFAULT_ALPHA,
                            step=0.05,
                            label="Isı haritası saydamlığı",
                        )
                        gr.HTML(f'<p class="sl-hint" style="padding:0 12px 12px">{CAM_NOTE}</p>')
            with gr.Tab("Model hakkında"):
                gr.Markdown(about_markdown(predictor), elem_classes="sl-about")

        cam_state = gr.State(None)

        def on_consent(checked: bool):
            # Onay kalkarsa önceki sonuç da gizlenir.
            return (
                gr.update(interactive=checked),
                gr.update(interactive=checked),
                empty_html(checked),
                gr.update(visible=False),
                None,
                gate_html(checked),
            )

        def analyze(checked: bool, image, a: float):
            hidden = (gr.update(visible=False), gr.update(value=None), None)
            if not checked:
                return error_html("Önce uyarıyı okuyun ve onay kutusunu işaretleyin."), *hidden
            if image is None:
                return error_html("Görüntü yok. Bir dermoskopi görüntüsü yükleyin."), *hidden
            try:
                pred = predictor.predict(image, with_cam=predictor.supports_cam)
                view = predictor.view(image)
            except InvalidImageError as exc:
                return error_html(f"{exc} Başka bir görüntü deneyin."), *hidden
            except Exception:
                logger.exception("Tahmin başarısız")
                return error_html(
                    "Görüntü işlenirken bir hata oluştu. Başka bir görüntü deneyin."
                ), *hidden
            html_out = result_html(
                pred, band_low=cfg.band_low, band_high=cfg.band_high, target=cfg.target_sensitivity
            )
            if pred["cam"] is None:
                return html_out, *hidden
            state = (view, pred["cam"], cam_alt(pred))
            return html_out, gr.update(visible=True), _blend(state, a), state

        def _blend(state, a: float):
            view, cam, alt = state
            return gr.update(value=overlay(view, cam, float(a)), alt_text=alt)

        def reblend(state, a: float):
            return gr.update() if state is None else _blend(state, a)

        def on_clear(checked: bool):
            return empty_html(checked), gr.update(visible=False), gr.update(value=None), None

        consent.change(on_consent, consent, [image_in, run_btn, result, cam_group, cam_state, gate])
        # Mobilde sonuç paneli görüntünün altında kalır; sayfa sonuca kayar.
        run_btn.click(
            analyze,
            [consent, image_in, alpha],
            [result, cam_group, cam_img, cam_state],
            scroll_to_output=True,
        )
        alpha.change(reblend, [cam_state, alpha], cam_img, show_progress="hidden")
        image_in.clear(on_clear, consent, [result, cam_group, cam_img, cam_state])
    return demo


def theme() -> gr.themes.Base:
    # Gradio paketiyle gelen yazı tipi aynı kökenden sunulur; ziyaretçi IP'si Google'a gitmez.
    font = [gr.themes.LocalFont("Source Sans 3", weights=(400, 600)), "system-ui", "sans-serif"]
    return gr.themes.Base(font=font, radius_size=gr.themes.sizes.radius_sm).set(
        body_background_fill="#f4f6f5",
        body_background_fill_dark="#111614",
        body_text_color="#16201d",
        body_text_color_dark="#e5ebe8",
        body_text_color_subdued="#4a5753",
        body_text_color_subdued_dark="#a7b3ae",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#19201e",
        background_fill_secondary="#f4f6f5",
        background_fill_secondary_dark="#111614",
        block_background_fill="#ffffff",
        block_background_fill_dark="#19201e",
        block_border_color="#cfd7d4",
        block_border_color_dark="#2f3a36",
        border_color_primary="#cfd7d4",
        border_color_primary_dark="#2f3a36",
        block_shadow="none",
        block_label_text_color="#4a5753",
        block_label_text_color_dark="#a7b3ae",
        color_accent="#1d5c58",
        border_color_accent="#1d5c58",
        border_color_accent_dark="#7cc2ba",
        border_color_accent_subdued="#93a19c",
        border_color_accent_subdued_dark="#5d6c66",
        color_accent_soft="#dcebe9",
        color_accent_soft_dark="#1c3331",
        slider_color="#1d5c58",
        slider_color_dark="#7cc2ba",
        checkbox_background_color_selected="#1d5c58",
        checkbox_background_color_selected_dark="#7cc2ba",
        checkbox_border_color_selected="#1d5c58",
        checkbox_border_color_selected_dark="#7cc2ba",
        checkbox_border_color="#93a19c",
        checkbox_border_color_dark="#5d6c66",
        checkbox_label_text_size="1rem",
        button_primary_background_fill="#1d5c58",
        button_primary_background_fill_hover="#164a47",
        button_primary_background_fill_dark="#7cc2ba",
        button_primary_background_fill_hover_dark="#95d0c9",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#0d1a18",
        button_primary_shadow="none",
        button_large_text_weight="600",
        button_medium_text_weight="600",
        loader_color="#1d5c58",
        link_text_color="#1d5c58",
        link_text_color_dark="#7cc2ba",
        input_border_color_focus="#1d5c58",
        input_border_color_focus_dark="#7cc2ba",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    loaded = load_predictor()
    if loaded.predictor is None:
        logger.warning("Checkpoint yüklenemedi: %s", loaded.reason)
    demo = build_demo(loaded)
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        theme=theme(),
        css=CSS_PATH.read_text(encoding="utf-8"),
        footer_links=[],
    )


if __name__ == "__main__":
    main()
