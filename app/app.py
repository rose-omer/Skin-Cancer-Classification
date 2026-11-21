"""Gradio demosu. Eğitim ve araştırma amaçlıdır; tıbbi teşhis değildir.

Çalıştırma:
    python app/app.py
Ortam değişkenleri: SKINLESION_CHECKPOINT, MODEL_REPO, MODEL_REVISION, SKINLESION_BACKEND,
GRADIO_SERVER_NAME (varsayılan 127.0.0.1), GRADIO_SERVER_PORT (varsayılan 7860),
GRADIO_ANALYTICS_ENABLED (varsayılan False: Hugging Face'e kullanım verisi gitmez).
"""

from __future__ import annotations

import html
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Gradio içe aktarılmadan önce: telemetri yalnız açıkça istenirse açılır.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402

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
FONTS_DIR = Path(__file__).with_name("fonts")
MODEL_CARD_URL = "https://github.com/rose-omer/Skin-Cancer-Classification/blob/main/MODEL_CARD.md"

BAND_LABELS = {"dusuk": "Düşük", "belirsiz": "Belirsiz", "yuksek": "Yüksek"}

DISCLAIMER_SHORT = "Eğitim ve araştırma demosu. Tıbbi teşhis değildir."
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
GATE_NOTE = "Yükleme alanı, uyarıyı onayladığınızda açılır."
UPLOAD_PLACEHOLDER = "Görüntüyü buraya bırakın ya da tıklayıp seçin. JPEG ya da PNG."

# Ölçüm ilk sürümden (README, `legacy/`); yeni model satırı checkpoint'in ölçümünden gelir.
LEGACY_ROW = ("Xception linear probe, ilk sürüm", "0,763", "0,864", "hesaplanmadı")

# Source Serif 4 (OFL-1.1, @fontsource-variable/source-serif-4 5.3.1). Aynı kökenden sunulur.
SERIF_LATIN = (
    "U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,"
    "U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD"
)
SERIF_LATIN_EXT = (
    "U+0100-02BA,U+02BD-02C5,U+02C7-02CC,U+02CE-02D7,U+02DD-02FF,U+0304,U+0308,U+0329,"
    "U+1D00-1DBF,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20C4,U+2113,"
    "U+2C60-2C7F,U+A720-A7FF"
)
SERIF_FILES = (
    ("source-serif-4-latin-ext-wght-normal.woff2", "normal", SERIF_LATIN_EXT),
    ("source-serif-4-latin-wght-normal.woff2", "normal", SERIF_LATIN),
    ("source-serif-4-latin-ext-wght-italic.woff2", "italic", SERIF_LATIN_EXT),
    ("source-serif-4-latin-wght-italic.woff2", "italic", SERIF_LATIN),
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


def page_css() -> str:
    """Yazı tipi tanımı + style.css. Yazı tipi dosyaları `set_static_paths` ile sunulur."""
    faces = "".join(
        f"@font-face{{font-family:'Source Serif 4';font-style:{style};font-weight:200 900;"
        f"font-display:swap;src:url('gradio_api/file={(FONTS_DIR / name).as_posix()}') "
        f"format('woff2');unicode-range:{urange}}}\n"
        for name, style, urange in SERIF_FILES
    )
    return faces + CSS_PATH.read_text(encoding="utf-8")


# --- HTML parçaları ---------------------------------------------------------------------------


def header_html() -> str:
    return (
        '<header class="sl-header">'
        f'<p class="sl-standfirst">{DISCLAIMER_SHORT}</p>'
        "<h1>Bir dermoskopi görüntüsü için kalibre malign olasılığı, karar eşiği ve Grad-CAM</h1>"
        "</header>"
    )


def input_intro_html() -> str:
    """Girdi kutusunun başı: kalıcı uyarı, onay kutusunun hemen üstünde."""
    return (
        '<div class="sl-input__intro">'
        '<h2 class="sl-h2" id="sl-input-h">Girdi</h2>'
        f'<p role="note">{html.escape(DISCLAIMER_TR)}</p>'
        "</div>"
    )


def privacy_html() -> str:
    return f'<p class="sl-hint">{html.escape(PRIVACY)}</p>'


def _abstract(body: str, *, attrs: str = 'aria-live="polite"', extra: str = "") -> str:
    return (
        f'<section class="sl-abstract" {attrs}>'
        f'<p><span class="sl-runin">Özet.</span> {body}</p>{extra}</section>'
    )


def empty_html(consented: bool) -> str:
    step = (
        "Bir dermoskopi görüntüsü yükleyin ya da örneklerden birini seçin, sonra "
        "“Görüntüyü değerlendir” düğmesine basın."
        if consented
        else "Önce Girdi bölümündeki uyarıyı okuyun ve “Bunu okudum ve anladım” kutusunu "
        "işaretleyin. Yükleme alanı ondan sonra açılır."
    )
    return _abstract(
        "Model, yüklenen görüntü için kalibre malign olasılığını, doğrulama kümesinde seçilen "
        f"karar eşiğini ve ağırlık verdiği bölgeyi gösterir. {step}"
    )


def loading_html() -> str:
    return _abstract(
        "Görüntü değerlendiriliyor. Olasılık, eşik ve ısı haritası birkaç saniye içinde "
        "Şekil 1’de görünür.",
        attrs='aria-live="polite" aria-busy="true"',
        extra='<div class="sl-progress" aria-hidden="true"></div>',
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
        '<section class="sl-abstract sl-error" role="alert">'
        f'<h2 class="sl-h2">{ICON_WARN}<span>Görüntü değerlendirilemedi</span></h2>'
        f"<p>{html.escape(message)}</p></section>"
    )


def verdict_label(pred: Prediction) -> str:
    """Bant ve eşik tek etikette: "Belirsiz, eşiğin üstünde"."""
    side = "üstünde" if pred["above_threshold"] else "altında"
    return f"{BAND_LABELS[pred['band']]}, eşiğin {side}"


def result_html(pred: Prediction, *, band_low: float, band_high: float, target: float) -> str:
    """Özet paragrafı: olasılık, eşik, bant ve öneri tek cümle dizisinde."""
    prob, threshold, band = pred["prob"], pred["threshold"], pred["band"]
    side = "üstünde" if pred["above_threshold"] else "altında"
    advice = (
        "Tarama kuralına göre bu lezyonun bir dermatoloğa gösterilmesi önerilir."
        if pred["above_threshold"]
        else "Bu sonuç tek başına lezyonun zararsız olduğunu göstermez."
    )
    ood = ""
    if pred["ood"]:
        ood = (
            f'<p class="sl-ood" role="alert">{ICON_WARN}<span>Bu görüntü dermoskopi görüntüsüne '
            "benzemiyor. Model bu tür görüntüyle eğitilmedi; sonuç anlamsız olabilir.</span></p>"
        )
    body = (
        f"Kalibre malign olasılığı <strong>{pct(prob)}</strong>; doğrulama kümesinde "
        f"{pct(target, 0)} duyarlılık için seçilen eşik {pct(threshold)}. Sonuç "
        f"{BAND_LABELS[band].lower()} bantta ve eşiğin <strong>{side}</strong> kaldı. {advice}"
    )
    return _abstract(
        body,
        attrs='aria-live="polite" aria-label="Sonuç özeti"',
        extra=f'{ood}<p class="sl-footnote">{html.escape(DISCLAIMER_TR)}</p>',
    )


# Panel c: dikey olasılık ekseni. viewBox 240 x 240; eksen y=10 (1,0) ile y=210 (0) arası.
AXIS_TOP, AXIS_BOTTOM = 10, 210


def _y(p: float) -> float:
    return AXIS_BOTTOM - (AXIS_BOTTOM - AXIS_TOP) * p


def _ticks(*values: float, min_gap: float = 0.06) -> list[float]:
    """Eksen etiketleri; birbirine çok yakın olanlar atlanır. 0 ve 1 hep kalır."""
    out = [0.0]
    for value in sorted(set(values)):
        if value - out[-1] >= min_gap and 1.0 - value >= min_gap:
            out.append(value)
    return [*out, 1.0]


def _tick_label(value: float) -> str:
    if value in (0.0, 1.0):
        return "0" if value == 0.0 else "1,0"
    return dec(value, 2)


def figure_html(pred: Prediction, *, band_low: float, band_high: float) -> str:
    """Şekil 1'in c (olasılık ekseni) ve d (kural tablosu) panelleri."""
    prob, threshold, band = pred["prob"], pred["threshold"], pred["band"]
    above = pred["above_threshold"]
    low, high = band_limits(threshold, band_low, band_high)
    label = (
        f"Panel c: olasılık ekseni. Sonuç {pct(prob)}, eşik {pct(threshold)}, "
        f"belirsiz bant {pct(low, 0)} ile {pct(high, 0)} arası."
    )
    ticks = "".join(
        f'<text x="34" y="{_y(t) + 4:.1f}">{_tick_label(t)}</text>'
        for t in _ticks(threshold, low, high)
    )
    decision = "İncelenmeli" if above else "Eşiğin altında"
    return f"""
<div class="sl-cd">
  <div class="sl-cell">
    <p class="sl-panel-label">c</p>
    <svg class="sl-axis" viewBox="0 0 240 240" role="img" aria-label="{html.escape(label)}">
      <rect class="sl-axis__frame" x="40" y="{AXIS_TOP}" width="160" height="200"/>
      <rect class="sl-axis__band" x="40.5" y="{_y(high):.1f}" width="159"
        height="{_y(low) - _y(high):.1f}"/>
      <line class="sl-axis__threshold" x1="40" y1="{_y(threshold):.1f}" x2="200"
        y2="{_y(threshold):.1f}"/>
      <circle class="sl-axis__mark" cx="120" cy="{_y(prob):.1f}" r="7"
        style="--sl-rise:{(AXIS_BOTTOM - _y(prob)):.1f}px"/>
      <g class="sl-axis__ticks" text-anchor="end">{ticks}</g>
      <text class="sl-axis__value" x="206" y="{_y(prob) + 4:.1f}">{dec(prob)}</text>
      <text class="sl-axis__title" x="120" y="234" text-anchor="middle">p(malign)</text>
    </svg>
  </div>
  <div class="sl-cell">
    <p class="sl-panel-label">d</p>
    <table class="sl-rule">
      <caption>Kural ve sonuç</caption>
      <tbody>
        <tr><th scope="row">p</th><td>{dec(prob)}</td></tr>
        <tr><th scope="row">Eşik</th><td>{dec(threshold)}</td></tr>
        <tr><th scope="row">Bant</th><td>{dec(low, 2)}–{dec(high, 2)}</td></tr>
        <tr><th scope="row">Bant etiketi</th><td>{BAND_LABELS[band]}</td></tr>
        <tr class="sl-rule__decision{" is-flagged" if above else ""}"><th scope="row">Karar</th>
          <td>{decision}</td></tr>
      </tbody>
    </table>
  </div>
</div>"""


GHOST = (
    '<div class="sl-ghost" aria-hidden="true">'
    + "".join(f'<div class="sl-ghost__cell"><span>{p}</span></div>' for p in "abcd")
    + "</div>"
)


def caption_html(
    pred: Prediction | None = None,
    *,
    band_low: float = 0.3,
    band_high: float = 0.7,
    img_size: int = 224,
    has_cam: bool = True,
    model_ready: bool = True,
) -> str:
    """Şekil 1'in alt yazısı. Sonuç yokken boş panel iskeletiyle birlikte döner."""
    lead = '<span class="sl-runin">Şekil 1.</span>'
    if pred is None:
        when = "değerlendirmeden sonra" if model_ready else "model yüklendiğinde"
        return (
            f'{GHOST}<p class="sl-figcaption">{lead} Tek görüntü çıktısı {when} burada görünür: '
            "<b>a</b>, model girişi; <b>b</b>, Grad-CAM ısı haritası; <b>c</b>, kalibre "
            "olasılık, eşik ve belirsiz bant; <b>d</b>, karar kuralı.</p>"
        )
    low, high = band_limits(pred["threshold"], band_low, band_high)
    widened = (
        " Bant eşiği kapsayacak biçimde genişletildi; düşük bant hep eşiğin altında kalır."
        if (low, high) != (band_low, band_high)
        else ""
    )
    cam = (
        "<b>b</b>, Grad-CAM, son evrişim aşaması hedeflenir, kaydırıcıdaki saydamlıkla "
        f"bindirilir (varsayılan {pct(DEFAULT_ALPHA, 0)}); ısı haritası modelin ağırlık verdiği "
        "bölgeyi gösterir, tanı kanıtı değildir."
        if has_cam
        else "<b>b</b>, Grad-CAM bu arka uçta (ONNX) üretilmez."
    )
    rule = (
        "olasılık eşiğin üstünde olduğu için lezyon incelemeye yönlendirilir."
        if pred["above_threshold"]
        else "olasılık eşiğin altında kaldı; bu tek başına lezyonun zararsız olduğunu göstermez."
    )
    return (
        f'<p class="sl-figcaption">{lead} Tek görüntü çıktısı. <b>a</b>, model girişi '
        f"({img_size} px). {cam} <b>c</b>, kalibre malign olasılığı (turuncu nokta), eşik "
        f"(kesik çizgi) ve belirsiz bant {pct(low, 0)}–{pct(high, 0)[1:]} (mavi alan).{widened} "
        f"<b>d</b>, karar kuralı: {rule}</p>"
    )


def missing_html(ckpt_dir: Path, reason: str) -> str:
    return f"""
<section class="sl-abstract sl-missing" role="alert" aria-labelledby="sl-missing-h">
  <h2 class="sl-h2" id="sl-missing-h">Model dosyası bulunamadı</h2>
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


def missing_gate_html() -> str:
    return (
        '<div class="sl-gate">'
        f'<p class="sl-gate__note">{ICON_LOCK}<span>Model yüklenmediği için görüntü yükleme '
        "kapalı.</span></p></div>"
    )


def _metric_cell(m: dict) -> str:
    lo, hi = m["ci95"]
    return f'{dec(m["value"])}<span class="sl-ci">%95 GA {dec(lo)}–{dec(hi)}</span>'


def about_html(predictor: Predictor | None) -> str:
    """Yöntem, Tablo 1 ve Sınırlar: makalenin sonuç dışı bölümleri."""
    cfg = predictor.config if predictor is not None else None
    if cfg is not None:
        method = (
            f"<p>Omurga <i>{html.escape(cfg.model_name)}</i>, giriş {cfg.img_size} px. "
            f"Olasılık doğrulama kümesinde tek parametreli temperature scaling ile kalibre "
            f"edildi (T = {dec(cfg.temperature)}). Eşik {dec(cfg.threshold)}: doğrulama "
            f"kümesinde malign duyarlılığı {pct(cfg.target_sensitivity, 0)} hedefiyle "
            "seçildi.</p>"
        )
    else:
        method = "<p>Model yüklenmedi. Omurga, kalibrasyon ve eşik checkpoint ile gelir.</p>"
    method += (
        "<p>Veri: Kaggle <i>fanconic/skin-cancer-malignant-vs-benign</i>, ISIC Archive kaynaklı. "
        "Lisans belirtilmemiş; açıklama ticari kullanımı yasaklıyor. Örnek görüntüler aynı "
        "setten alındı; görüntü başına ISIC kimliği henüz doğrulanmadı. Demo ticari olmayan "
        "kullanım içindir.</p>"
    )

    metrics = predictor.metrics if predictor is not None else None
    if metrics:
        sel = metrics["test"]["at_selected"]
        current = [_metric_cell(sel[k]) for k in ("sensitivity", "specificity", "auroc")]
        note = (
            f"Test örneği {sel['n']}. Test kümesi dengelidir; %1,8 yaygınlıkta (ISIC 2020) "
            f"pozitif öngörü değeri yaklaşık {pct(sel['ppv_at_isic2020_prevalence'])} olur."
        )
    else:
        current = ["ölçüm yok"] * 3
        note = (
            "Bu checkpoint için test ölçümü yok. "
            "<code>python -m skinlesion.evaluate --final</code> komutu ölçümü üretir."
            if predictor is not None
            else "Model yüklenmediği için bu checkpoint'in test ölçümü yok."
        )
    legacy = "".join(f"<td>{v}</td>" for v in LEGACY_ROW[1:])
    current_cells = "".join(f"<td>{v}</td>" for v in current)
    table = f"""
<div class="sl-table-scroll" role="region" aria-labelledby="sl-tab1-h" tabindex="0">
  <table class="sl-booktabs">
    <thead><tr><th scope="col">Model</th><th scope="col">Duyarlılık</th>
      <th scope="col">Özgüllük</th><th scope="col">AUROC</th></tr></thead>
    <tbody>
      <tr><td>{LEGACY_ROW[0]}</td>{legacy}</tr>
      <tr class="sl-booktabs__main"><td>Bu checkpoint, seçilen eşik</td>{current_cells}</tr>
    </tbody>
  </table>
</div>"""
    limits = (
        "Cilt tonu (Fitzpatrick) dağılımı ölçülmedi; koyu tenlerde başarım bilinmiyor. "
        "Test kümesi yapay olarak dengeli (%45 malign); gerçek yaygınlık çok daha düşüktür. "
        "Veri tek kaynaktan gelir ve hasta kimliği taşımaz; aynı lezyon farklı kümelerde "
        "olabilir. Telefon fotoğrafı kapsam dışıdır."
    )
    return f"""
<div class="sl-about">
  <div class="sl-cols">
    <section aria-labelledby="sl-method-h">
      <h2 class="sl-h2" id="sl-method-h">Yöntem</h2>
      {method}
    </section>
    <section aria-labelledby="sl-tab1-h">
      <h2 class="sl-h2" id="sl-tab1-h">Tablo 1. Test sonuçları</h2>
      {table}
      <p class="sl-small">{note} İlk sürümün değerleri iyimserdir: test kümesi doğrulama ve
        erken durdurma için de kullanıldı. <a href="{MODEL_CARD_URL}">Model kartı</a></p>
    </section>
  </div>
  <section class="sl-limits" aria-labelledby="sl-lim-h">
    <h2 class="sl-h2" id="sl-lim-h">Sınırlar</h2>
    <p>{limits}</p>
  </section>
  <p class="sl-note" role="note">{html.escape(DISCLAIMER_TR)}</p>
  <p class="sl-note sl-en" lang="en">{html.escape(DISCLAIMER_EN)}</p>
</div>"""


def cam_alt(pred: Prediction) -> str:
    return (
        f"Panel b: Grad-CAM ısı haritası. Kalibre malign olasılığı {pct(pred['prob'])}, "
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
    gr.set_static_paths([FONTS_DIR])
    with gr.Blocks(title="Dermoskopi sınıflandırma demosu", delete_cache=DELETE_CACHE) as demo:
        with gr.Row(equal_height=False, elem_classes="sl-top"):
            with gr.Column(min_width=0, elem_classes="sl-lede"):
                gr.HTML(header_html())
                if predictor is None:
                    summary = gr.HTML(missing_html(loaded.ckpt_dir, loaded.reason))
                else:
                    summary = gr.HTML(empty_html(False))
            with gr.Column(min_width=0, elem_classes="sl-input"):
                gr.HTML(input_intro_html())
                if predictor is None:
                    gr.HTML(missing_gate_html())
                else:
                    consent = gr.Checkbox(label="Bunu okudum ve anladım", value=False)
                    gate = gr.HTML(gate_html(False))
                    image_in = gr.Image(
                        label="Dermoskopi görüntüsü",
                        type="pil",
                        sources=["upload"],
                        interactive=False,
                        visible=False,  # onaydan sonra kilit notunun yerine açılır
                        buttons=[],
                        height=220,
                        placeholder=UPLOAD_PLACEHOLDER,
                        elem_classes="sl-upload",
                    )
                    run_btn = gr.Button(
                        "Görüntüyü değerlendir", variant="primary", interactive=False
                    )
                    gr.HTML(privacy_html())
                    examples = sorted(EXAMPLES_DIR.glob("*.jpg"))
                    if examples:
                        gr.Examples(
                            examples=[[str(p)] for p in examples],
                            inputs=image_in,
                            label=EXAMPLES_LABEL,
                        )

        with gr.Column(elem_classes="sl-figure"):
            if predictor is not None:
                with gr.Row(visible=False, equal_height=False, elem_classes="sl-panels") as panels:
                    with gr.Column(min_width=0, elem_classes="sl-cell"):
                        gr.HTML('<p class="sl-panel-label">a</p>')
                        view_img = gr.Image(
                            label="Panel a: model girişi",
                            show_label=False,
                            type="pil",
                            interactive=False,
                            buttons=[],
                            container=False,
                        )
                    with gr.Column(min_width=0, elem_classes="sl-cell") as cam_col:
                        gr.HTML('<p class="sl-panel-label">b</p>')
                        cam_img = gr.Image(
                            label="Panel b: Grad-CAM",
                            show_label=False,
                            type="pil",
                            interactive=False,
                            buttons=[],
                            container=False,
                        )
                        alpha = gr.Slider(
                            0.0,
                            1.0,
                            value=DEFAULT_ALPHA,
                            step=0.05,
                            label="Saydamlık",
                            buttons=[],
                            elem_classes="sl-alpha",
                        )
                    chart = gr.HTML("", min_width=0, elem_classes="sl-cd-wrap")
            caption = gr.HTML(caption_html(model_ready=predictor is not None))

        gr.HTML(about_html(predictor))

        if predictor is None:
            return demo

        cfg = predictor.config
        cam_state = gr.State(None)
        empty_caption = caption_html()

        def on_consent(checked: bool):
            # Onay kalkarsa önceki sonuç da gizlenir.
            return (
                gr.update(interactive=checked, visible=checked),
                gr.update(interactive=checked),
                empty_html(checked),
                gr.update(visible=False),
                empty_caption,
                None,
                gate_html(checked),
            )

        def start():
            return loading_html(), gr.update(visible=False), empty_caption

        def analyze(checked: bool, image, a: float):
            def fail(message: str):
                return (
                    error_html(message),
                    gr.update(visible=False),
                    gr.update(value=None),
                    gr.update(),
                    gr.update(value=None),
                    "",
                    empty_caption,
                    None,
                )

            if not checked:
                return fail("Önce uyarıyı okuyun ve onay kutusunu işaretleyin.")
            if image is None:
                return fail("Görüntü yok. Bir dermoskopi görüntüsü yükleyin.")
            try:
                pred = predictor.predict(image, with_cam=predictor.supports_cam)
                view = predictor.view(image)
            except InvalidImageError as exc:
                return fail(f"{exc} Başka bir görüntü deneyin.")
            except Exception:
                logger.exception("Tahmin başarısız")
                return fail("Görüntü işlenirken bir hata oluştu. Başka bir görüntü deneyin.")
            bands = {"band_low": cfg.band_low, "band_high": cfg.band_high}
            has_cam = pred["cam"] is not None
            state = (view, pred["cam"], cam_alt(pred)) if has_cam else None
            return (
                result_html(pred, target=cfg.target_sensitivity, **bands),
                gr.update(visible=True),
                gr.update(value=view, alt_text=f"Panel a: model girişi, {cfg.img_size} px."),
                gr.update(visible=has_cam),
                _blend(state, a) if has_cam else gr.update(value=None),
                figure_html(pred, **bands),
                caption_html(pred, img_size=cfg.img_size, has_cam=has_cam, **bands),
                state,
            )

        def _blend(state, a: float):
            view, cam, alt = state
            return gr.update(value=overlay(view, cam, float(a)), alt_text=alt)

        def reblend(state, a: float):
            return gr.update() if state is None else _blend(state, a)

        def on_clear(checked: bool):
            return empty_html(checked), gr.update(visible=False), empty_caption, None

        consent.change(
            on_consent, consent, [image_in, run_btn, summary, panels, caption, cam_state, gate]
        )
        # Mobilde Şekil 1 girdinin altında kalır; sayfa özete kayar.
        run_btn.click(
            start, None, [summary, panels, caption], queue=False, show_progress="hidden"
        ).then(
            analyze,
            [consent, image_in, alpha],
            [summary, panels, view_img, cam_col, cam_img, chart, caption, cam_state],
            scroll_to_output=True,
            show_progress="hidden",
        )
        alpha.change(reblend, [cam_state, alpha], cam_img, show_progress="hidden")
        image_in.clear(on_clear, consent, [summary, panels, caption, cam_state])
    return demo


def theme() -> gr.themes.Base:
    # Arayüz yazısı Gradio paketiyle gelen Source Sans 3; metin yazısı app/fonts içindeki
    # Source Serif 4. İkisi de aynı kökenden sunulur; ziyaretçi IP'si Google'a gitmez.
    font = [gr.themes.LocalFont("Source Sans 3", weights=(400, 600)), "system-ui", "sans-serif"]
    ink, ink_dark = "#1a1a1a", "#e8e6e1"
    paper, paper_dark = "#ffffff", "#17191c"
    rule, rule_dark = "#bdbdbd", "#45484d"
    control, control_dark = "#767676", "#8e8b85"
    accent, accent_dark = "#2b5c8a", "#8fb8de"
    # Eş aralıklı yazı sistem yazı tipinden: IBM Plex Mono indirilmez.
    mono = ["ui-monospace", "Consolas", "monospace"]
    return gr.themes.Base(font=font, font_mono=mono, radius_size=gr.themes.sizes.radius_none).set(
        body_background_fill=paper,
        body_background_fill_dark=paper_dark,
        body_text_color=ink,
        body_text_color_dark=ink_dark,
        body_text_color_subdued="#4d4d4d",
        body_text_color_subdued_dark="#b3b0a9",
        background_fill_primary=paper,
        background_fill_primary_dark=paper_dark,
        background_fill_secondary=paper,
        background_fill_secondary_dark=paper_dark,
        block_background_fill=paper,
        block_background_fill_dark=paper_dark,
        block_border_color=rule,
        block_border_color_dark=rule_dark,
        block_border_width="0px",
        block_shadow="none",
        block_label_text_color="#3d3d3d",
        block_label_text_color_dark="#c9c6bf",
        block_label_background_fill=paper,
        block_label_background_fill_dark=paper_dark,
        border_color_primary=rule,
        border_color_primary_dark=rule_dark,
        color_accent=accent,
        border_color_accent=accent,
        border_color_accent_dark=accent_dark,
        border_color_accent_subdued=control,
        border_color_accent_subdued_dark=control_dark,
        color_accent_soft="#e4ecf4",
        color_accent_soft_dark="#22303d",
        input_background_fill=paper,
        input_background_fill_dark=paper_dark,
        input_border_color=control,
        input_border_color_dark=control_dark,
        input_border_color_focus=accent,
        input_border_color_focus_dark=accent_dark,
        slider_color=accent,
        slider_color_dark=accent_dark,
        checkbox_background_color=paper,
        checkbox_background_color_dark=paper_dark,
        checkbox_background_color_selected=accent,
        checkbox_background_color_selected_dark=accent_dark,
        checkbox_border_color=control,
        checkbox_border_color_dark=control_dark,
        checkbox_border_color_selected=accent,
        checkbox_border_color_selected_dark=accent_dark,
        checkbox_label_text_size="1rem",
        button_primary_background_fill=ink,
        button_primary_background_fill_hover="#3d3d3d",
        button_primary_background_fill_dark=ink_dark,
        button_primary_background_fill_hover_dark="#ffffff",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark=paper_dark,
        button_primary_border_color=ink,
        button_primary_border_color_dark=ink_dark,
        button_primary_shadow="none",
        button_large_text_weight="600",
        button_medium_text_weight="600",
        loader_color=accent,
        link_text_color=accent,
        link_text_color_dark=accent_dark,
        table_border_color=ink,
        table_border_color_dark=ink_dark,
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
        css=page_css(),
        footer_links=[],
    )


if __name__ == "__main__":
    main()
