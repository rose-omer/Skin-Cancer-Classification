"""Son değerlendirme: test kümesi yalnız burada ve yalnız bir kez kullanılır.

Kullanım:
    python -m skinlesion.evaluate --config configs/effv2s_fanconic.yaml --final
    python -m skinlesion.evaluate --table   # README tablosunu metrics.json'dan üretir
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from skinlesion.calibrate import apply_temperature, expected_calibration_error, reliability_bins
from skinlesion.config import load_config
from skinlesion.data import ImageDataset, build_transforms, read_split
from skinlesion.explain import grad_cam, overlay, save_gallery
from skinlesion.metrics import summarize
from skinlesion.model import METRICS_FILE, load_checkpoint, predict_logits

# İlk sürüm (legacy/ notebook, hücre 28): test kümesi doğrulama için de kullanıldı.
LEGACY_XCEPTION = {
    "name": "Xception linear probe (ilk sürüm)",
    "accuracy": 0.818,
    "sensitivity": 0.763,
    "specificity": 0.864,
    "precision": 0.824,
    "auroc": None,
    "note": "Test kümesi val ve erken durdurma için de kullanıldı; sonuç iyimser.",
}
GALLERY_SIZE = 12


def _logits(model, samples, img_size, mean, std, batch_size) -> tuple[np.ndarray, np.ndarray]:
    tf = build_transforms(img_size, mean, std, train=False)
    loader = DataLoader(ImageDataset(samples, tf), batch_size=batch_size, shuffle=False)
    logits, labels, _ = predict_logits(model, loader, torch.device("cpu"))
    return logits, labels


def _plot_curves(labels: np.ndarray, probs: np.ndarray, out_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    prec, rec, _ = precision_recall_curve(labels, probs)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(9, 4))
    ax_roc.plot(fpr, tpr)
    ax_roc.plot([0, 1], [0, 1], linestyle=":", color="grey")
    ax_roc.axhline(0.8, linestyle="--", color="grey", linewidth=0.8)
    ax_roc.set(title="ROC (kesikli çizgi: pAUC için TPR = 0.80)", xlabel="FPR", ylabel="TPR")
    ax_pr.plot(rec, prec)
    ax_pr.axhline(labels.mean(), linestyle=":", color="grey")
    ax_pr.set(title="Precision–Recall", xlabel="Recall", ylabel="Precision")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr.png", dpi=120)
    plt.close(fig)


def _plot_reliability(labels: np.ndarray, before: np.ndarray, after: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle=":", color="grey")
    for probs, name in ((before, "önce"), (after, "sonra")):
        conf, freq, counts = reliability_bins(probs, labels)
        mask = counts > 0
        ece = expected_calibration_error(probs, labels)
        ax.plot(conf[mask], freq[mask], marker="o", label=f"{name} (ECE {ece:.3f})")
    ax.set(xlabel="Ortalama olasılık", ylabel="Gözlenen malign oranı", title="Güvenilirlik")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_confusion(confusion: dict[str, int], threshold: float, out: Path) -> None:
    matrix = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(matrix, cmap="Greys")
    for (i, j), value in np.ndenumerate(matrix):
        colour = "white" if value > matrix.max() / 2 else "black"
        ax.text(j, i, str(value), ha="center", va="center", color=colour)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["benign", "malign"],
        yticklabels=["benign", "malign"],
        xlabel="Tahmin",
        ylabel="Gerçek",
        title=f"Eşik {threshold:.3f}",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _galleries(model, samples, probs, labels, threshold, cfg, out_dir: Path) -> None:
    tf = build_transforms(cfg.img_size, cfg.mean, cfg.std, train=False)

    def render(indices: list[int], title_fn) -> tuple[list[Image.Image], list[str]]:
        images, titles = [], []
        for i in indices:
            with Image.open(samples[i][0]) as img:
                rgb = img.convert("RGB").resize((cfg.img_size, cfg.img_size))
            cam = grad_cam(model, tf(rgb).unsqueeze(0))
            images.append(overlay(rgb, cam))
            titles.append(title_fn(i))
        return images, titles

    pred = probs >= threshold
    fn = [int(i) for i in np.argsort(probs) if labels[i] == 1 and not pred[i]][: GALLERY_SIZE // 2]
    fp = [int(i) for i in np.argsort(-probs) if labels[i] == 0 and pred[i]][: GALLERY_SIZE // 2]
    images, titles = render(fn + fp, lambda i: f"{'FN' if labels[i] else 'FP'} p={probs[i]:.2f}")
    save_gallery(images, titles, out_dir / "errors.png")

    rng = np.random.default_rng(0)
    pick = rng.choice(len(samples), size=min(GALLERY_SIZE, len(samples)), replace=False)
    images, titles = render(
        [int(i) for i in pick],
        lambda i: f"{'malign' if labels[i] else 'benign'} p={probs[i]:.2f}",
    )
    save_gallery(images, titles, out_dir / "gradcam.png")


def run_final(
    config_path: str, ckpt_dir: str | None, reports: Path, n_boot: int, rerun: bool
) -> dict:
    metrics_path = reports / METRICS_FILE
    if metrics_path.is_file() and not rerun:
        raise SystemExit(
            f"{metrics_path} zaten var: test kümesi bir kez ölçüldü. "
            "Bilerek yeniden ölçmek için --rerun-final verin; bu, raporda kayıt altına alınır."
        )
    train_cfg = load_config(config_path)
    model, cfg = load_checkpoint(ckpt_dir or train_cfg.run_dir)
    val = read_split(train_cfg.splits_dir, "val", train_cfg.data_root)
    test = read_split(train_cfg.splits_dir, "test", train_cfg.data_root, allow_test=True)
    args = (cfg.img_size, cfg.mean, cfg.std, train_cfg.batch_size)
    val_logits, val_labels = _logits(model, val, *args)
    test_logits, test_labels = _logits(model, test, *args)

    val_raw, val_cal = (
        apply_temperature(val_logits, 1.0),
        apply_temperature(val_logits, cfg.temperature),
    )
    test_raw, test_cal = (
        apply_temperature(test_logits, 1.0),
        apply_temperature(test_logits, cfg.temperature),
    )
    calibration = {
        "temperature": cfg.temperature,
        "val_ece_before": expected_calibration_error(val_raw, val_labels),
        "val_ece_after": expected_calibration_error(val_cal, val_labels),
        "test_ece_before": expected_calibration_error(test_raw, test_labels),
        "test_ece_after": expected_calibration_error(test_cal, test_labels),
    }
    if calibration["test_ece_after"] >= calibration["test_ece_before"]:
        calibration["note"] = (
            "Testte ECE düşmedi. Olası neden: val küçük ya da val ile test dağılımı farklı."
        )
    previous_runs = 0
    if metrics_path.is_file():
        previous_runs = json.loads(metrics_path.read_text(encoding="utf-8")).get("final_runs", 1)

    result = {
        "experiment_id": cfg.info.get("experiment_id"),
        "model_name": cfg.model_name,
        "final_runs": previous_runs + 1,
        "threshold": cfg.threshold,
        "target_sensitivity": cfg.target_sensitivity,
        "val": summarize(val_cal, val_labels, cfg.threshold, n_boot=n_boot),
        "test": {
            "at_0.5": summarize(test_cal, test_labels, 0.5, n_boot=n_boot),
            "at_selected": summarize(test_cal, test_labels, cfg.threshold, n_boot=n_boot),
        },
        "calibration": calibration,
        "legacy": LEGACY_XCEPTION,
    }

    reports.mkdir(parents=True, exist_ok=True)
    _plot_curves(test_labels, test_cal, reports)
    _plot_reliability(test_labels, test_raw, test_cal, reports / "reliability.png")
    _plot_confusion(
        result["test"]["at_selected"]["confusion"], cfg.threshold, reports / "confusion.png"
    )
    _galleries(model, test, test_cal, test_labels, cfg.threshold, cfg, reports)
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy(metrics_path, Path(ckpt_dir or train_cfg.run_dir) / METRICS_FILE)
    return result


def _fmt(metric: dict | None) -> str:
    if not metric:
        return "—"
    low, high = metric["ci95"]
    return f"{metric['value']:.3f} ({low:.3f}–{high:.3f})"


def render_table(metrics: dict) -> str:
    """README sonuç tablosu. Değerler elle yazılmaz; bu fonksiyon üretir."""
    legacy = metrics["legacy"]
    sel = metrics["test"]["at_selected"]
    half = metrics["test"]["at_0.5"]
    cal = metrics["calibration"]
    rows = [
        "| Model | Eşik | AUROC | Duyarlılık | Özgüllük | ECE |",
        "|---|---|---|---|---|---|",
        f"| {legacy['name']} | 0.5 (argmax) | hesaplanmadı | {legacy['sensitivity']:.3f} "
        f"| {legacy['specificity']:.3f} | hesaplanmadı |",
        f"| {metrics['model_name']} | 0.5 | {_fmt(half['auroc'])} | {_fmt(half['sensitivity'])} "
        f"| {_fmt(half['specificity'])} | {cal['test_ece_after']:.3f} |",
        f"| {metrics['model_name']} | {sel['threshold']:.3f} (val, hedef duyarlılık "
        f"{metrics['target_sensitivity']:.2f}) | {_fmt(sel['auroc'])} | {_fmt(sel['sensitivity'])} "
        f"| {_fmt(sel['specificity'])} | {cal['test_ece_after']:.3f} |",
    ]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Son değerlendirme ve rapor")
    parser.add_argument("--config", default="configs/effv2s_fanconic.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--final", action="store_true", help="Test kümesini bir kez ölç")
    parser.add_argument("--rerun-final", action="store_true")
    parser.add_argument("--table", action="store_true", help="metrics.json'dan README tablosu")
    args = parser.parse_args(argv)
    reports = Path(args.reports)

    if args.table:
        print(render_table(json.loads((reports / METRICS_FILE).read_text(encoding="utf-8"))))
        return
    if not args.final:
        parser.error("Test kümesini ölçmek için --final gerekir.")
    result = run_final(args.config, args.checkpoint, reports, args.n_boot, args.rerun_final)
    print(render_table(result))


if __name__ == "__main__":
    main()
