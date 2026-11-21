"""Kaggle fanconic setini indirir. Kaggle hesabı ve `~/.kaggle/kaggle.json` gerekir.

Kullanım:
    python -m skinlesion.download --out data
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

DATASET = "fanconic/skin-cancer-malignant-vs-benign"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Kaggle veri setini indir")
    parser.add_argument("--out", default="data")
    args = parser.parse_args(argv)

    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise SystemExit("Kaggle CLI bulunamadı. Kurulum: pip install kaggle")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [kaggle, "datasets", "download", "-d", DATASET, "-p", str(out), "--unzip"], check=True
    )
    for name in ("train", "test"):
        if not (out / name).is_dir():
            raise SystemExit(f"Beklenen klasör yok: {out / name}. Arşiv düzenini denetleyin.")
    print(f"Veri hazır: {out}")


if __name__ == "__main__":
    main()
