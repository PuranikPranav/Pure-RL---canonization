"""Download a small image dataset for canonicalization experiments.

Pulls ``num_images`` images from a HuggingFace image dataset (default
``frgfm/imagenette``) and saves them as ``data/images/img_*.jpg``,
preprocessed to a square ``image_size`` by ``image_size`` RGB. Idempotent:
re-running with the same target dir is a no-op if the directory already
contains the requested number of images.

Usage
-----
    python scripts/download_data.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import download_pool_from_hf  # noqa: E402
from src.utils import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs/default.yaml"))
    parser.add_argument("--force", action="store_true", help="re-download even if data already present")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    out_dir = Path(data_cfg["dir"])

    existing = list(out_dir.glob("*.jp*g")) + list(out_dir.glob("*.png")) if out_dir.exists() else []
    if not args.force and len(existing) >= data_cfg["num_images"]:
        print(f"[download] {len(existing)} images already in {out_dir}; skipping (use --force to redo)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    pool = download_pool_from_hf(
        hf_dataset=data_cfg["hf_dataset"],
        split=data_cfg["hf_split"],
        num_images=data_cfg["num_images"],
        image_size=data_cfg["image_size"],
        out_dir=out_dir,
        seed=cfg["experiment"]["seed"],
        hf_config=data_cfg.get("hf_config"),
    )
    print(f"[download] saved {len(pool)} images to {out_dir} at size {pool.image_size}")


if __name__ == "__main__":
    main()
