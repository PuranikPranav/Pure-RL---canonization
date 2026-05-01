"""Train the CNN rotation-regressor baseline.

Trains a small ResNet-18 to predict ``(cos theta, sin theta)`` of a
randomly rotated image, *exactly the same image pool* the PPO agent
uses, with the *same* reflect-padded rotation. This makes it a fair
SL counterpart -- any PPO win has to come from the iterative refinement
+ VLM-as-oracle structure, not from a bigger / different dataset.

Usage
-----
    python scripts/train_baseline_cnn.py --config configs/combined.yaml \
        --epochs 30 --output checkpoints/baseline_cnn/cnn.pt

The output checkpoint is a tiny dict ``{"state_dict": ...}`` that
:meth:`CNNRotationRegressor.load` can read back.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.optim import AdamW  # noqa: E402

from src.baselines import CNNRotationRegressor  # noqa: E402
from src.dataset import (  # noqa: E402
    build_combined_pool,
    download_pool_from_hf,
    load_pool_from_dir,
)
from src.rotation import rotate_image  # noqa: E402
from src.utils import get_device, load_config, set_seed  # noqa: E402


def build_pool(cfg):
    data_cfg = cfg["data"]
    if data_cfg.get("combined"):
        cache_dir = Path(data_cfg.get("dir", "data"))
        return build_combined_pool(
            specs=data_cfg["combined"],
            image_size=data_cfg["image_size"],
            base_dir=cache_dir,
            seed=cfg["experiment"]["seed"],
            shuffle=True,
        )
    data_dir = Path(data_cfg["dir"])
    if data_dir.exists() and len(list(data_dir.glob("*.*"))) >= data_cfg["num_images"]:
        return load_pool_from_dir(
            data_dir, image_size=data_cfg["image_size"], max_images=data_cfg["num_images"]
        )
    return download_pool_from_hf(
        hf_dataset=data_cfg["hf_dataset"],
        split=data_cfg["hf_split"],
        num_images=data_cfg["num_images"],
        image_size=data_cfg["image_size"],
        out_dir=data_dir,
        seed=cfg["experiment"]["seed"],
        hf_config=data_cfg.get("hf_config"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs/combined.yaml"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--init_rot_max", type=float, default=None,
        help="Override training rotation range; default = data.init_rot_max from config."
    )
    parser.add_argument(
        "--output", type=str,
        default=str(REPO_ROOT / "checkpoints/baseline_cnn/cnn.pt"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--no_pretrained", action="store_true",
        help="Skip ImageNet pretrained weights (rare; default uses pretrained)."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(args.device)
    print(f"[cnn-baseline] device={device}")

    pool = build_pool(cfg)
    print(f"[cnn-baseline] pool: {len(pool)} x {pool.image_size}^2")

    init_rot_max = float(
        args.init_rot_max if args.init_rot_max is not None else cfg["data"]["init_rot_max"]
    )

    model = CNNRotationRegressor(pretrained=not args.no_pretrained).to(device)
    model._device_cache = device
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[cnn-baseline] params total={n_params/1e6:.2f}M trainable={n_train/1e6:.2f}M")

    optim = AdamW(model.parameters(), lr=args.lr)

    rng = np.random.default_rng(cfg["experiment"]["seed"])
    n = len(pool)
    steps_per_epoch = max(1, n // args.batch_size)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        t0 = time.time()
        for _ in range(steps_per_epoch):
            idx = rng.integers(0, n, size=args.batch_size)
            angles = rng.uniform(-init_rot_max, init_rot_max, size=args.batch_size).astype(np.float32)
            imgs = np.stack(
                [rotate_image(pool.get(int(i)), float(a)) for i, a in zip(idx, angles)],
                axis=0,
            )
            target = torch.tensor(
                np.stack(
                    [np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))],
                    axis=-1,
                ),
                dtype=torch.float32,
                device=device,
            )

            pred = model(imgs)
            loss = F.mse_loss(pred, target)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running.append(loss.item())

        # Eval angle error on a fresh batch
        model.eval()
        with torch.no_grad():
            ev_idx = rng.integers(0, n, size=args.batch_size)
            ev_angles = rng.uniform(-init_rot_max, init_rot_max, size=args.batch_size).astype(np.float32)
            ev_imgs = np.stack(
                [rotate_image(pool.get(int(i)), float(a)) for i, a in zip(ev_idx, ev_angles)],
                axis=0,
            )
            pred = model(ev_imgs).cpu().numpy()
            pred_ang = np.rad2deg(np.arctan2(pred[:, 1], pred[:, 0]))
            err = np.abs(((pred_ang - ev_angles) + 180) % 360 - 180)

        elapsed = time.time() - t0
        print(
            f"[cnn-baseline] epoch={epoch:>3d} "
            f"loss={np.mean(running):.4f} "
            f"abs_angle_err_mean={np.mean(err):.2f} deg "
            f"abs_angle_err_med={np.median(err):.2f} deg "
            f"time={elapsed:.1f}s"
        )

    model.save(out_path)
    print(f"[cnn-baseline] saved -> {out_path}")


if __name__ == "__main__":
    main()
