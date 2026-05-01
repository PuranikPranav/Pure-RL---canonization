"""Sanity-probe the reward model on a rotation sweep.

Why: the configured reward model (VLM, SigLIP, or synthetic) is the
*only* thing telling PPO what "upright" means. If it is silently broken
on your dataset -- e.g. the VLM is OOD on tiny EMNIST characters and
gives nearly random scores -- training will look like it's progressing
on the loss curve but the policy will never converge.

This script:

  1. Builds the same image pool ``train.py`` would build.
  2. Calibrates the reward model on the upright pool (if supported).
  3. For ``num_images`` images, rotates each by an angle sweep
     (default ``0..180`` deg), scores every (image, angle) pair, and
     prints a per-image table.
  4. Computes a *monotonicity score* per image: the fraction of
     adjacent angle pairs whose rewards decrease as ``|angle|`` grows.
     A healthy reward model is monotonic for almost all images.

If the average monotonicity is below ~0.7, you should worry about the
reward (consider switching reward.type, dropping the OOD subset, or
boosting the shaping coefficients in ``ppo.shaping``).

Usage
-----
    python scripts/probe_reward.py --config configs/combined.yaml \
        --num_images 8 \
        --angles 0,15,30,45,60,90,120,150,180

Run it once before each new training experiment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.dataset import (  # noqa: E402
    build_combined_pool,
    download_pool_from_hf,
    load_pool_from_dir,
)
from src.reward_model import build_reward_model  # noqa: E402
from src.rotation import rotate_image  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402


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


def parse_angles(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument(
        "--angles", type=str, default="0,15,30,45,60,90,120,150,180",
        help="Comma-separated rotation sweep, in degrees. Symmetric in sign \
internally (we evaluate both +a and -a).",
    )
    parser.add_argument("--no_calibrate", action="store_true",
                        help="Skip reward-model calibration (debug only).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])

    pool = build_pool(cfg)
    print(f"[probe] pool: {len(pool)} x {pool.image_size}^2")

    reward = build_reward_model(cfg["reward"])
    if hasattr(reward, "calibrate") and not args.no_calibrate:
        try:
            reward.calibrate(pool.images)
        except Exception as e:
            print(f"[probe] WARNING: calibration failed ({e})")

    angles = parse_angles(args.angles)
    angles_signed: List[float] = []
    for a in angles:
        angles_signed.append(a)
        if a != 0.0 and -a not in angles_signed:
            angles_signed.append(-a)
    angles_sorted = sorted(angles_signed, key=lambda x: (abs(x), x))

    n = min(args.num_images, len(pool))
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    indices = rng.choice(len(pool), size=n, replace=False).tolist()

    # Score (n_images x n_angles) in batches of all-angles-per-image.
    print()
    header = "img"
    for a in angles_sorted:
        header += f" | {a:>+6.1f}"
    header += " |  mono"
    print(header)
    print("-" * len(header))

    monotonic_scores: List[float] = []
    upright_scores: List[float] = []
    worst_offset: List[float] = []   # angle (deg) at which the reward peaks (should be 0)

    for img_no, idx in enumerate(indices):
        src = pool.get(int(idx))
        rotated = np.stack([rotate_image(src, a) for a in angles_sorted], axis=0)
        try:
            rewards = reward.score(
                rotated,
                angles=np.asarray(angles_sorted, dtype=np.float32),
                image_ids=np.full(len(angles_sorted), int(idx), dtype=np.int64),
            )
        except TypeError:
            rewards = reward.score(rotated, angles=np.asarray(angles_sorted, dtype=np.float32))
        rewards = np.asarray(rewards, dtype=np.float32)

        # Monotonicity: split sorted angles into the negative-side and the
        # positive-side, both moving away from 0. Within each side, reward
        # should decrease as |angle| increases.
        zero_idx = angles_sorted.index(0.0) if 0.0 in angles_sorted else None
        ok = 0
        total = 0
        if zero_idx is not None:
            # Walk left from 0
            for j in range(zero_idx, 0, -1):
                total += 1
                if rewards[j - 1] < rewards[j]:
                    ok += 1
            # Walk right from 0
            for j in range(zero_idx, len(angles_sorted) - 1):
                total += 1
                if rewards[j + 1] < rewards[j]:
                    ok += 1
        mono = (ok / total) if total else float("nan")
        monotonic_scores.append(mono)

        upright = rewards[zero_idx] if zero_idx is not None else float(rewards.max())
        upright_scores.append(float(upright))
        peak_idx = int(np.argmax(rewards))
        worst_offset.append(float(angles_sorted[peak_idx]))

        line = f"{img_no:>3d}"
        for r in rewards:
            line += f" | {r:+.3f}"
        line += f" |  {mono:>4.2f}"
        print(line)

    print()
    mono_mean = float(np.mean(monotonic_scores))
    upright_mean = float(np.mean(upright_scores))
    peak_at_zero = float(np.mean([1.0 if abs(p) < 1e-6 else 0.0 for p in worst_offset]))
    health = (
        "GOOD" if mono_mean >= 0.85 and peak_at_zero >= 0.85 else
        ("OK"  if mono_mean >= 0.70 and peak_at_zero >= 0.60 else "POOR")
    )
    print(f"[probe] summary")
    print(f"          monotonicity_mean       = {mono_mean:.2f}   (>=0.85 healthy)")
    print(f"          peak_at_0deg_rate       = {peak_at_zero:.2f}   (>=0.85 healthy)")
    print(f"          upright_reward_mean     = {upright_mean:+.3f}")
    print(f"          health verdict          : {health}")
    print()
    if health == "POOR":
        print(
            "[probe] The reward model gives very weak / non-monotonic signal on this\n"
            "        pool. Recommended actions:\n"
            "         1. Crank up `ppo.shaping.cos_alpha` (e.g. 0.25 -> 0.5).\n"
            "         2. Try `reward.type: siglip` -- often smoother for this task.\n"
            "         3. Drop OOD subsets (e.g. EMNIST chars at 224^2 are OOD for\n"
            "            Qwen2-VL) or replace with real Chars74K via\n"
            "            `data.combined: [{source: dir, path: data/chars74k, ...}]`."
        )


if __name__ == "__main__":
    main()
