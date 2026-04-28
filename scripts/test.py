"""Test-time canonicalization: load a checkpoint, canonicalize images.

Two modes:

1. Batch evaluation on the held-out (or training) image pool:
       python scripts/test.py --config configs/default.yaml \\
           --checkpoint checkpoints/canon_ppo_default/policy_final.pt \\
           --num_images 32

2. Single image canonicalization:
       python scripts/test.py --config configs/default.yaml \\
           --checkpoint checkpoints/canon_ppo_default/policy_final.pt \\
           --image_path path/to/photo.jpg \\
           --initial_angle 73 \\
           --save_trace test_outputs/trace.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.dataset import ImagePool, load_pool_from_dir  # noqa: E402
from src.env import ActionSpace, CanonicalizationEnv  # noqa: E402
from src.evaluate import canonicalize, summarize  # noqa: E402
from src.policy import CanonicalizationPolicy, ImageEncoderPreprocessor  # noqa: E402
from src.ppo import load_policy_state  # noqa: E402
from src.reward_model import build_reward_model  # noqa: E402
from src.rotation import np_to_pil, pil_to_np, rotate_image, square_resize  # noqa: E402
from src.utils import get_device, load_config, set_seed  # noqa: E402


def build_pool(cfg: dict, image_path: str | None, initial_angle: float | None) -> ImagePool:
    """Either load the dataset pool, or build a 1-image pool from a path."""
    if image_path is None:
        data_cfg = cfg["data"]
        return load_pool_from_dir(
            data_cfg["dir"],
            image_size=data_cfg["image_size"],
            max_images=data_cfg["num_images"],
        )
    img = Image.open(image_path).convert("RGB")
    img = square_resize(img, cfg["data"]["image_size"])
    arr = pil_to_np(img)
    return ImagePool(arr[None, ...])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs/default.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--initial_angle", type=float, default=None,
                        help="if set with --image_path, override random init rotation")
    parser.add_argument("--save_trace", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(args.device)

    pool = build_pool(cfg, args.image_path, args.initial_angle)
    num_envs = 1 if args.image_path is not None else min(args.num_images, len(pool))

    action_space = ActionSpace(
        bound=cfg["action"]["bound"],
        step_size=cfg["action"]["step_size"],
    )
    env = CanonicalizationEnv(
        pool=pool,
        num_envs=num_envs,
        action_space=action_space,
        init_rot_max=cfg["data"]["init_rot_max"],
        max_episode_steps=cfg["inference"]["max_steps"],
        early_terminate=False,           # convergence-based, not streak-based
        seed=cfg["experiment"]["seed"],
    )
    if args.image_path is not None and args.initial_angle is not None:
        env._initial_angle[0] = float(args.initial_angle)
        env._current_angle[0] = float(args.initial_angle)
        env._render(0)

    pol_cfg = cfg["policy"]
    policy = CanonicalizationPolicy(
        backbone_name=pol_cfg["backbone"],
        num_actions=action_space.n,
        hidden_dim=pol_cfg["hidden_dim"],
        dropout=pol_cfg["dropout"],
        freeze_backbone=pol_cfg["freeze_backbone"],
    ).to(device)
    preprocessor = ImageEncoderPreprocessor(pol_cfg["backbone"])

    ckpt = torch.load(args.checkpoint, map_location=device)
    load_policy_state(policy, ckpt)
    policy.eval()
    print(
        f"[test] loaded checkpoint (slim={ckpt.get('slim', False)}, "
        f"update={ckpt.get('update_idx', '?')}) from {args.checkpoint}"
    )

    reward_model = build_reward_model(cfg["reward"])

    inf_cfg = cfg["inference"]
    trajectories = canonicalize(
        env=env,
        policy=policy,
        preprocessor=preprocessor,
        reward_model=reward_model,
        device=device,
        tolerance=inf_cfg["tolerance"],
        patience=inf_cfg["patience"],
        max_steps=inf_cfg["max_steps"],
        greedy=inf_cfg["greedy"],
    )

    summary = summarize(trajectories)
    print("[test] summary:", json.dumps(summary, indent=2))

    if args.save_trace is not None:
        out = Path(args.save_trace)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            angles=np.array([tr.angles for tr in trajectories], dtype=object),
            actions=np.array([tr.actions for tr in trajectories], dtype=object),
            rewards=np.array([tr.rewards for tr in trajectories], dtype=object),
            converged_at=np.array([tr.converged_at for tr in trajectories]),
        )
        print(f"[test] saved trace -> {out}")


if __name__ == "__main__":
    main()
