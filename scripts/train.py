"""Train the canonicalization PPO policy.

Usage
-----
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/debug.yaml      # synthetic reward, tiny

Resume
------
    python scripts/train.py --config configs/default.yaml \\
        --resume checkpoints/canon_ppo_default/policy_update00050.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.dataset import load_pool_from_dir, download_pool_from_hf, download_combined_pool  # noqa: E402
from src.env import ActionSpace, CanonicalizationEnv  # noqa: E402
from src.policy import CanonicalizationPolicy, ImageEncoderPreprocessor  # noqa: E402
from src.ppo import PPOTrainer  # noqa: E402
from src.reward_model import build_reward_model  # noqa: E402
from src.utils import Logger, get_device, load_config, save_config, set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs/default.yaml"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])

    exp_name = cfg["experiment"]["name"]
    output_dir = Path(cfg["experiment"]["output_dir"]) / exp_name
    log_dir = Path(cfg["experiment"]["log_dir"]) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    device = get_device(args.device)
    print(f"[train] device={device}")

    # ---- Data ----------------------------------------------------------
# Inside main() in train.py:
    data_cfg = cfg["data"]
    
    # Split your total requested images between the two datasets
    half_count = data_cfg["num_images"] // 2
    
    my_datasets = [
        {"name": "Asteriks/chars74k-eng-good", "num": half_count},
        {"name": "cifar10", "num": half_count}
    ]

    # Call the fixed function with the clean variables
    pool = download_combined_pool(
        dataset_configs=my_datasets,
        split="train",
        image_size=data_cfg["image_size"],
        out_dir=data_cfg["dir"],
        seed=cfg["experiment"]["seed"]
    )

    # ---- Env -----------------------------------------------------------
    action_space = ActionSpace(
        bound=cfg["action"]["bound"],
        step_size=cfg["action"]["step_size"],
    )
    env_cfg = cfg["env"]
    env = CanonicalizationEnv(
        pool=pool,
        num_envs=env_cfg["num_envs"],
        action_space=action_space,
        init_rot_max=data_cfg["init_rot_max"],
        max_episode_steps=env_cfg["max_episode_steps"],
        early_terminate=env_cfg["early_terminate"]["enabled"],
        reward_threshold=env_cfg["early_terminate"]["reward_threshold"],
        consecutive_steps=env_cfg["early_terminate"]["consecutive_steps"],
        seed=cfg["experiment"]["seed"],
    )
    print(f"[train] env: num_envs={env.num_envs} action_n={action_space.n}")
    if action_space.n <= 1:
        print("WARNING: Action space too small! Forcing action_n calculation.")
        # This is a fallback if the class properties are failing
        action_n = (cfg["action"]["bound"] * 2) + 1
    else:
        action_n = action_space.n
    # ---- Policy --------------------------------------------------------
    pol_cfg = cfg["policy"]
    policy = CanonicalizationPolicy(
        backbone_name=pol_cfg["backbone"],
        num_actions=action_space.n,
        hidden_dim=pol_cfg["hidden_dim"],
        dropout=pol_cfg["dropout"],
        freeze_backbone=pol_cfg["freeze_backbone"],
    ).to(device)
    preprocessor = ImageEncoderPreprocessor(pol_cfg["backbone"])
    print(
        f"[train] policy: backbone={pol_cfg['backbone']} "
        f"frozen={pol_cfg['freeze_backbone']} "
        f"trainable={policy.trainable_parameter_count()/1e6:.2f}M / "
        f"total={policy.total_parameter_count()/1e6:.2f}M"
    )

    # ---- Reward model --------------------------------------------------
    reward_model = build_reward_model(cfg["reward"])
    rp = reward_model.num_parameters()
    if rp > 0:
        print(f"[train] reward model: {cfg['reward']['type']} ({rp/1e6:.1f}M params)")
    else:
        print(f"[train] reward model: {cfg['reward']['type']} (no params)")

    # Calibrate VLM reward on the *unrotated* pool images so per-image
    # bias is subtracted at runtime. This mostly cancels the VLM's
    # generic "Yes" bias and per-image content bias, leaving rotation as
    # the dominant signal.
    if hasattr(reward_model, "calibrate"):
        try:
            print("[train] calibrating reward model on upright pool images ...")
            reward_model.calibrate(pool.images)
        except Exception as e:
            print(f"[train] WARNING: calibration failed ({e}); continuing without it")

    # ---- Trainer -------------------------------------------------------
    logger = Logger(log_dir, use_tensorboard=cfg["logging"].get("tensorboard", True))
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        preprocessor=preprocessor,
        reward_model=reward_model,
        cfg=cfg,
        device=device,
        logger=logger,
    )
    if args.resume:
        print(f"[train] resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    try:
        trainer.fit(save_dir=str(output_dir))
    finally:
        logger.close()


if __name__ == "__main__":
    main()
