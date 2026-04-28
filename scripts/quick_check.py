"""Local sanity check that doesn't need the VLM or a GPU.

Verifies the rotation utility, environment dynamics, action space, the
synthetic reward, and a single forward/backward pass through the policy
on a tiny pool of solid-color images. Use this to confirm the codebase
imports and runs end-to-end before doing anything heavy.

    python scripts/quick_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.dataset import ImagePool  # noqa: E402
from src.env import ActionSpace, CanonicalizationEnv  # noqa: E402
from src.policy import CanonicalizationPolicy, ImageEncoderPreprocessor  # noqa: E402
from src.reward_model import SyntheticRewardModel  # noqa: E402
from src.rotation import rotate_image, wrap_angle  # noqa: E402


def fake_pool(n: int = 4, size: int = 224) -> ImagePool:
    """Build a small set of synthetic images with strong vertical structure."""
    images = []
    rng = np.random.default_rng(0)
    for _ in range(n):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for c in range(3):
            img[..., c] = np.linspace(0, 255, size, dtype=np.uint8)[None, :]
        # Add a colored vertical bar -- a strong "up" cue.
        bar_x = rng.integers(size // 4, 3 * size // 4)
        bar_color = rng.integers(0, 255, size=3)
        img[:, bar_x - 5 : bar_x + 5] = bar_color
        images.append(img)
    return ImagePool(np.stack(images, axis=0))


def main() -> None:
    print("[check] rotation utility...")
    pool = fake_pool()
    rotated = rotate_image(pool.get(0), 45)
    assert rotated.shape == pool.get(0).shape

    print("[check] wrap_angle...")
    assert wrap_angle(370) == 10
    assert wrap_angle(-190) == 170
    assert wrap_angle(180) == 180

    print("[check] env dynamics...")
    aspace = ActionSpace(bound=5, step_size=1)
    assert aspace.n == 11
    env = CanonicalizationEnv(
        pool=pool, num_envs=2, action_space=aspace,
        init_rot_max=30.0, max_episode_steps=10, seed=0,
    )
    obs = env.observe()
    assert obs.shape == (2, 224, 224, 3) and obs.dtype == np.uint8

    print("[check] synthetic reward...")
    rmodel = SyntheticRewardModel(use_cosine=True)
    r = rmodel.score(obs, angles=env.current_angles())
    assert r.shape == (2,) and r.dtype == np.float32 and (-1 <= r).all() and (r <= 1).all()

    print("[check] env step...")
    next_obs, done, ang = env.step(np.array([0, 10]) % aspace.n, reward=r)
    assert next_obs.shape == obs.shape

    print("[check] policy forward/backward (small backbone, CPU-friendly)...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    policy = CanonicalizationPolicy(
        backbone_name="facebook/dinov2-small",
        num_actions=aspace.n,
        hidden_dim=128,
        freeze_backbone=True,
    ).to(device)
    pre = ImageEncoderPreprocessor("facebook/dinov2-small")
    px = pre(obs).to(device)
    logits, value = policy(px)
    assert logits.shape == (2, aspace.n)
    assert value.shape == (2,)
    a, lp, v = policy.act(px, greedy=False)
    log_prob, entropy, value2 = policy.evaluate_actions(px, a)
    loss = -log_prob.mean() + 0.5 * value2.pow(2).mean()
    loss.backward()

    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[check] policy trainable params: {n_train/1e6:.2f}M")
    print("[check] ALL OK")


if __name__ == "__main__":
    main()
