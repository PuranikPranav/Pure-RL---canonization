"""Test-time canonicalization loop.

This is the loop the user described:

    while True:
        action = policy(image)            # greedy by default
        image  = rotate(image, action)
        reward = vlm(image)
        if |reward_t - reward_{t-1}| < tol  for ``patience`` steps in a row:
            break

Implemented in a vectorized way so we can canonicalize a batch of test
images in one call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from .env import ActionSpace, CanonicalizationEnv
from .policy import CanonicalizationPolicy, ImageEncoderPreprocessor
from .reward_model import RewardModel
from .rotation import wrap_angle


@dataclass
class CanonicalizationTrajectory:
    angles: List[float] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    converged_at: int = -1


@torch.no_grad()
def canonicalize(
    env: CanonicalizationEnv,
    policy: CanonicalizationPolicy,
    preprocessor: ImageEncoderPreprocessor,
    reward_model: RewardModel,
    device: torch.device,
    tolerance: float = 0.005,
    patience: int = 50,
    max_steps: int = 400,
    greedy: bool = True,
    reward_threshold: float | None = None,
    threshold_patience: int = 5,
) -> List[CanonicalizationTrajectory]:
    """Run policy on all envs until convergence or ``max_steps``.

    Two stopping criteria, applied per env (whichever fires first):

    1. **Reward-delta convergence.** ``|r_t - r_{t-1}| < tolerance`` for
       ``patience`` consecutive steps -- the user's "score doesn't change
       between iterations" criterion.
    2. **Reward threshold (optional).** ``r_t >= reward_threshold`` for
       ``threshold_patience`` consecutive steps -- a "the image is
       upright enough, stop" criterion. Disabled when ``reward_threshold``
       is ``None``.
    """
    n = env.num_envs
    trajectories = [CanonicalizationTrajectory() for _ in range(n)]
    last_reward = np.full(n, np.nan, dtype=np.float32)
    delta_counter = np.zeros(n, dtype=np.int32)
    high_counter = np.zeros(n, dtype=np.int32)
    converged = np.zeros(n, dtype=bool)

    policy.eval()
    for t in range(max_steps):
        obs = env.observe()
        pixel = preprocessor(obs).to(device, non_blocking=True)
        action, _, _ = policy.act(pixel, greedy=greedy)

        angles = env.current_angles()
        try:
            image_ids = env.current_image_ids()
            rewards = reward_model.score(obs, angles=angles, image_ids=image_ids)
        except TypeError:
            rewards = reward_model.score(obs, angles=angles)

        # Update trajectories (only for envs that haven't converged)
        for i in range(n):
            if converged[i]:
                continue
            trajectories[i].angles.append(float(angles[i]))
            trajectories[i].rewards.append(float(rewards[i]))
            trajectories[i].actions.append(int(action[i].item()))

        # Convergence checks
        if t > 0:
            delta = np.abs(rewards - last_reward)
            delta_counter = np.where(delta < tolerance, delta_counter + 1, 0)
            delta_done = delta_counter >= patience
        else:
            delta_done = np.zeros(n, dtype=bool)

        if reward_threshold is not None:
            high_counter = np.where(rewards >= reward_threshold, high_counter + 1, 0)
            threshold_done = high_counter >= threshold_patience
        else:
            threshold_done = np.zeros(n, dtype=bool)

        newly_converged = (delta_done | threshold_done) & (~converged)
        for i in np.where(newly_converged)[0]:
            converged[i] = True
            trajectories[i].converged_at = t
        last_reward = rewards

        if converged.all():
            break

        env.step(action.cpu().numpy(), reward=rewards)

    # For envs that didn't converge, mark with -1.
    for i in range(n):
        if trajectories[i].converged_at == -1 and not converged[i]:
            pass  # leave -1
    return trajectories


def summarize(trajectories: List[CanonicalizationTrajectory]) -> Dict[str, float]:
    finals = []
    converged = []
    initials = []
    steps = []
    for tr in trajectories:
        if not tr.angles:
            continue
        initials.append(abs(wrap_angle(tr.angles[0])))
        finals.append(abs(wrap_angle(tr.angles[-1])))
        converged.append(int(tr.converged_at != -1))
        steps.append(len(tr.angles))
    return {
        "n": float(len(trajectories)),
        "initial_abs_angle_mean": float(np.mean(initials)) if initials else float("nan"),
        "final_abs_angle_mean": float(np.mean(finals)) if finals else float("nan"),
        "convergence_rate": float(np.mean(converged)) if converged else 0.0,
        "mean_steps": float(np.mean(steps)) if steps else float("nan"),
    }
