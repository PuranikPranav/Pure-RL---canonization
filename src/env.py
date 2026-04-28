"""Vectorized canonicalization environment.

We treat each parallel "env" as one (image, current-rotation) pair. All
``num_envs`` envs step in lockstep -- this matches the user's pipeline:
"start with batch of 16 images, each will be sent to a PPO copy and run
trajectories in parallel".

Action space (discrete)
-----------------------
``a in {-bound, -bound+step, ..., +bound}`` -- typical 11 actions for
``bound=5, step=1`` (i.e. -5..5 deg).

State / observation
-------------------
``(num_envs, H, W, 3)`` uint8 numpy array of the *currently rotated*
images. The policy is responsible for any preprocessing (e.g. DINOv2
normalization). We always re-render from the original image at the
current cumulative angle, never compose successive rotations -- this
keeps interpolation noise from drifting across an episode.

Reward
------
Reward is *not* computed here. Computing reward requires the (heavy)
VLM, which we want to keep batched at the rollout level. The collector
calls ``env.observe()`` then ``reward_model.score(obs)`` then
``env.step(action)`` to get next state.

Termination
-----------
- Hard cap: ``max_episode_steps``.
- Optional early termination on a streak of high rewards (configured by
  ``early_terminate``). The streak is fed in by the collector via
  ``step(..., reward=...)``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dataset import ImagePool
from .rotation import rotate_image, wrap_angle


# ---------------------------------------------------------------------------
# Action space helper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionSpace:
    bound: int
    step_size: int

    @property
    def n(self) -> int:
        return 2 * (self.bound // self.step_size) + 1

    def to_degrees(self, action_idx: np.ndarray) -> np.ndarray:
        """Map discrete action index in ``[0, n)`` to signed degrees."""
        return (action_idx.astype(np.int32) - self.bound // self.step_size) * self.step_size


# ---------------------------------------------------------------------------
# Vectorized environment
# ---------------------------------------------------------------------------

class CanonicalizationEnv:
    """Vectorized environment over ``num_envs`` images."""

    def __init__(
        self,
        pool: ImagePool,
        num_envs: int,
        action_space: ActionSpace,
        init_rot_max: float = 90.0,
        max_episode_steps: int = 64,
        early_terminate: bool = False,
        reward_threshold: float = 0.85,
        consecutive_steps: int = 6,
        seed: int = 0,
    ):
        self.pool = pool
        self.num_envs = num_envs
        self.action_space = action_space
        self.init_rot_max = float(init_rot_max)
        self.max_episode_steps = int(max_episode_steps)
        self.early_terminate = bool(early_terminate)
        self.reward_threshold = float(reward_threshold)
        self.consecutive_steps = int(consecutive_steps)

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Per-env state
        self._image_idx = np.zeros(num_envs, dtype=np.int64)
        # initial_angle[i] is the rotation we applied at episode start.
        # current_angle[i] is the *cumulative* angle from the natural orientation
        # (= initial_angle + sum of agent actions). We re-render the original
        # image at current_angle at every step (no compounding interpolation).
        self._initial_angle = np.zeros(num_envs, dtype=np.float32)
        self._current_angle = np.zeros(num_envs, dtype=np.float32)
        self._steps = np.zeros(num_envs, dtype=np.int32)
        self._streak = np.zeros(num_envs, dtype=np.int32)

        # Pre-allocated obs buffer
        H = pool.image_size
        self._obs_buf = np.zeros((num_envs, H, H, 3), dtype=np.uint8)

        self.reset_all()

    # ---------------------------------------------------------------- core

    @property
    def image_size(self) -> int:
        return self.pool.image_size

    def reset_all(self) -> np.ndarray:
        for i in range(self.num_envs):
            self._reset_one(i)
        return self.observe()

    def _reset_one(self, i: int) -> None:
        self._image_idx[i] = self._rng.randrange(len(self.pool))
        # Sample initial angle uniformly in [-init_rot_max, +init_rot_max].
        # We deliberately avoid the trivial case of 0 by enforcing a small
        # minimum magnitude so every episode requires some correction.
        ang = self._np_rng.uniform(-self.init_rot_max, self.init_rot_max)
        if abs(ang) < 5.0:
            ang = 5.0 * np.sign(ang) if ang != 0 else 5.0
        self._initial_angle[i] = ang
        self._current_angle[i] = ang
        self._steps[i] = 0
        self._streak[i] = 0
        self._render(i)

    def _render(self, i: int) -> None:
        src = self.pool.get(int(self._image_idx[i]))
        self._obs_buf[i] = rotate_image(src, float(self._current_angle[i]))

    # -------------------------------------------------------------- public

    def observe(self) -> np.ndarray:
        """Return the current observation buffer (a *view*)."""
        return self._obs_buf

    def current_angles(self) -> np.ndarray:
        """Return cumulative angle from natural orientation (for logging / synthetic reward)."""
        return self._current_angle.copy()

    def current_image_ids(self) -> np.ndarray:
        """Return per-env pool index (used by VLM reward for per-image calibration)."""
        return self._image_idx.copy()

    def step(
        self,
        action_idx: np.ndarray,
        reward: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply ``action_idx`` per env and update rendered observations.

        Parameters
        ----------
        action_idx : np.ndarray, shape (num_envs,), int
            Discrete action index per env.
        reward : np.ndarray | None, shape (num_envs,), float
            Reward associated with the *previous* observation (i.e. before
            this step). Used for early-termination streak tracking. The
            actual reward used by PPO is still managed by the collector.

        Returns
        -------
        next_obs : np.ndarray (num_envs, H, W, 3) uint8
        done     : np.ndarray (num_envs,) bool
        info_angles : np.ndarray (num_envs,) float
            Cumulative angle from natural orientation, *after* the action.
        """
        action_idx = np.asarray(action_idx, dtype=np.int64)
        if action_idx.shape != (self.num_envs,):
            raise ValueError(
                f"action_idx shape {action_idx.shape} != ({self.num_envs},)"
            )
        delta_deg = self.action_space.to_degrees(action_idx).astype(np.float32)

        # Update streak before stepping (uses previous-state reward)
        if reward is not None and self.early_terminate:
            hot = (reward >= self.reward_threshold).astype(np.int32)
            self._streak = (self._streak + 1) * hot  # reset on cold step

        self._current_angle = np.array(
            [wrap_angle(self._current_angle[i] + delta_deg[i]) for i in range(self.num_envs)],
            dtype=np.float32,
        )
        self._steps += 1

        done = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            terminate = self._steps[i] >= self.max_episode_steps
            if (
                self.early_terminate
                and self._streak[i] >= self.consecutive_steps
            ):
                terminate = True
            done[i] = terminate
            if terminate:
                self._reset_one(i)
            else:
                self._render(i)

        return self.observe(), done, self.current_angles()
