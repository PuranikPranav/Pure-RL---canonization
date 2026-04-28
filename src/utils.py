"""Misc utilities: config loading, seeding, running stats, logging."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg: Dict[str, Any], path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Running mean / std (used for return normalization)
# ---------------------------------------------------------------------------

@dataclass
class RunningMeanStd:
    """Welford-style running mean/var for arbitrary tensors of one dim."""

    mean: float = 0.0
    var: float = 1.0
    count: float = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.size

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        new_var = m2 / tot

        self.mean = float(new_mean)
        self.var = float(new_var)
        self.count = float(tot)

    @property
    def std(self) -> float:
        return float(max(self.var, 1e-8) ** 0.5)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(prefer: str = "auto") -> torch.device:
    pref = (prefer or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Lightweight logger
# ---------------------------------------------------------------------------

class Logger:
    """Console + TensorBoard logger with a pretty per-update dashboard.

    The console output prints a compact multi-line "panel" each update
    that groups the metrics that matter for diagnosing PPO health:

        [PPO update     7]
            reward     mean=+0.183  std=0.215  max=+0.93  min=-0.71
            angle      first=37.4  mean=22.6  final=11.1   progress=+26.3 deg
            policy     adv=+0.001  std=0.998  return=+0.082  value=+0.061
            ppo        kl=0.0042  clip_frac=0.071  expl_var=+0.214  ent=+1.83
            losses     pi=+0.027  v=+0.142  total=+0.196
            opt        lr=2.95e-4  early_stop=0  rollout=18.4s  update=1.7s
            trend      reward(last10)=+0.05/upd  best so far=+0.21

    A short "trend" line tracks the *slope* of recent reward to make
    "is the policy still improving?" obvious at a glance.

    Eval blocks (passed with ``prefix='eval/'``) are printed as a
    separate compact panel so you don't confuse them with rollout stats.
    """

    REWARD_HISTORY = 20  # how many recent updates to use for the trend line

    def __init__(self, log_dir: str | os.PathLike, use_tensorboard: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(str(self.log_dir))
            except ImportError:
                print("[logger] tensorboard not installed, falling back to console only")
        self._reward_history: list[tuple[int, float]] = []
        self._best_reward: float | None = None

    @staticmethod
    def _g(metrics: Dict[str, float], key: str, default: float = float("nan")) -> float:
        v = metrics.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _trend(self, current: float) -> str:
        history = self._reward_history[-self.REWARD_HISTORY :]
        if len(history) < 3:
            return "warming up"
        steps = np.array([s for s, _ in history], dtype=np.float64)
        vals = np.array([v for _, v in history], dtype=np.float64)
        # Slope per update (least squares fit)
        slope, _ = np.polyfit(steps, vals, 1)
        sign = "+" if slope >= 0 else ""
        best = self._best_reward if self._best_reward is not None else current
        return f"reward(last{len(history)})={sign}{slope:.4f}/upd  best so far={best:+.3f}"

    def _log_tb(self, metrics: Dict[str, float], step: int, prefix: str) -> None:
        if self._tb is None:
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._tb.add_scalar(f"{prefix}{k}", float(v), step)

    def _print_eval_panel(self, metrics: Dict[str, float], step: int) -> None:
        g = self._g
        lines = [
            f"[Eval @ update {step:>5d}]",
            f"    final_abs_angle_mean={g(metrics, 'final_abs_angle_mean'):.2f} deg"
            f"   reward_mean={g(metrics, 'reward_mean'):+.3f}"
            f"   steps_to_solve_mean={g(metrics, 'steps_to_solve_mean'):.1f}",
        ]
        print("\n".join(lines), flush=True)

    def _print_rollout_panel(self, metrics: Dict[str, float], step: int) -> None:
        g = self._g
        reward_mean = g(metrics, "rollout/reward_mean")
        if not np.isnan(reward_mean):
            self._reward_history.append((step, reward_mean))
            if self._best_reward is None or reward_mean > self._best_reward:
                self._best_reward = reward_mean

        lines = [
            f"[PPO update {step:>5d}]",
            (
                f"    reward     mean={reward_mean:+.3f}"
                f"  std={g(metrics, 'rollout/reward_std'):.3f}"
                f"  max={g(metrics, 'rollout/reward_max'):+.2f}"
                f"  min={g(metrics, 'rollout/reward_min'):+.2f}"
            ),
            (
                f"    angle      first={g(metrics, 'rollout/abs_angle_first'):.1f}"
                f"  mean={g(metrics, 'rollout/abs_angle_mean'):.1f}"
                f"  final={g(metrics, 'rollout/abs_angle_final'):.1f}"
                f"   progress={g(metrics, 'rollout/angle_progress'):+.1f} deg"
            ),
            (
                f"    policy     adv={g(metrics, 'rollout/adv_mean'):+.3f}"
                f"  std={g(metrics, 'rollout/adv_std'):.3f}"
                f"  return={g(metrics, 'rollout/return_mean'):+.3f}"
                f"  value={g(metrics, 'rollout/value_mean'):+.3f}"
            ),
            (
                f"    ppo        kl={g(metrics, 'ppo/approx_kl'):.4f}"
                f"  clip_frac={g(metrics, 'ppo/clip_frac'):.3f}"
                f"  expl_var={g(metrics, 'ppo/explained_var'):+.3f}"
                f"  ent={g(metrics, 'loss/entropy'):+.3f}"
            ),
            (
                f"    losses     pi={g(metrics, 'loss/policy'):+.3f}"
                f"  v={g(metrics, 'loss/value'):+.3f}"
                f"  total={g(metrics, 'loss/total'):+.3f}"
            ),
            (
                f"    opt        lr={g(metrics, 'ppo/lr'):.2e}"
                f"  early_stop={int(g(metrics, 'ppo/early_stopped', 0))}"
                f"  rollout={g(metrics, 'time/rollout_s'):.1f}s"
                f"  update={g(metrics, 'time/update_s'):.1f}s"
            ),
            f"    trend      {self._trend(reward_mean)}",
        ]
        print("\n".join(lines), flush=True)

    def log(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        # Always mirror to TensorBoard (if available) regardless of prefix.
        self._log_tb(metrics, step, prefix)
        if prefix.startswith("eval"):
            self._print_eval_panel(metrics, step)
        else:
            self._print_rollout_panel(metrics, step)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
