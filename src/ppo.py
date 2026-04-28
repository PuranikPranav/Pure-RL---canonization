"""PPO trainer for the canonicalization environment.

Pipeline (matches the user's spec):

    for update in range(total_updates):
        rollout = collect_rollout(num_envs, T)        # (T, N) per-step tensors
        advantages = gae(rollout)                      # bootstrap + lambda-return
        flatten + shuffle without replacement
        for epoch in range(K):
            for minibatch in iterate(flat, mb_size):
                clipped surrogate + value loss + entropy bonus
                update policy (and critic)
        log; periodically save & evaluate

Stop conditions:
    - Outer ``total_updates`` cap.
    - Optional ``target_kl`` early-stop *within* an update (skip remaining
      epochs if approx-KL exceeds threshold) -- standard PPO hygiene.

The "test-time" stop condition (reward delta < tol for K consecutive
steps) lives in :mod:`evaluate`, since it's an inference loop, not a
training loop.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from .env import ActionSpace, CanonicalizationEnv
from .policy import CanonicalizationPolicy, ImageEncoderPreprocessor
from .reward_model import RewardModel
from .utils import Logger, RunningMeanStd


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    """A flat batch of transitions, ready for shuffled minibatching.

    All tensors live on CPU until pulled into a minibatch, to keep GPU
    memory usage bounded by minibatch size (important for large backbones).
    """

    pixel_values: torch.Tensor   # (B, 3, H, W) float
    actions: torch.Tensor        # (B,) long
    log_probs: torch.Tensor      # (B,) float
    advantages: torch.Tensor     # (B,) float
    returns: torch.Tensor        # (B,) float
    values: torch.Tensor         # (B,) float
    angles: torch.Tensor         # (B,) float -- ground-truth, for logging

    def __len__(self) -> int:
        return self.actions.numel()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    def __init__(
        self,
        env: CanonicalizationEnv,
        policy: CanonicalizationPolicy,
        preprocessor: ImageEncoderPreprocessor,
        reward_model: RewardModel,
        cfg: Dict,
        device: torch.device,
        logger: Optional[Logger] = None,
    ):
        self.env = env
        self.policy = policy
        self.preprocessor = preprocessor
        self.reward_model = reward_model
        self.cfg = cfg
        self.device = device
        self.logger = logger

        ppo_cfg = cfg["ppo"]
        self.lr = float(ppo_cfg["learning_rate"])
        self.lr_schedule = ppo_cfg.get("lr_schedule", "constant")
        self.total_updates = int(ppo_cfg["total_updates"])
        self.ppo_epochs = int(ppo_cfg["ppo_epochs"])
        self.minibatch_size = int(ppo_cfg["minibatch_size"])
        self.clip_range = float(ppo_cfg["clip_range"])
        self.clip_range_vf = ppo_cfg.get("clip_range_vf")
        self.vf_coef = float(ppo_cfg["vf_coef"])
        self.entropy_coef = float(ppo_cfg["entropy_coef"])
        self.max_grad_norm = float(ppo_cfg["max_grad_norm"])
        self.gamma = float(ppo_cfg["gamma"])
        self.gae_lambda = float(ppo_cfg["gae_lambda"])
        self.target_kl = ppo_cfg.get("target_kl")

        self.normalize_returns = bool(cfg["env"].get("reward_normalization", True))
        self.return_rms = RunningMeanStd()

        # Optimizer over the *trainable* params only.
        params = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = AdamW(params, lr=self.lr)

        self.action_space: ActionSpace = env.action_space
        self.rollout_steps = int(cfg["env"]["rollout_steps"])
        self.num_envs = env.num_envs
        self.update_idx = 0

    # ------------------------------------------------------------- helpers

    def _set_lr(self, frac_remaining: float) -> None:
        if self.lr_schedule == "linear":
            new_lr = self.lr * frac_remaining
        else:
            new_lr = self.lr
        for g in self.optimizer.param_groups:
            g["lr"] = max(new_lr, 1e-6)

    def _preprocess(self, obs_uint8: np.ndarray) -> torch.Tensor:
        """``(N,H,W,3) uint8`` -> ``(N,3,H',W') float`` tensor on device."""
        return self.preprocessor(obs_uint8).to(self.device, non_blocking=True)

    # ------------------------------------------------------------- rollout

    @torch.no_grad()
    def collect_rollout(self) -> Rollout:
        """Run ``rollout_steps`` env-steps across all envs."""
        T, N = self.rollout_steps, self.num_envs

        obs_list: List[torch.Tensor] = []   # T x (N, 3, H, W)
        act_list: List[torch.Tensor] = []   # T x (N,)
        logp_list: List[torch.Tensor] = []  # T x (N,)
        val_list: List[torch.Tensor] = []   # T x (N,)
        rew_list: List[np.ndarray] = []     # T x (N,)
        done_list: List[np.ndarray] = []    # T x (N,)
        ang_list: List[np.ndarray] = []     # T x (N,)

        obs_np = self.env.observe()                                 # (N,H,W,3) u8
        last_reward: Optional[np.ndarray] = None

        for t in range(T):
            pixel = self._preprocess(obs_np)
            self.policy.eval()
            action, log_prob, value = self.policy.act(pixel, greedy=False)
            action = action.to(torch.long)

            # Reward for the *current* obs (before stepping). Pass image
            # ids when supported so VLMRewardModel can apply per-image
            # bias calibration; fall back gracefully for SyntheticReward
            # / SigLIP which don't accept that kwarg.
            angles = self.env.current_angles()
            try:
                image_ids = self.env.current_image_ids()
                r = self.reward_model.score(obs_np, angles=angles, image_ids=image_ids)
            except TypeError:
                r = self.reward_model.score(obs_np, angles=angles)

            obs_list.append(pixel.cpu())
            # Keep a detached CPU long copy to avoid backend-specific dtype/copy issues.
            act_list.append(action.detach().to(torch.long).cpu().clone())
            logp_list.append(log_prob.cpu())
            val_list.append(value.cpu())
            rew_list.append(r)
            ang_list.append(angles)

            next_obs_np, done, _ = self.env.step(action.cpu().numpy(), reward=r)
            done_list.append(done.copy())
            obs_np = next_obs_np
            last_reward = r

        # Bootstrap value for the last obs (after final step).
        with torch.no_grad():
            pixel = self._preprocess(obs_np)
            _, last_value = self.policy.forward(pixel)
        last_value = last_value.cpu()

        # GAE
        rewards = torch.tensor(np.stack(rew_list), dtype=torch.float32)        # (T, N)
        values = torch.stack(val_list)                                          # (T, N)
        dones = torch.tensor(np.stack(done_list), dtype=torch.float32)          # (T, N)

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(N, dtype=torch.float32)
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            advantages[t] = gae
        returns = advantages + values

        if self.normalize_returns:
            self.return_rms.update(returns.numpy())
            returns_normalized = (returns - self.return_rms.mean) / (
                self.return_rms.std + 1e-8
            )
            # Re-derive normalized advantages from the unnormalized ones --
            # a per-batch advantage normalization is applied below as well.
            advantages = returns_normalized - (values - self.return_rms.mean) / (
                self.return_rms.std + 1e-8
            )
            returns = returns_normalized

        # Flatten time x env -> (T*N, ...)
        def _flat(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(-1, *x.shape[2:]) if x.dim() > 2 else x.reshape(-1)

        rollout = Rollout(
            pixel_values=torch.cat(obs_list, dim=0),                     # (T*N,3,H,W)
            actions=torch.cat(act_list, dim=0),                          # (T*N,)
            log_probs=torch.cat(logp_list, dim=0),                       # (T*N,)
            advantages=_flat(advantages),                                # (T*N,)
            returns=_flat(returns),                                      # (T*N,)
            values=_flat(values),                                        # (T*N,)
            angles=torch.tensor(np.stack(ang_list).reshape(-1), dtype=torch.float32),
        )

        # Stash rollout summary for logging.
        adv_for_log = advantages
        ret_for_log = returns
        ang_arr = np.stack(ang_list)
        ang_first = np.mean(np.abs(ang_arr[0]))
        ang_final = np.mean(np.abs(ang_arr[-1]))
        self._last_rollout_metrics = {
            "rollout/reward_mean": float(rewards.mean().item()),
            "rollout/reward_max": float(rewards.max().item()),
            "rollout/reward_min": float(rewards.min().item()),
            "rollout/reward_std": float(rewards.std().item()),
            "rollout/adv_mean": float(adv_for_log.mean().item()),
            "rollout/adv_std": float(adv_for_log.std().item()),
            "rollout/return_mean": float(ret_for_log.mean().item()),
            "rollout/return_std": float(ret_for_log.std().item()),
            "rollout/value_mean": float(values.mean().item()),
            "rollout/abs_angle_first": float(ang_first),
            "rollout/abs_angle_mean": float(np.mean(np.abs(ang_arr))),
            "rollout/abs_angle_final": float(ang_final),
            "rollout/angle_progress": float(ang_first - ang_final),
        }
        return rollout

    # ---------------------------------------------------------------- update

    def update(self, rollout: Rollout) -> Dict[str, float]:
        """Run K epochs of mini-batch PPO updates."""
        device = self.device
        n = len(rollout)
        # Per-batch advantage normalization (PPO standard practice).
        adv = rollout.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = np.arange(n)
        rng = np.random.default_rng()
        stats: Dict[str, List[float]] = {
            "loss/policy": [], "loss/value": [], "loss/entropy": [],
            "loss/total": [], "ppo/clip_frac": [], "ppo/approx_kl": [],
            "ppo/explained_var": [],
        }

        early_stopped = False
        for epoch in range(self.ppo_epochs):
            rng.shuffle(idx)
            self.policy.train()
            kls_this_epoch: List[float] = []

            for start in range(0, n, self.minibatch_size):
                mb = idx[start : start + self.minibatch_size]

                px = rollout.pixel_values[mb].to(device, non_blocking=True)
                # Use blocking transfer + explicit cast on action ids; MPS can be
                # fragile with non-blocking integer transfers.
                a = rollout.actions[mb].to(dtype=torch.long).to(device, non_blocking=False)
                a = a.clamp_(0, self.action_space.n - 1)
                old_logp = rollout.log_probs[mb].to(device, non_blocking=True)
                mb_adv = adv[mb].to(device, non_blocking=True)
                mb_ret = rollout.returns[mb].to(device, non_blocking=True)
                old_val = rollout.values[mb].to(device, non_blocking=True)

                new_logp, entropy, value = self.policy.evaluate_actions(px, a)

                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.clip_range_vf is None:
                    value_loss = F.mse_loss(value, mb_ret)
                else:
                    v_clip = old_val + (value - old_val).clamp(-self.clip_range_vf, self.clip_range_vf)
                    vl1 = (value - mb_ret).pow(2)
                    vl2 = (v_clip - mb_ret).pow(2)
                    value_loss = 0.5 * torch.max(vl1, vl2).mean()

                entropy_loss = -entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_logp - old_logp
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    clip_frac = (torch.abs(ratio - 1) > self.clip_range).float().mean().item()
                    var_y = mb_ret.var().item() + 1e-8
                    explained = 1.0 - (mb_ret - value).var().item() / var_y

                stats["loss/policy"].append(policy_loss.item())
                stats["loss/value"].append(value_loss.item())
                stats["loss/entropy"].append(-entropy_loss.item())
                stats["loss/total"].append(loss.item())
                stats["ppo/clip_frac"].append(clip_frac)
                stats["ppo/approx_kl"].append(approx_kl)
                stats["ppo/explained_var"].append(explained)
                kls_this_epoch.append(approx_kl)

            if self.target_kl is not None and float(np.mean(kls_this_epoch)) > self.target_kl:
                early_stopped = True
                break

        agg = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
        agg["ppo/early_stopped"] = float(early_stopped)
        agg["ppo/lr"] = float(self.optimizer.param_groups[0]["lr"])
        return agg

    # ----------------------------------------------------------------- loop

    def fit(self, save_dir: Optional[str] = None) -> None:
        save_path = Path(save_dir) if save_dir is not None else None
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)

        log_cfg = self.cfg.get("logging", {})
        log_every = int(log_cfg.get("log_every", 1))
        save_every = int(log_cfg.get("save_every", 25))
        eval_every = int(log_cfg.get("eval_every", 25))
        n_eval = int(log_cfg.get("num_eval_episodes", 8))

        for update in range(1, self.total_updates + 1):
            self.update_idx = update
            self._set_lr(1.0 - (update - 1) / max(1, self.total_updates))

            t0 = time.time()
            rollout = self.collect_rollout()
            t_roll = time.time() - t0

            t0 = time.time()
            ppo_stats = self.update(rollout)
            t_upd = time.time() - t0

            metrics = {
                **self._last_rollout_metrics,
                **ppo_stats,
                "time/rollout_s": t_roll,
                "time/update_s": t_upd,
            }

            if update % log_every == 0 and self.logger is not None:
                self.logger.log(metrics, step=update)

            if save_path is not None and update % save_every == 0:
                self.save_checkpoint(save_path / f"policy_update{update:05d}.pt")

            if update % eval_every == 0:
                eval_metrics = self.evaluate_short(n_episodes=n_eval)
                if self.logger is not None:
                    self.logger.log(eval_metrics, step=update, prefix="eval/")

        if save_path is not None:
            self.save_checkpoint(save_path / "policy_final.pt")

    # ---------------------------------------------------------- evaluation

    @torch.no_grad()
    def evaluate_short(self, n_episodes: int = 8, max_steps: int = 64) -> Dict[str, float]:
        """Greedy rollout for diagnostics. Returns mean final |angle|."""
        # Use the live env: just run greedy for a few episodes.
        env = self.env
        self.policy.eval()
        finals: List[float] = []
        rewards: List[float] = []
        steps_to_solve: List[int] = []

        # Reset all envs and run a single batch of greedy rollouts.
        env.reset_all()
        episode_done_step = [-1] * env.num_envs
        for t in range(max_steps):
            obs = env.observe()
            pixel = self._preprocess(obs)
            action, _, _ = self.policy.act(pixel, greedy=True)
            ang = env.current_angles()
            try:
                image_ids = env.current_image_ids()
                r = self.reward_model.score(obs, angles=ang, image_ids=image_ids)
            except TypeError:
                r = self.reward_model.score(obs, angles=ang)
            rewards.append(float(r.mean()))
            obs2, done, _ = env.step(action.cpu().numpy(), reward=r)
            for i, d in enumerate(done):
                if d and episode_done_step[i] == -1:
                    episode_done_step[i] = t
                    finals.append(float(abs(ang[i])))
            if all(s != -1 for s in episode_done_step):
                break

        if not finals:
            finals = [float(abs(a)) for a in env.current_angles()]
        steps_to_solve = [s for s in episode_done_step if s != -1]

        return {
            "final_abs_angle_mean": float(np.mean(finals)) if finals else float("nan"),
            "reward_mean": float(np.mean(rewards)) if rewards else float("nan"),
            "steps_to_solve_mean": float(np.mean(steps_to_solve)) if steps_to_solve else float("nan"),
        }

    # ------------------------------------------------------------- save / load

    def _trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return only the parameters that have ``requires_grad=True``.

        For a frozen backbone, this is just the actor/critic heads -- in our
        ViT-Huge config that's ~5 MB worth of weights vs ~2.5 GB for the full
        state dict. The frozen backbone is reproduced exactly at load time by
        calling ``AutoModel.from_pretrained(backbone_name)`` again, which is
        deterministic given a fixed HF revision.
        """
        trainable_names = {
            n for n, p in self.policy.named_parameters() if p.requires_grad
        }
        full = self.policy.state_dict()
        return {k: v for k, v in full.items() if k in trainable_names}

    def save_checkpoint(self, path: str | Path, slim: Optional[bool] = None) -> None:
        """Save a checkpoint.

        Parameters
        ----------
        path : str | Path
        slim : bool | None
            If ``None`` (default), automatically uses slim mode whenever the
            backbone is frozen -- there's no information loss in that case
            because the backbone is reloaded from HuggingFace at load time.
            If ``False``, always save the full state_dict (useful if you've
            unfrozen the backbone partway through training).
        """
        if slim is None:
            slim = bool(self.policy.freeze_backbone)

        if slim:
            policy_state = self._trainable_state_dict()
            backbone_name = self.policy.backbone_name
        else:
            policy_state = self.policy.state_dict()
            backbone_name = self.policy.backbone_name

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": policy_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "update_idx": self.update_idx,
                "config": self.cfg,
                "return_rms": {
                    "mean": self.return_rms.mean,
                    "var": self.return_rms.var,
                    "count": self.return_rms.count,
                },
                "slim": slim,
                "backbone_name": backbone_name,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        load_policy_state(self.policy, ckpt)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.update_idx = int(ckpt.get("update_idx", 0))
        rms = ckpt.get("return_rms")
        if rms is not None:
            self.return_rms.mean = float(rms["mean"])
            self.return_rms.var = float(rms["var"])
            self.return_rms.count = float(rms["count"])


# ---------------------------------------------------------------------------
# Module-level helper, also used by ``scripts/test.py``.
# ---------------------------------------------------------------------------

def load_policy_state(policy: CanonicalizationPolicy, ckpt: Dict) -> None:
    """Load policy weights, handling both slim and full checkpoints.

    A slim checkpoint stores only ``requires_grad=True`` parameters; the
    backbone is reproduced by re-instantiating it from its HF name (which
    is what :class:`CanonicalizationPolicy` does in its constructor anyway).
    """
    state = ckpt["policy_state_dict"]
    is_slim = bool(ckpt.get("slim", False))

    saved_backbone = ckpt.get("backbone_name")
    if saved_backbone is not None and saved_backbone != policy.backbone_name:
        print(
            f"[load_policy_state] WARNING: checkpoint backbone "
            f"'{saved_backbone}' differs from current policy backbone "
            f"'{policy.backbone_name}'. Loading anyway with strict=False."
        )

    missing, unexpected = policy.load_state_dict(state, strict=False)
    if not is_slim:
        if missing:
            raise RuntimeError(
                f"Full checkpoint is missing {len(missing)} keys, e.g. {missing[:3]}"
            )
        if unexpected:
            raise RuntimeError(
                f"Full checkpoint has {len(unexpected)} unexpected keys, "
                f"e.g. {unexpected[:3]}"
            )
    else:
        # In slim mode the backbone keys are expected to be missing.
        non_backbone_missing = [k for k in missing if not k.startswith("backbone.")]
        if non_backbone_missing:
            raise RuntimeError(
                f"Slim checkpoint is missing non-backbone keys: "
                f"{non_backbone_missing[:5]}"
            )
        if unexpected:
            raise RuntimeError(
                f"Slim checkpoint has unexpected keys: {unexpected[:5]}"
            )
