"""Compare PPO against same-size baselines on a fixed test set.

For each test sample (image, initial_angle), every method is asked to
canonicalize starting from the *same* state, and we record:

* ``final_abs_angle_err`` -- |final residual angle| (degrees), wrt the true
  upright. This is the headline metric.
* ``final_reward`` -- the big VLM's reward at the final image (so the
  comparison is judged by the same oracle PPO was trained against).
* ``n_steps`` -- how many model forward passes the method used (1 for
  one-shot baselines).

Methods compared (configurable via ``--methods``):

* ``ppo``               trained PPO policy (loaded from checkpoint)
* ``vlm_bruteforce``    similar-size VLM, scored across an angle grid
* ``vlm_iterative``     similar-size VLM as a step-by-step greedy policy
* ``cnn``               supervised ResNet-18 angle regressor
* ``random``            random actions, sanity floor

Usage
-----
    python scripts/compare_baselines.py \
        --config configs/combined.yaml \
        --ppo_checkpoint checkpoints/canon_ppo_combined/policy_final.pt \
        --cnn_checkpoint checkpoints/baseline_cnn/cnn.pt \
        --num_images 32 --num_inits 4 \
        --output_json compare_results.json

Methods that need a model you don't pass a checkpoint for are skipped
gracefully with a warning, so you can run partial comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.baselines import (  # noqa: E402
    BruteForceVLMBaseline,
    CanonicalizationBaseline,
    CNNRotationRegressor,
    IterativeVLMBaseline,
    RandomBaseline,
    _SmallVLMScorer,
)
from src.dataset import (  # noqa: E402
    build_combined_pool,
    download_pool_from_hf,
    load_pool_from_dir,
)
from src.policy import CanonicalizationPolicy, ImageEncoderPreprocessor  # noqa: E402
from src.ppo import load_policy_state  # noqa: E402
from src.reward_model import build_reward_model  # noqa: E402
from src.rotation import rotate_image, wrap_angle  # noqa: E402
from src.utils import get_device, load_config, set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Pool loader (mirrors train.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PPO baseline (wraps the trained policy as a CanonicalizationBaseline)
# ---------------------------------------------------------------------------

class PPOBaseline(CanonicalizationBaseline):
    """Trained PPO policy run as a step-by-step greedy canonicalizer.

    Mirrors :func:`src.evaluate.canonicalize` but operates on a single
    (image, initial_angle) so the comparison harness can call every
    method through the same API.
    """

    name = "ppo"

    def __init__(
        self,
        policy: CanonicalizationPolicy,
        preprocessor: ImageEncoderPreprocessor,
        device: torch.device,
        action_bound: int = 5,
        action_step: int = 1,
        max_steps: int = 64,
        tolerance: float = 0.005,
        patience: int = 5,
        reward_threshold: float | None = None,
        scorer: _SmallVLMScorer | None = None,
    ):
        self.policy = policy
        self.preprocessor = preprocessor
        self.device = device
        self.action_bound = int(action_bound)
        self.action_step = int(action_step)
        self.max_steps = int(max_steps)
        self.tolerance = float(tolerance)
        self.patience = int(patience)
        self.reward_threshold = reward_threshold
        # Optional secondary scorer for stopping (so PPO at test-time can
        # use the *small* VLM purely for the stopping signal -- avoids
        # the 2B oracle at deployment). If None, we stop on action stillness.
        self.scorer = scorer

    @torch.no_grad()
    def canonicalize(self, image: np.ndarray, initial_angle: float):
        from src.baselines import BaselineTrace
        angle = float(initial_angle)
        trace = BaselineTrace()
        last_r: float | None = None
        delta_count = 0
        high_count = 0

        for step in range(self.max_steps):
            current = rotate_image(image, angle)
            pixel = self.preprocessor(current[None]).to(self.device, non_blocking=True)
            action_idx, _, _ = self.policy.act(pixel, greedy=True)
            a = int(action_idx.item()) - self.action_bound

            if self.scorer is not None:
                r = float(self.scorer.score_pil([Image.fromarray(current)])[0])
            else:
                r = float(np.cos(np.deg2rad(angle)))   # proxy for logging only
            trace.angles.append(angle)
            trace.rewards.append(r)

            if last_r is not None and abs(r - last_r) < self.tolerance:
                delta_count += 1
            else:
                delta_count = 0
            if self.reward_threshold is not None and r >= self.reward_threshold:
                high_count += 1
            else:
                high_count = 0
            if a == 0 or delta_count >= self.patience or (
                self.reward_threshold is not None and high_count >= self.patience
            ):
                return wrap_angle(angle), step + 1, trace
            last_r = r

            angle = wrap_angle(angle + a)

        return wrap_angle(angle), self.max_steps, trace


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def make_test_set(
    pool, num_images: int, num_inits: int, init_rot_max: float, seed: int
) -> List[Tuple[int, float]]:
    """Deterministic (pool_idx, initial_angle) pairs."""
    rng = np.random.default_rng(seed)
    pool_indices = rng.choice(len(pool), size=min(num_images, len(pool)), replace=False)
    pairs: List[Tuple[int, float]] = []
    for pi in pool_indices:
        for _ in range(num_inits):
            ang = float(rng.uniform(-init_rot_max, init_rot_max))
            if abs(ang) < 5.0:
                ang = 5.0 * (np.sign(ang) or 1.0)
            pairs.append((int(pi), ang))
    return pairs


def evaluate_method(
    method: CanonicalizationBaseline,
    pool,
    test_pairs: List[Tuple[int, float]],
    judge,
) -> Dict[str, float]:
    abs_errs: List[float] = []
    rewards: List[float] = []
    steps: List[int] = []
    converged: List[int] = []   # |residual| < 5 deg counts as a "win"
    t0 = time.time()
    for pi, ang in test_pairs:
        image = pool.get(pi)
        residual, n_steps, _ = method.canonicalize(image, ang)
        abs_errs.append(abs(residual))
        steps.append(n_steps)
        converged.append(int(abs(residual) < 5.0))
        # Judge the final image with the same VLM PPO trained against
        final_image = rotate_image(image, residual)
        rewards.append(float(judge.score(final_image[None])[0]))
    elapsed = time.time() - t0
    n = max(1, len(abs_errs))
    return {
        "n": n,
        "abs_angle_err_mean": float(np.mean(abs_errs)),
        "abs_angle_err_median": float(np.median(abs_errs)),
        "abs_angle_err_p90": float(np.percentile(abs_errs, 90)),
        "convergence_rate_5deg": float(np.mean(converged)),
        "judge_reward_mean": float(np.mean(rewards)),
        "steps_mean": float(np.mean(steps)),
        "wall_seconds": float(elapsed),
    }


def print_table(rows: Dict[str, Dict[str, float]]) -> None:
    keys = [
        "n", "abs_angle_err_mean", "abs_angle_err_median", "abs_angle_err_p90",
        "convergence_rate_5deg", "judge_reward_mean", "steps_mean", "wall_seconds",
    ]
    name_w = max(len(k) for k in rows.keys())
    col_w = 14
    print()
    header = "method".ljust(name_w) + " | " + " | ".join(k[:col_w].ljust(col_w) for k in keys)
    print(header)
    print("-" * len(header))
    for name, m in rows.items():
        line = name.ljust(name_w) + " | " + " | ".join(
            (f"{m[k]:.3f}" if isinstance(m[k], float) else f"{m[k]}").ljust(col_w)
            for k in keys
        )
        print(line)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ppo_checkpoint", type=str, default=None)
    parser.add_argument("--cnn_checkpoint", type=str, default=None)
    parser.add_argument(
        "--small_vlm", type=str, default="openai/clip-vit-base-patch16",
        help="Hugging Face model id for the similar-size VLM baseline (CLIP/SigLIP)."
    )
    parser.add_argument(
        "--methods", type=str, default="ppo,vlm_bruteforce,vlm_iterative,cnn,random",
        help="Comma-separated subset of methods to run."
    )
    parser.add_argument("--num_images", type=int, default=32)
    parser.add_argument("--num_inits", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--init_rot_max", type=float, default=None,
        help="Override the rotation range; defaults to data.init_rot_max from config."
    )
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(args.device)
    print(f"[compare] device={device}")

    pool = build_pool(cfg)
    init_rot_max = float(
        args.init_rot_max if args.init_rot_max is not None else cfg["data"]["init_rot_max"]
    )
    test_pairs = make_test_set(
        pool=pool,
        num_images=args.num_images,
        num_inits=args.num_inits,
        init_rot_max=init_rot_max,
        seed=cfg["experiment"]["seed"] + 7,   # decouple from the training subset RNG
    )
    print(f"[compare] test set: {len(test_pairs)} (image, init_angle) pairs")

    methods_to_run = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows: Dict[str, Dict[str, float]] = {}

    # ---- Build the *judge* (same big VLM PPO trained against) -----
    print("[compare] loading judge reward model ...")
    judge = build_reward_model(cfg["reward"])
    if hasattr(judge, "calibrate"):
        try:
            judge.calibrate(pool.images)
        except Exception as e:
            print(f"[compare] WARNING: judge calibration failed ({e})")

    # ---- Build a small scorer if needed -------------------------------
    need_scorer = any(m in methods_to_run for m in ("vlm_bruteforce", "vlm_iterative"))
    small_scorer: _SmallVLMScorer | None = None
    if need_scorer:
        print(f"[compare] loading similar-size VLM scorer: {args.small_vlm}")
        small_scorer = _SmallVLMScorer(
            model_name=args.small_vlm, device=str(device), dtype="float16",
        )
        print(f"[compare]   scorer params={small_scorer.num_parameters()/1e6:.1f}M")

    # ---- PPO ----------------------------------------------------------
    if "ppo" in methods_to_run:
        if args.ppo_checkpoint is None:
            print("[compare] WARNING: --ppo_checkpoint not provided; skipping ppo")
        else:
            from src.env import ActionSpace
            action_space = ActionSpace(
                bound=cfg["action"]["bound"],
                step_size=cfg["action"]["step_size"],
            )
            pol_cfg = cfg["policy"]
            policy = CanonicalizationPolicy(
                backbone_name=pol_cfg["backbone"],
                num_actions=action_space.n,
                hidden_dim=pol_cfg["hidden_dim"],
                dropout=pol_cfg["dropout"],
                freeze_backbone=pol_cfg["freeze_backbone"],
            ).to(device)
            preprocessor = ImageEncoderPreprocessor(pol_cfg["backbone"])
            ckpt = torch.load(args.ppo_checkpoint, map_location=device)
            load_policy_state(policy, ckpt)
            policy.eval()
            print(
                f"[compare] PPO loaded (slim={ckpt.get('slim', False)}) from {args.ppo_checkpoint}"
            )
            ppo_baseline = PPOBaseline(
                policy=policy, preprocessor=preprocessor, device=device,
                action_bound=int(cfg["action"]["bound"]),
                action_step=int(cfg["action"]["step_size"]),
                max_steps=args.max_steps,
                tolerance=float(cfg["inference"]["tolerance"]),
                patience=int(cfg["inference"].get("patience", 5)),
                reward_threshold=cfg["inference"].get("reward_threshold"),
                scorer=small_scorer,
            )
            rows["ppo"] = evaluate_method(ppo_baseline, pool, test_pairs, judge)
            print(f"[compare] ppo  -> {rows['ppo']}")

    # ---- vlm_bruteforce -----------------------------------------------
    if "vlm_bruteforce" in methods_to_run and small_scorer is not None:
        bl = BruteForceVLMBaseline(scorer=small_scorer)
        rows["vlm_bruteforce"] = evaluate_method(bl, pool, test_pairs, judge)
        print(f"[compare] vlm_bruteforce -> {rows['vlm_bruteforce']}")

    # ---- vlm_iterative ------------------------------------------------
    if "vlm_iterative" in methods_to_run and small_scorer is not None:
        bl = IterativeVLMBaseline(
            scorer=small_scorer,
            action_bound=int(cfg["action"]["bound"]),
            action_step=int(cfg["action"]["step_size"]),
            max_steps=args.max_steps,
            tolerance=float(cfg["inference"]["tolerance"]),
            patience=int(cfg["inference"].get("patience", 5)),
            reward_threshold=cfg["inference"].get("reward_threshold"),
        )
        rows["vlm_iterative"] = evaluate_method(bl, pool, test_pairs, judge)
        print(f"[compare] vlm_iterative -> {rows['vlm_iterative']}")

    # ---- cnn ----------------------------------------------------------
    if "cnn" in methods_to_run:
        if args.cnn_checkpoint is None:
            print("[compare] WARNING: --cnn_checkpoint not provided; skipping cnn")
        else:
            cnn = CNNRotationRegressor.load(args.cnn_checkpoint, device=device)
            n_p = sum(p.numel() for p in cnn.parameters())
            print(f"[compare] CNN baseline loaded: {n_p/1e6:.1f}M params")
            rows["cnn"] = evaluate_method(cnn, pool, test_pairs, judge)
            print(f"[compare] cnn -> {rows['cnn']}")

    # ---- random -------------------------------------------------------
    if "random" in methods_to_run:
        rb = RandomBaseline(
            action_bound=int(cfg["action"]["bound"]),
            action_step=int(cfg["action"]["step_size"]),
            max_steps=args.max_steps,
            seed=cfg["experiment"]["seed"],
        )
        rows["random"] = evaluate_method(rb, pool, test_pairs, judge)
        print(f"[compare] random -> {rows['random']}")

    # ---- Summary ------------------------------------------------------
    print_table(rows)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))
        print(f"[compare] saved -> {out}")


if __name__ == "__main__":
    main()
