"""Render training curves and the baseline-comparison bars to PNG.

Reads:
* the **TensorBoard event files** written by ``src/utils.Logger`` during
  ``scripts/train.py`` (one or more directories under ``logs/<exp>/``).
* the **JSON** written by ``scripts/compare_baselines.py``.

Writes everything to ``--out_dir`` so you can ``scp`` the directory off
Gilbreth and drop the figures straight into a report.

Usage
-----
    python scripts/plot_results.py \
        --log_dir logs/canon_ppo_combined \
        --compare_json results/compare_results.json \
        --out_dir results/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")     # headless: required on a SLURM compute node
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402


# ---------------------------------------------------------------------------
# TensorBoard scalar reader
# ---------------------------------------------------------------------------

def load_tb_scalars(log_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return ``{tag: (steps, values)}`` for every scalar in ``log_dir``."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=np.int64)
        vals = np.array([e.value for e in events], dtype=np.float64)
        out[tag] = (steps, vals)
    return out


def _maybe_plot(ax, scalars, tag, label=None, **kw):
    if tag in scalars:
        s, v = scalars[tag]
        ax.plot(s, v, label=label or tag.split("/")[-1], **kw)
        return True
    return False


# ---------------------------------------------------------------------------
# Plot: training curves
# ---------------------------------------------------------------------------

def plot_training(log_dir: Path, out_dir: Path) -> None:
    scalars = load_tb_scalars(log_dir)
    if not scalars:
        print(f"[plot] no scalars found under {log_dir}; skipping training plot.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) reward decomposition --------------------------------------------------
    ax = axes[0, 0]
    _maybe_plot(ax, scalars, "rollout/reward_mean", "total", lw=2.0)
    _maybe_plot(ax, scalars, "rollout/r_vlm_mean", "vlm only", alpha=0.85)
    _maybe_plot(ax, scalars, "rollout/r_cos_mean", "cos shaping", alpha=0.85, ls="--")
    _maybe_plot(ax, scalars, "rollout/r_progress_mean", "progress shaping", alpha=0.85, ls=":")
    ax.set_title("Reward decomposition (rollout)")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("reward")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) angle progress --------------------------------------------------------
    ax = axes[0, 1]
    _maybe_plot(ax, scalars, "rollout/abs_angle_first", "first step")
    _maybe_plot(ax, scalars, "rollout/abs_angle_mean", "rollout mean")
    _maybe_plot(ax, scalars, "rollout/abs_angle_final", "final step")
    ax.set_title("Absolute residual angle (deg)")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("|angle|")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) PPO health -----------------------------------------------------------
    ax = axes[1, 0]
    _maybe_plot(ax, scalars, "ppo/approx_kl", "approx KL")
    _maybe_plot(ax, scalars, "ppo/clip_frac", "clip fraction")
    _maybe_plot(ax, scalars, "ppo/explained_var", "explained var")
    ax.set_title("PPO health")
    ax.set_xlabel("PPO update")
    ax.legend()
    ax.grid(alpha=0.3)

    # (d) eval --------------------------------------------------------------
    ax = axes[1, 1]
    plotted = False
    plotted |= _maybe_plot(ax, scalars, "eval/final_abs_angle_mean", "|angle| final", marker="o")
    plotted |= _maybe_plot(ax, scalars, "eval/reward_mean", "eval reward", marker="s")
    plotted |= _maybe_plot(ax, scalars, "eval/steps_to_solve_mean", "steps to solve", marker="^")
    if not plotted:
        ax.text(0.5, 0.5, "no eval scalars found", transform=ax.transAxes,
                ha="center", va="center", color="gray")
    ax.set_title("Greedy evaluation (every eval_every updates)")
    ax.set_xlabel("PPO update")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"Training curves -- {log_dir.name}", fontsize=14, y=1.02)
    fig.tight_layout()
    out = out_dir / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


# ---------------------------------------------------------------------------
# Plot: baseline comparison
# ---------------------------------------------------------------------------

# Friendly metric labels + whether smaller is better (for axis annotation).
COMPARE_METRICS = [
    ("abs_angle_err_mean", "|angle err| mean (deg, lower=better)", "lower"),
    ("abs_angle_err_median", "|angle err| median (deg, lower=better)", "lower"),
    ("convergence_rate_5deg", "Conv. rate < 5 deg (higher=better)", "higher"),
    ("judge_reward_mean", "Judge VLM reward (higher=better)", "higher"),
    ("steps_mean", "Steps used (n forward passes)", "neutral"),
    ("wall_seconds", "Wall time (s, lower=better)", "lower"),
]


def plot_comparison(json_path: Path, out_dir: Path) -> None:
    if not json_path.exists():
        print(f"[plot] {json_path} missing; skipping comparison plot.")
        return
    rows = json.loads(json_path.read_text())
    if not rows:
        print(f"[plot] {json_path} is empty; skipping comparison plot.")
        return

    methods = list(rows.keys())
    out_dir.mkdir(parents=True, exist_ok=True)

    n_metrics = len(COMPARE_METRICS)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for ax, (key, label, direction) in zip(axes, COMPARE_METRICS):
        vals = [float(rows[m].get(key, np.nan)) for m in methods]
        bars = ax.bar(methods, vals, color="#4c72b0")
        # Highlight PPO if present
        for b, m in zip(bars, methods):
            if m == "ppo":
                b.set_color("#dd8452")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y")
        # Numeric annotation on top of each bar
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Hide leftover axes
    for ax in axes[n_metrics:]:
        ax.axis("off")

    fig.suptitle("PPO vs baselines on a fixed test set", fontsize=14, y=1.02)
    fig.tight_layout()
    out = out_dir / "comparison_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")

    # Also dump a clean CSV for the report.
    import csv
    csv_path = out_dir / "comparison_table.csv"
    keys = list(COMPARE_METRICS)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [k for (k, _, _) in keys])
        for m in methods:
            w.writerow([m] + [rows[m].get(k, "") for (k, _, _) in keys])
    print(f"[plot] wrote {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Path to a TensorBoard log dir (e.g. logs/canon_ppo_combined).")
    parser.add_argument("--compare_json", type=str, default=None,
                        help="Path to the JSON written by scripts/compare_baselines.py.")
    parser.add_argument("--out_dir", type=str, default="results/plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.log_dir:
        plot_training(Path(args.log_dir), out_dir)
    if args.compare_json:
        plot_comparison(Path(args.compare_json), out_dir)
    if not args.log_dir and not args.compare_json:
        print("[plot] nothing to do; pass --log_dir and/or --compare_json")


if __name__ == "__main__":
    main()
