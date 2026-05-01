#!/bin/bash
# ============================================================
# Full canonicalization-via-PPO pipeline as a SLURM job.
#
# Submit with:
#
#     sbatch scripts/run_gilbreth.sh
#
# Or to run a specific phase only:
#
#     sbatch scripts/run_gilbreth.sh probe
#     sbatch scripts/run_gilbreth.sh train
#     sbatch scripts/run_gilbreth.sh cnn
#     sbatch scripts/run_gilbreth.sh compare
#     sbatch scripts/run_gilbreth.sh plot
#
# Default (no arg) runs them in order: probe -> train -> cnn -> compare -> plot.
# Each phase is idempotent enough to re-run on its own (it skips data
# that's already on disk).
#
# IMPORTANT: edit the SBATCH header below if your account / partition
# differs. The defaults assume Gilbreth A100 GPUs.
# ============================================================

#SBATCH --job-name=canon_ppo
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
# Gilbreth `liu334` allocation -- 2x A100-40GB. Override at submit time
# with e.g. `sbatch --account=foo --partition=foo scripts/run_gilbreth.sh`
# if you want to use a different queue.
#SBATCH --account=liu334
#SBATCH --partition=liu334

set -euo pipefail

# ---- Where to find things --------------------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"

mkdir -p slurm_logs results/plots checkpoints logs

# ---- Modules + env (mirrors setup_gilbreth.sh) ----------------------
ANACONDA_MOD="${ANACONDA_MOD:-anaconda/2024.10-py312}"
CUDA_MOD="${CUDA_MOD:-cuda/12.1.1}"
CONDA_ENV="${CONDA_ENV:-canon}"
CONFIG="${CANON_CONFIG:-configs/combined.yaml}"

module --force purge >/dev/null 2>&1 || true
module load external           # exposes anaconda/cuda hierarchy on Gilbreth
module load "$ANACONDA_MOD"
module load "$CUDA_MOD"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Restore HF / torch caches set up on the login node.
[ -f "$HOME/.canon_env" ] && source "$HOME/.canon_env"

echo "[run] node          = $(hostname)"
echo "[run] gpu           = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo NONE)"
echo "[run] python        = $(which python)"
echo "[run] config        = $CONFIG"
echo "[run] HF_HOME       = ${HF_HOME:-unset}"
echo "[run] cwd           = $PWD"

# Print the config so the SLURM log is self-contained.
echo "------ config ------"
cat "$CONFIG"
echo "--------------------"

# ---- Phase implementations ------------------------------------------
phase_probe() {
    echo "==== probe_reward ===="
    python scripts/probe_reward.py --config "$CONFIG" --num_images 8 \
        | tee results/probe_reward.txt
}

phase_data() {
    echo "==== download_data ===="
    if [ ! -d "data/chars74k" ] || [ -z "$(ls -A data/chars74k 2>/dev/null)" ]; then
        python scripts/download_chars74k.py --subset fnt --max_images 200
    else
        echo "[run] data/chars74k already populated; skipping chars74k download."
    fi
    python scripts/download_data.py --config "$CONFIG"
}

phase_train() {
    echo "==== train PPO ===="
    python scripts/train.py --config "$CONFIG"
}

phase_cnn() {
    echo "==== train CNN baseline ===="
    python scripts/train_baseline_cnn.py \
        --config "$CONFIG" \
        --epochs 30 \
        --output checkpoints/baseline_cnn/cnn.pt
}

phase_compare() {
    echo "==== compare baselines ===="
    EXP_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment']['name'])")
    PPO_CKPT="checkpoints/${EXP_NAME}/policy_final.pt"
    CNN_CKPT="checkpoints/baseline_cnn/cnn.pt"

    if [ ! -f "$PPO_CKPT" ]; then
        echo "[run] WARNING: PPO checkpoint $PPO_CKPT missing -- run phase 'train' first."
    fi
    if [ ! -f "$CNN_CKPT" ]; then
        echo "[run] WARNING: CNN checkpoint $CNN_CKPT missing -- run phase 'cnn' first."
    fi

    python scripts/compare_baselines.py \
        --config "$CONFIG" \
        --ppo_checkpoint "$PPO_CKPT" \
        --cnn_checkpoint "$CNN_CKPT" \
        --small_vlm "openai/clip-vit-base-patch16" \
        --num_images 32 --num_inits 4 \
        --output_json results/compare_results.json
}

phase_plot() {
    echo "==== plot results ===="
    EXP_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['experiment']['name'])")
    python scripts/plot_results.py \
        --log_dir "logs/${EXP_NAME}" \
        --compare_json "results/compare_results.json" \
        --out_dir "results/plots"
}

# ---- Dispatch --------------------------------------------------------
PHASE="${1:-all}"
case "$PHASE" in
    probe)   phase_probe ;;
    data)    phase_data ;;
    train)   phase_train ;;
    cnn)     phase_cnn ;;
    compare) phase_compare ;;
    plot)    phase_plot ;;
    all)
        phase_probe
        phase_data
        phase_train
        phase_cnn
        phase_compare
        phase_plot
        ;;
    *)
        echo "[run] unknown phase '$PHASE'. Try one of: probe data train cnn compare plot all"
        exit 2
        ;;
esac

echo
echo "==========================================================="
echo "[run] DONE phase=$PHASE"
echo "  artifacts:"
echo "    results/probe_reward.txt"
echo "    results/compare_results.json"
echo "    results/plots/training_curves.png"
echo "    results/plots/comparison_bars.png"
echo "    results/plots/comparison_table.csv"
echo "  scp these off Gilbreth for your report:"
echo "    scp -r <login>:$(realpath results)  ./local_results/"
echo "==========================================================="
