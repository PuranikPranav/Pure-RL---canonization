#!/bin/bash
# ============================================================
# One-time interactive setup for the Purdue Gilbreth cluster.
# Run this on the LOGIN NODE before submitting any sbatch job.
#
#   bash scripts/setup_gilbreth.sh
#
# What it does:
#   1. Loads anaconda + cuda modules (adjust for the cluster you're on).
#   2. Creates a conda env named `canon` (override with $CONDA_ENV).
#   3. pip-installs requirements.txt.
#   4. Configures HuggingFace caches to live on $RCAC_SCRATCH so
#      repeated runs don't redownload Qwen2-VL-2B etc, and HOME quotas
#      don't fill up.
#   5. Pre-fetches Chars74K (~51 MB) into data/chars74k/.
#   6. Pre-fetches CIFAR-10 into data/cifar10/.
#   7. Runs scripts/quick_check.py to confirm things import cleanly.
#
# IMPORTANT: this script is *not* meant to be sbatch'd. It uses internet
# from the login node (compute nodes can also reach HF, but the login
# node is faster / no-queue). Once this finishes, submit your training
# job with:
#
#   sbatch scripts/run_gilbreth.sh
# ============================================================

set -euo pipefail

# ---- Tunable knobs -----------------------------------------------------
CONDA_ENV="${CONDA_ENV:-canon}"
PY_VERSION="${PY_VERSION:-3.12}"
# Gilbreth (May 2026) exposes:
#   anaconda/2024.10-py312, anaconda/2025.06-py313
#   cuda/12.1.1, cuda/12.6.0, cuda/13.1.0
# We pin python 3.12 + cuda 12.1.1 because PyTorch's default GPU wheels
# match this combo and Python 3.13 still lacks wheels for some HF deps.
ANACONDA_MOD="${ANACONDA_MOD:-anaconda/2024.10-py312}"
CUDA_MOD="${CUDA_MOD:-cuda/12.1.1}"
CHARS74K_SUBSET="${CHARS74K_SUBSET:-fnt}"
CHARS74K_MAX="${CHARS74K_MAX:-200}"

# ---- Modules -----------------------------------------------------------
# Gilbreth keeps anaconda and cuda inside the `external` Lmod hierarchy,
# so we need to load that meta-module before the actual ones become
# visible. ``module load external`` is idempotent.
echo "[setup] loading modules: external $ANACONDA_MOD $CUDA_MOD"
module --force purge >/dev/null 2>&1 || true
module load external
module load "$ANACONDA_MOD"
module load "$CUDA_MOD"

# ---- HF cache on scratch ----------------------------------------------
# Gilbreth exposes scratch as $CLUSTER_SCRATCH; older docs use $RCAC_SCRATCH;
# fall back to $HOME/scratch on machines that have neither.
SCRATCH_DIR="${CLUSTER_SCRATCH:-${RCAC_SCRATCH:-$HOME/scratch}}"
mkdir -p "$SCRATCH_DIR/hf_cache"
mkdir -p "$SCRATCH_DIR/torch_cache"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$SCRATCH_DIR/torch_cache"

# Persist these so SBATCH jobs see them too.
ENV_FILE="$HOME/.canon_env"
cat > "$ENV_FILE" <<EOF
export HF_HOME="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export TORCH_HOME="$TORCH_HOME"
EOF
echo "[setup] HF/Torch caches at $SCRATCH_DIR (saved to $ENV_FILE)"

# ---- Conda env ---------------------------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "[setup] creating conda env '$CONDA_ENV' (python $PY_VERSION)"
    conda create -y -n "$CONDA_ENV" "python=$PY_VERSION"
else
    echo "[setup] conda env '$CONDA_ENV' already exists"
fi

# Activate
# (use eval-based activation since we are not in an interactive shell init)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ---- Pip install -------------------------------------------------------
echo "[setup] pip install -r requirements.txt"
pip install --upgrade pip
pip install -r requirements.txt

# ---- Data ---------------------------------------------------------------
echo "[setup] pre-fetching Chars74K subset=$CHARS74K_SUBSET (~one-time)"
if [ ! -d "data/chars74k" ] || [ -z "$(ls -A data/chars74k 2>/dev/null)" ]; then
    python scripts/download_chars74k.py \
        --subset "$CHARS74K_SUBSET" \
        --max_images "$CHARS74K_MAX" \
        --output "data/chars74k"
else
    echo "[setup] data/chars74k already populated; skipping (use --force to redo)"
fi

echo "[setup] pre-fetching CIFAR-10 (downloads to data/cifar10 via torchvision)"
python - <<'PY'
import torchvision
torchvision.datasets.CIFAR10(root="data/cifar10", train=True, download=True)
print("CIFAR-10 cached.")
PY

# ---- Sanity check ------------------------------------------------------
echo "[setup] quick_check.py"
python scripts/quick_check.py

echo
echo "==========================================================="
echo "[setup] DONE."
echo "==========================================================="
echo "Next:"
echo "  sbatch scripts/run_gilbreth.sh"
echo
echo "Useful one-offs:"
echo "  python scripts/probe_reward.py --config configs/combined.yaml"
echo "  tensorboard --logdir logs/        (run from login node)"
echo "==========================================================="
