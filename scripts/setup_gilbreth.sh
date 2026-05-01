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

# Never let a stale CONDA_ENVS_DIRS from the shell collide with ~/.condarc.
unset CONDA_ENVS_DIRS

# ---- HF + conda + pip caches on scratch -------------------------------
# Gilbreth exposes scratch as $CLUSTER_SCRATCH; older docs use $RCAC_SCRATCH;
# fall back to $HOME/scratch on machines that have neither. We move *every*
# heavyweight cache here because $HOME has a tight quota (~25-100 GB) and
# torch + Qwen2-VL alone are ~8 GB.
SCRATCH_DIR="${CLUSTER_SCRATCH:-${RCAC_SCRATCH:-$HOME/scratch}}"
mkdir -p "$SCRATCH_DIR/hf_cache" "$SCRATCH_DIR/torch_cache"
mkdir -p "$SCRATCH_DIR/conda_envs" "$SCRATCH_DIR/conda_pkgs" "$SCRATCH_DIR/pip_cache"

export HF_HOME="$SCRATCH_DIR/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$SCRATCH_DIR/torch_cache"
export CONDA_PKGS_DIRS="$SCRATCH_DIR/conda_pkgs"
export PIP_CACHE_DIR="$SCRATCH_DIR/pip_cache"

# Absolute path to the conda env we will create on scratch. We use
# ``conda create --prefix`` instead of ``conda create -n`` so nothing
# ever lands in \$HOME, and we deliberately do *not* set
# ``CONDA_ENVS_DIRS`` -- that env var collides with ``envs_path`` /
# ``envs_dirs`` keys in some users' ~/.condarc and triggers
# ``MultipleKeysError`` on Gilbreth.
ENV_PREFIX="$SCRATCH_DIR/conda_envs/$CONDA_ENV"

# Persist these so SBATCH jobs see them too.
ENV_FILE="$HOME/.canon_env"
cat > "$ENV_FILE" <<EOF
# Drop any stale env-var from a previous attempt that could confuse conda.
unset CONDA_ENVS_DIRS

export HF_HOME="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export TORCH_HOME="$TORCH_HOME"
export CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export CANON_ENV_PREFIX="$ENV_PREFIX"
EOF
echo "[setup] all caches relocated to $SCRATCH_DIR (saved to $ENV_FILE)"

# ---- Conda env (force-create on scratch) -------------------------------
# If a previous attempt landed an env in $HOME (default ~/.conda/envs/...),
# nuke it -- it's almost certainly broken from a quota-exceeded install.
HOME_ENV_DIR="$HOME/.conda/envs/$CONDA_ENV"
HOME_ENV_DIR_NESTED="$HOME/.conda/envs/2024.10-py312/$CONDA_ENV"
for d in "$HOME_ENV_DIR" "$HOME_ENV_DIR_NESTED"; do
    if [ -d "$d" ]; then
        echo "[setup] removing busted env in \$HOME: $d"
        rm -rf "$d"
    fi
done

if [ ! -d "$ENV_PREFIX" ] || [ ! -x "$ENV_PREFIX/bin/python" ]; then
    echo "[setup] creating conda env at $ENV_PREFIX (python $PY_VERSION)"
    rm -rf "$ENV_PREFIX"   # in case a partial dir is hanging around
    conda create -y --prefix "$ENV_PREFIX" "python=$PY_VERSION"
else
    echo "[setup] conda env at $ENV_PREFIX already exists"
fi

# Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# ---- Pip install -------------------------------------------------------
echo "[setup] pip install -r requirements.txt (cache=$PIP_CACHE_DIR)"
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
