# Canonicalization-via-PPO

Learn a small policy that canonicalizes (un-rotates) an image, using a
larger pretrained Vision-Language Model purely as a *reward oracle*. At
test time the policy is iterated on the live image until the reward
plateaus -- no VLM needed.

The hypothesis being tested:

> If a small (~10-25% of the VLM size) RL-trained policy can canonicalize
> images using the VLM as an oracle during training, then we can amortize
> the VLM's cost: pay it once during training, then use only the small
> policy at deployment.

## Why RL, not Supervised Learning

A supervised classifier learns the *pattern* that distinguishes a rotated
image from an unrotated one. The classic shortcut: **the four black
triangular corners** that appear when you rotate an image with zero
padding. Fill those corners with reflection-padded continuations of the
image (`cv2.BORDER_REFLECT_101`, see `src/rotation.py`) and the visual
shortcut largely disappears -- the SL classifier loses its easy signal.

The RL formulation sidesteps this: the policy doesn't need to *recognize*
"rotated-ness", it just needs to find actions that increase reward. The
VLM-as-oracle does the recognition, and the policy distills its
preference structure into a much smaller model.

## Problem formulation (mapped from your spec)

| Spec component        | Implementation |
| --------------------- | -------------- |
| State                 | Current rotated image (re-rendered each step from the original at the cumulative angle so interpolation noise doesn't compound). |
| Action                | Discrete: `{-5, -4, ..., 4, 5}` degrees. Configurable via `action.bound` and `action.step_size`. |
| Reward                | `tanh(logit("Yes") - logit("No"))` from `Qwen2-VL-2B-Instruct` answering "Is this image upright?". One forward pass, no generation. Range `(-1, 1)`. |
| Trajectory            | `s0, a0, r0, s1, ...` collected from `num_envs=16` parallel images for `T=32` steps each = 512 transitions per rollout. |
| Test-time stop        | `|r_t - r_{t-1}| < tolerance` for `patience=50` consecutive steps (configurable). |

## Layout

```
.
├── configs/
│   ├── default.yaml          # full run with VLM reward (imagenette)
│   ├── combined.yaml         # CIFAR-10 + Chars74K-style combined pool
│   ├── colab_500m.yaml       # ~500M policy class for free Colab T4
│   └── debug.yaml            # tiny, synthetic reward, CPU-friendly
├── src/
│   ├── rotation.py           # smart reflect-padded rotation
│   ├── dataset.py            # image pool: HF + torchvision + dir + combined
│   ├── env.py                # vectorized canonicalization environment
│   ├── policy.py             # actor-critic w/ DINOv2 backbone
│   ├── reward_model.py       # VLM / SigLIP / synthetic (training oracle)
│   ├── baselines.py          # CNN regressor + small-VLM baselines
│   ├── ppo.py                # PPO trainer (GAE, clipping, K epochs)
│   ├── evaluate.py           # convergence-based test-time loop
│   └── utils.py              # config, seeding, running stats, logging
├── scripts/
│   ├── download_data.py      # resolve a single or combined data spec
│   ├── train.py              # PPO training entry point
│   ├── train_baseline_cnn.py # train the SL CNN rotation regressor
│   ├── test.py               # canonicalize at inference time
│   ├── compare_baselines.py  # PPO vs small-VLM vs CNN vs random
│   └── quick_check.py        # local sanity test (no VLM, no GPU needed)
├── requirements.txt
└── README.md
```

## Why a custom PPO, not TRL / VERL?

You suggested TRL or VERL. Both are excellent **for LLM RLHF** -- but
their abstractions are designed around language-model rollouts (prompt
strings, reference models, KL penalties on logits, etc.) and the API
fights you when you try to plug in (a) a vision encoder as the policy,
(b) a custom non-language environment, (c) a VLM-as-reward-oracle that
returns scalars per image. Adapting TRL/VERL would mean either monkey-
patching their `PPOTrainer` to disable LM-specific assumptions or using
~5% of the library while paying full dependency cost.

So `src/ppo.py` is a clean, ~300 line PPO with all the standard tricks:
GAE, clipped surrogate, value loss (with optional clipping), entropy
bonus, K epochs of mini-batch updates without replacement, target-KL
early stopping inside an update, linear LR schedule, gradient clipping,
and running-mean/std return normalization. It's deliberately small so
you can read and modify it.

## Quickstart

### 0. Install

```bash
pip install -r requirements.txt
```

### 1. Sanity check (CPU, no model downloads except a tiny DINOv2-small)

```bash
python scripts/quick_check.py
```
Should print `[check] ALL OK` in under a minute.

### 2. Debug run with synthetic reward (CPU or GPU)

```bash
python scripts/train.py --config configs/debug.yaml
```
This validates the *PPO loop itself* using a perfect oracle reward
(`r = cos(angle)`). If the policy can't drive the angle to zero here, no
VLM is going to save you. Should converge to mean `|angle| < 5°` in
~30 PPO updates.

### 3. Download 100 images

```bash
python scripts/download_data.py --config configs/default.yaml
```
Pulls 100 images from `frgfm/imagenette` (no auth needed) and saves them
under `data/images/`.

### 4. Full training run with VLM reward

```bash
python scripts/train.py --config configs/default.yaml
```

GPU-only in practice: the 2B Qwen2-VL forward pass is what dominates the
rollout cost. Total training is `total_updates * num_envs * rollout_steps =
500 * 16 * 32 = 256 000` VLM forward passes; on an A100 fp16 with batch
16 that's roughly 30-90 minutes depending on the image size.

TensorBoard:

```bash
tensorboard --logdir logs/
```

### 5. Canonicalize at test time

Batched on the training pool:

```bash
python scripts/test.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/canon_ppo_default/policy_final.pt \
    --num_images 32
```

Single image with a chosen initial rotation:

```bash
python scripts/test.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/canon_ppo_default/policy_final.pt \
    --image_path data/images/img_0000.jpg \
    --initial_angle 73 \
    --save_trace outputs/trace.npz
```

The test loop iterates the policy until `|r_t - r_{t-1}| < tolerance`
for `patience=50` consecutive steps -- exactly your "reward score
between two consecutive iterations is less than tolerance" criterion.

## Combined-dataset experiment (CIFAR-10 + Chars74K)

`configs/combined.yaml` builds a concatenated, shuffled pool of two
sources:

| Source | Default | Why |
| ------ | ------- | --- |
| CIFAR-10 (torchvision) | 100 imgs | Natural objects, well-defined upright. |
| **Chars74K** (`data/chars74k/`) | 100 imgs | English alphanumeric characters with a clear upright orientation. Default `fnt` subset = computer-rendered fonts (every image guaranteed upright). |

### One-time setup: fetch Chars74K

Chars74K is **not** on Hugging Face -- the canonical source is Google
Drive links on the [author's page](https://teodecampos.github.io/chars74k/).
We provide a script that downloads, unpacks and flattens it into
`data/chars74k/`:

```bash
pip install gdown
python scripts/download_chars74k.py --subset fnt --max_images 200
```

Subset choices (English script, 62 classes 0-9 / A-Z / a-z):

| `--subset` | Size  | Imgs | Notes |
| ---------- | ----- | ---- | ----- |
| `fnt`      | 51 MB | 63K  | **Default.** Computer fonts -- guaranteed upright -> cleanest reward. |
| `img`      | 128 MB | 7.7K | Natural-scene crops -- closest to the published benchmark spirit, but some characters in the source photos are naturally tilted, which adds reward noise. |
| `hnd`      | 13 MB | 3.4K | Tablet handwriting -- always upright. |

If Google Drive blocks `gdown` (quota), download the tarball yourself
from the author's page and re-run with `--tarball /path/to/EnglishFnt.tgz`.

### Run the experiment

```bash
python scripts/download_chars74k.py --subset fnt --max_images 200      # one-time
python scripts/probe_reward.py     --config configs/combined.yaml      # 30s sanity
python scripts/download_data.py    --config configs/combined.yaml
python scripts/train.py            --config configs/combined.yaml
```

The PPO loop is unchanged: at every step the env re-renders the original
image at the current cumulative angle (reflect-padded), the big VLM scores
it, the policy picks a ±5° action, the rotated image is the next state.

If `download_chars74k.py` is unavailable for any reason, swap the second
entry of `data.combined` to the torchvision EMNIST stand-in (instructions
inside `configs/combined.yaml`); training still works, just with noisier
reward on the character half of the pool.

## Reward shaping (training-time only)

The raw VLM reward `tanh(logit_yes − logit_no)` has two weaknesses for
PPO: it **saturates** near upright (tanh) so the late-stage gradient is
flat, and it is **direction-blind** (a reward of −0.4 doesn't say
"rotate clockwise"). We add two zero-cost auxiliary terms during
training, both controlled by `ppo.shaping` in the YAML:

```yaml
ppo:
  shaping:
    cos_alpha: 0.10        # adds α · cos(angle) -- smooth global gradient
    progress_beta: 0.05    # adds β · max(0, |angle_{t-1}| - |angle_t|)
    vlm_score_every: 1     # >1 to reuse cached VLM score across N steps
```

* `cos_alpha` gives a **smooth, monotonic gradient** that is non-zero
  even when the VLM tanh saturates -- the policy keeps getting useful
  signal in the last few degrees.
* `progress_beta` rewards every step that **reduces** `|angle|`. This is
  direction-aware in a way the VLM scalar isn't, and it discourages
  oscillation (rotate +5, then -5, then +5, ...).
* `vlm_score_every: 2` halves the rollout VLM cost at the price of a
  little staleness; the cos/progress terms supply signal at the cached
  steps too.

**At test time (`scripts/test.py` and `scripts/compare_baselines.py`),
shaping is OFF by construction** -- only the raw VLM reward (and the
delta / threshold stop) is used. Shaping is an *accelerator* for
training, not a part of the deployed canonicalizer.

The dashboard now decomposes the reward each update so you can see the
contributions:

```
[PPO update    37]
    reward     mean=+0.241  std=0.310  max=+0.93  min=-0.55
    shaping    vlm=+0.184  cos=+0.046  progress=+0.011
    angle      first=42.6  mean=21.3  final=8.4   progress=+34.2 deg
    ...
```

## Sanity-probe the reward (run before training)

If the reward model is silently broken on your image pool (e.g. the VLM
is OOD on tiny upscaled characters), training will look fine on losses
but the policy will not converge. Run a 30-second probe first:

```bash
python scripts/probe_reward.py --config configs/combined.yaml \
    --num_images 8 --angles 0,15,30,45,60,90,120,150,180
```

Output is a per-image table of (angle → reward) and an aggregate health
verdict (`GOOD` / `OK` / `POOR`) based on:

* **monotonicity** -- how often does reward decrease as `|angle|` grows?
* **peak_at_0deg_rate** -- does the maximum reward occur at upright?

If the verdict is `POOR`, the script prints concrete remedies (raise
`cos_alpha`, switch reward type, drop OOD subsets, or use real
Chars74K via `source: dir`).

## Test-time stopping criteria

`src/evaluate.canonicalize` supports two stopping rules per env:

1. **Reward delta** -- `|r_t − r_{t−1}| < tolerance` for `patience`
   consecutive steps. The "score doesn't change between iterations"
   criterion from the spec.
2. **Reward threshold** (optional) -- `r_t ≥ reward_threshold` for
   `threshold_patience` consecutive steps. The "the image is upright
   enough, stop" criterion. Set `reward_threshold: null` to disable.

Both live in `inference:` in the config; either can fire and stop the env.

## Baselines & comparison

We compare the trained PPO against three same-size baselines plus a
control. All four implement the same `canonicalize(image, init_angle)`
API (see `src/baselines.py`) so the harness can score them identically.

| Baseline | Type | Approx params | What it tests |
| -------- | ---- | ------------- | ------------- |
| `vlm_bruteforce` | Small VLM (CLIP-base / SigLIP-base) scored across an angle grid | ~150M | Could a similar-size, *off-the-shelf* VLM solve this without learning, given enough compute at test time? |
| `vlm_iterative` | Same VLM, used as 1-step-lookahead policy | ~150M | Same as above but in PPO's loop shape -- isolates the contribution of training. |
| `cnn` | ResNet-18 SL regressor on `(cos θ, sin θ)` | ~11M | Classic SL baseline. Reflect padding is meant to defeat it -- this validates the RL hypothesis. |
| `random` | Uniformly random actions | 0 | Sanity floor. |

**Train the CNN baseline** on the same pool:

```bash
python scripts/train_baseline_cnn.py \
    --config configs/combined.yaml \
    --epochs 30 \
    --output checkpoints/baseline_cnn/cnn.pt
```

**Run the full comparison** on a fixed test set (deterministic seed):

```bash
python scripts/compare_baselines.py \
    --config configs/combined.yaml \
    --ppo_checkpoint checkpoints/canon_ppo_combined/policy_final.pt \
    --cnn_checkpoint checkpoints/baseline_cnn/cnn.pt \
    --small_vlm openai/clip-vit-base-patch16 \
    --num_images 32 --num_inits 4 \
    --output_json compare_results.json
```

Output is a single table, e.g.:

```
method          | n  | abs_angle_err_mean | abs_angle_err_median | convergence_rate_5deg | judge_reward_mean | steps_mean | wall_seconds
ppo             | 128 |       3.4          |        2.0            |        0.84            |      +0.62        |   18.2     |   31.1
vlm_bruteforce  | 128 |       6.1          |        4.0            |        0.66            |      +0.41        |    1.0     |   88.4
vlm_iterative   | 128 |       7.9          |        5.0            |        0.55            |      +0.33        |   42.8     |  154.0
cnn             | 128 |      18.5          |       16.7            |        0.21            |      -0.04        |    1.0     |    2.4
random          | 128 |      55.2          |       60.1            |        0.04            |      -0.18        |   64.0     |    0.3
```

Hypothesis: PPO wins on `abs_angle_err_mean` and `convergence_rate_5deg`
*at deployment cost no greater than the small VLM's*, because all the
oracle compute was paid during training.

## Sizing the "policy = 25% of VLM" experiment

Default reward model: `Qwen/Qwen2-VL-2B-Instruct` (~2.2B params).

To get policy ≈ 25% of that, swap the backbone:

```yaml
policy:
  backbone: "facebook/dinov2-large"   # ~300M
  freeze_backbone: false              # ~310M trainable; ~14% of VLM
```

For ~25%, switch the VLM to a smaller open VLM (e.g.
`Qwen/Qwen2-VL-1.2B-Instruct` if/when available) or pair `dinov2-large`
+ unfreeze + add MLP heads. The point of the architecture is that any
HF `AutoModel` works as the backbone -- only `_infer_feat_dim` matters.

## What lives where in the code

- **Smart rotation** (the SL-defeating trick): `src/rotation.py`,
  `rotate_image` uses `cv2.BORDER_REFLECT_101`.
- **Re-render-from-original** trick: `src/env.py`, see `_render`.
  Avoids interpolation drift across an episode.
- **VLM reward without generation**: `src/reward_model.py`,
  `VLMRewardModel._score_batch`. We `gather` logits at the last
  attended position and contrast `Yes` vs `No` token logits.
- **PPO core**: `src/ppo.py` -- the `update()` method has the clipped
  surrogate, value loss, entropy bonus, target-KL early stop.
- **Test-time convergence**: `src/evaluate.py`, `canonicalize`. Tracks
  per-env reward delta and stops when it's been below `tolerance` for
  `patience` consecutive steps.

## Running on the Purdue Gilbreth cluster

Two scripts handle the full workflow end-to-end:

| Script | Where to run | What it does |
| ------ | ------------ | ------------ |
| `scripts/setup_gilbreth.sh` | **Login node**, interactively, ONCE | Loads `anaconda` + `cuda` modules, creates conda env `canon`, pip-installs deps, redirects HF / Torch caches to `$RCAC_SCRATCH` (saved to `~/.canon_env`), pre-fetches Chars74K + CIFAR-10, runs `quick_check.py`. |
| `scripts/run_gilbreth.sh`   | Submitted via `sbatch`              | Full pipeline: probe -> data -> train PPO -> train CNN baseline -> compare -> plot. Each phase is idempotent and re-runnable individually. |

### One-time setup

```bash
ssh <user>@gilbreth.rcac.purdue.edu
cd $CLUSTER_SCRATCH                           # Gilbreth's scratch env var
git clone -b Original-Pure-RL-codebase \
    https://github.com/PuranikPranav/Pure-RL---canonization.git canon_ppo
cd canon_ppo
bash scripts/setup_gilbreth.sh
```

This takes ~10-15 minutes (mostly pip install + Chars74K download).
Re-running it is safe (skips steps that are already done).

### Submit a full training run

```bash
sbatch scripts/run_gilbreth.sh
```

Default SBATCH header in the script: 1 GPU, 8 CPUs, 48 GB RAM, 6 hours.

**Account vs partition (common Gilbreth confusion):** the first column
of ``slist`` (e.g. ``liu334``) is your **Slurm account** -- who is billed
for the job. The **partition** is which queue you submit into; on
Gilbreth the usual GPU queue is ``gpu``. If ``sbatch`` complains about an
invalid partition, run ``sinfo -s`` or ``slist`` and pick a valid
``PARTITION`` name, then either edit the two ``#SBATCH`` lines in
``scripts/run_gilbreth.sh`` or override at submit time:

```bash
sbatch --account=liu334 --partition=gpu scripts/run_gilbreth.sh
```

To run a subset of phases (each is its own SLURM job):

```bash
sbatch scripts/run_gilbreth.sh probe      # ~30 sec sanity probe
sbatch scripts/run_gilbreth.sh train      # PPO only
sbatch scripts/run_gilbreth.sh cnn        # CNN baseline only
sbatch scripts/run_gilbreth.sh compare    # comparison only (needs ckpts)
sbatch scripts/run_gilbreth.sh plot       # regenerate plots
```

Override the config without editing the script:

```bash
sbatch --export=ALL,CANON_CONFIG=configs/colab_500m.yaml scripts/run_gilbreth.sh train
```

### Outputs

After a successful run, everything you need for the report lives in
`results/`:

```
results/
├── probe_reward.txt          # reward-model health summary
├── compare_results.json      # PPO vs all baselines, machine-readable
└── plots/
    ├── training_curves.png   # reward decomp, angle, PPO health, eval
    ├── comparison_bars.png   # one bar plot per metric, PPO highlighted
    └── comparison_table.csv  # same numbers, paste into a table
```

`scp` them off Gilbreth for your write-up:

```bash
scp -r <user>@gilbreth.rcac.purdue.edu:/scratch/gilbreth/<user>/canon_ppo/results ./local_results
```

The code is single-GPU; multi-GPU is straightforward to bolt on with
`accelerate` (already in `requirements.txt`) but isn't needed for 100
images and a 2B reward model.

## Known caveats / things to validate empirically

1. **Qwen2-VL Yes/No calibration**: the VLM's preference for "Yes" vs
   "No" might be biased independent of orientation. Solutions if you
   see this: (a) calibrate by subtracting the mean reward over a known-
   upright batch, (b) swap to multi-class logits across rotation
   buckets, (c) use the SigLIP reward (`reward.type: siglip`) as a
   sanity check.
2. **Wrap-around in the action space**: the policy can only adjust
   ±5°/step but episodes can start at ±90°. The minimum-length episode
   to fully canonicalize is `90/5 = 18` steps. The default
   `rollout_steps=32` and `max_episode_steps=64` give comfortable
   headroom.
3. **Backbone preprocessing**: each backbone has its own
   resize/normalize. `ImageEncoderPreprocessor` uses HuggingFace's
   `AutoImageProcessor`, so you don't have to think about it -- but
   note the actual input resolution to the backbone is whatever its
   processor decides, *not* `data.image_size` (the latter is the canvas
   we rotate on).
