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
│   ├── default.yaml          # full run with VLM reward
│   └── debug.yaml            # tiny, synthetic reward, CPU-friendly
├── src/
│   ├── rotation.py           # smart reflect-padded rotation
│   ├── dataset.py            # in-memory image pool, HF auto-download
│   ├── env.py                # vectorized canonicalization environment
│   ├── policy.py             # actor-critic w/ DINOv2 backbone
│   ├── reward_model.py       # VLM / SigLIP / synthetic
│   ├── ppo.py                # PPO trainer (GAE, clipping, K epochs)
│   ├── evaluate.py           # convergence-based test-time loop
│   └── utils.py              # config, seeding, running stats, logging
├── scripts/
│   ├── download_data.py      # download 100 imagenette images
│   ├── train.py              # entry point
│   ├── test.py               # canonicalize at inference time
│   └── quick_check.py        # local sanity test (no VLM, no GPU needed)
├── notebooks/colab_setup.ipynb
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

A minimal SLURM script outline (adjust account/partition):

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:a100:1
#SBATCH -t 04:00:00
#SBATCH --mem=64G
#SBATCH -J canon_ppo

module load cuda anaconda
source activate canon

python scripts/download_data.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
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
