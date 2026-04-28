"""Reward models for canonicalization.

Three implementations are provided behind a common interface:

1. :class:`VLMRewardModel` -- Qwen2-VL-2B-Instruct, free, ~2B params.
   Single forward pass per batch (no autoregressive generation): we just
   read the logits at the first generation position and compute
   ``reward = tanh(logit("Yes") - logit("No"))`` for the prompt
   "Is this image displayed in its natural, correct upright orientation?
   Answer with only Yes or No.". Output range: ``(-1, 1)``.

2. :class:`SigLIPRewardModel` -- ``google/siglip-large-patch16-384`` (~880M).
   Compares image embedding against text embeddings of upright vs rotated
   prompts, returns ``2*p(upright) - 1`` in ``[-1, 1]``. Faster but less
   precise than the VLM.

3. :class:`SyntheticRewardModel` -- perfect oracle from ground-truth
   cumulative angle: ``r = cos(angle)``. Used for debugging the PPO loop
   independently of the (heavy, noisy) VLM.

All models accept ``(N, H, W, 3) uint8`` numpy/Tensor images and return
``(N,)`` float32 rewards in ``[-1, 1]``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .rotation import np_to_pil


def _resolve_torch_device(device: str) -> torch.device:
    dev = (device or "auto").lower()
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if dev == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if dev == "cpu":
        return torch.device("cpu")
    return torch.device("cpu")


def _to_pil_list(images) -> List[Image.Image]:
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    out: List[Image.Image] = []
    for arr in images:
        if isinstance(arr, Image.Image):
            out.append(arr)
        else:
            out.append(np_to_pil(np.asarray(arr)))
    return out


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class RewardModel(ABC):
    """Common interface for reward models."""

    @abstractmethod
    def score(
        self,
        images,
        angles: Optional[np.ndarray] = None,
        image_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return ``(N,) float32`` rewards in ``[-1, 1]``.

        ``angles`` is the ground-truth cumulative angle per env, used by
        the synthetic model and ignored by VLM/SigLIP.
        ``image_ids`` is per-env pool index (used by VLM for per-image
        bias calibration; ignored by other reward models).
        """

    def num_parameters(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Synthetic oracle (debugging)
# ---------------------------------------------------------------------------

class SyntheticRewardModel(RewardModel):
    """Perfect-information oracle. Reward = cos(angle) in [-1, 1]."""

    def __init__(self, use_cosine: bool = True):
        self.use_cosine = use_cosine

    def score(
        self,
        images,
        angles: Optional[np.ndarray] = None,
        image_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if angles is None:
            raise ValueError("SyntheticRewardModel requires `angles`")
        ang = np.asarray(angles, dtype=np.float32)
        if self.use_cosine:
            return np.cos(np.deg2rad(ang)).astype(np.float32)
        # Triangular: 1 at 0, -1 at 180
        return (1.0 - np.abs(((ang + 180.0) % 360.0) - 180.0) / 90.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Qwen2-VL-2B reward (default)
# ---------------------------------------------------------------------------

class VLMRewardModel(RewardModel):
    """Qwen2-VL-2B-Instruct as an orientation oracle (multi-prompt + calibrated).

    Three improvements over the naive single-prompt yes/no:

    1. **Multi-prompt ensemble.** We ask multiple complementary yes/no
       questions, including one *inverted* question ("Is this image rotated?")
       whose sign is flipped before averaging. This cancels the well-known
       VLM "Yes-bias" because that bias works in our favor on positive
       prompts and *against* us on the inverted prompt -- after summation
       it cancels.

    2. **Per-image bias calibration.** Before training, we score every
       image in the pool *unrotated* and store the resulting per-image
       reward as a "baseline". At runtime we subtract that baseline from
       the live reward, so an image that the VLM scored only +0.4 when
       upright is centered: live reward - 0.4 = 0 (not penalized for
       being a hard-to-rate image), and only deviations from upright show
       up as negative reward.

    3. **Soft margin via tanh.** Reward = tanh(logit_yes - logit_no) per
       prompt -> bounded (-1, +1), then averaged with signs. Final
       calibrated reward is clipped to [-1, +1].

    Backward-compatible: if you pass the old single ``prompt=...``
    argument and no ``prompts=[...]`` list, behavior matches the original.
    """

    DEFAULT_PROMPTS = [
        {
            "text": "Is this image displayed in its natural, correct upright orientation? Answer with only Yes or No.",
            "sign": +1.0,
        },
        {
            "text": "Is this image rotated away from its natural upright orientation? Answer with only Yes or No.",
            "sign": -1.0,
        },
        {
            "text": "Is the top of the scene at the top of this image? Answer with only Yes or No.",
            "sign": +1.0,
        },
    ]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        prompt: str | None = None,
        prompts: list | None = None,
        yes_token: str = "Yes",
        no_token: str = "No",
        dtype: str = "float16",
        device: str = "cuda",
        batch_size: int = 16,
        calibrate: bool = True,
        calibration_clip: float = 0.95,
    ):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = _resolve_torch_device(device)
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.dtype = torch_dtype
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.calibrate_enabled = bool(calibrate)
        self.calibration_clip = float(calibration_clip)

        # Resolve prompts list (multi-prompt is the default; single prompt
        # for backward compat).
        if prompts is None or len(prompts) == 0:
            if prompt is not None:
                prompts = [{"text": prompt, "sign": +1.0}]
            else:
                prompts = list(self.DEFAULT_PROMPTS)
        # Normalize entries to dicts with text + sign.
        self.prompts: list = []
        for p in prompts:
            if isinstance(p, str):
                self.prompts.append({"text": p, "sign": +1.0})
            else:
                self.prompts.append(
                    {"text": str(p["text"]), "sign": float(p.get("sign", +1.0))}
                )

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()

        tok = self.processor.tokenizer
        self.yes_id = self._resolve_token_id(tok, yes_token)
        self.no_id = self._resolve_token_id(tok, no_token)

        # Per-image bias, populated by ``calibrate(...)``. Until then,
        # bias defaults to 0.0 (i.e. no calibration applied).
        self._image_bias: dict[int, float] = {}
        self._global_bias: float = 0.0

    @staticmethod
    def _resolve_token_id(tokenizer, word: str) -> int:
        for cand in (word, " " + word, word.lower(), " " + word.lower()):
            ids = tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
        return int(tokenizer.encode(word, add_special_tokens=False)[0])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    # ---------------------------------------------------------------- score

    @torch.no_grad()
    def score(
        self,
        images,
        angles: Optional[np.ndarray] = None,
        image_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return reward in roughly ``[-1, +1]``.

        Parameters
        ----------
        images : (N, H, W, 3) uint8 or list of PIL
        angles : optional, ignored for VLM (used only by synthetic).
        image_ids : optional ``(N,)`` int array of pool indices. If given
            and the model has been calibrated, we subtract the per-image
            bias for each entry. If not given, we subtract a single
            global bias.
        """
        pil_images = _to_pil_list(images)
        raw = self._score_pil_batched(pil_images)            # (N,)
        if not self.calibrate_enabled or self._global_bias == 0.0 and not self._image_bias:
            return raw.astype(np.float32)

        if image_ids is not None and self._image_bias:
            biases = np.asarray(
                [self._image_bias.get(int(i), self._global_bias) for i in image_ids],
                dtype=np.float32,
            )
        else:
            biases = np.full(raw.shape, self._global_bias, dtype=np.float32)

        calibrated = raw - biases
        return np.clip(calibrated, -1.0, 1.0).astype(np.float32)

    def _score_pil_batched(self, pil_images: List[Image.Image]) -> np.ndarray:
        rewards: List[float] = []
        for start in range(0, len(pil_images), self.batch_size):
            batch = pil_images[start : start + self.batch_size]
            r = self._score_batch_multi_prompt(batch)
            rewards.extend(r.tolist())
        return np.asarray(rewards, dtype=np.float32)

    def _score_batch_multi_prompt(self, batch: List[Image.Image]) -> torch.Tensor:
        """Average tanh(yes-no) across all prompts (with signs)."""
        accum = torch.zeros(len(batch), dtype=torch.float32)
        total_weight = 0.0
        for prompt_spec in self.prompts:
            r = self._score_batch_single(batch, prompt_spec["text"])
            accum += prompt_spec["sign"] * r
            total_weight += abs(prompt_spec["sign"])
        if total_weight > 0:
            accum /= total_weight
        return accum

    def _score_batch_single(self, batch: List[Image.Image], prompt_text: str) -> torch.Tensor:
        messages_per_image = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            for _ in batch
        ]
        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages_per_image
        ]
        inputs = self.processor(
            text=texts,
            images=batch,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        if "pixel_values" in inputs and inputs["pixel_values"].dtype != self.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        outputs = self.model(**inputs)
        logits = outputs.logits  # (B, T, V)

        attn = inputs["attention_mask"]
        last_idx = attn.sum(dim=1) - 1
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
        next_logits = logits.gather(1, gather_idx).squeeze(1)

        yes_logit = next_logits[:, self.yes_id]
        no_logit = next_logits[:, self.no_id]
        reward = torch.tanh(yes_logit - no_logit).float().cpu()
        return reward

    # ----------------------------------------------------------- calibration

    @torch.no_grad()
    def calibrate(self, images_uint8_hwc: np.ndarray) -> None:
        """Score known-upright images and store per-image baselines.

        Parameters
        ----------
        images_uint8_hwc : ``(N, H, W, 3)`` uint8
            The original, *unrotated* images. After this call,
            ``self._image_bias[i]`` holds the model's reward on the
            upright version of image ``i``. We also record the mean
            as ``self._global_bias`` for use when no ``image_ids`` are
            passed at runtime.
        """
        if not self.calibrate_enabled:
            return
        pil_images = _to_pil_list(images_uint8_hwc)
        raw = self._score_pil_batched(pil_images)
        clip_v = self.calibration_clip
        # Bias is the *expected* upright reward; clip away from +/-1 so
        # we never make the calibrated reward purely zero or saturate it.
        clipped = np.clip(raw, -clip_v, clip_v).astype(np.float32)
        self._image_bias = {i: float(clipped[i]) for i in range(len(clipped))}
        self._global_bias = float(np.mean(clipped))
        print(
            f"[VLMRewardModel] Calibrated on {len(clipped)} upright images. "
            f"global_bias={self._global_bias:+.3f}  "
            f"min_bias={float(clipped.min()):+.3f}  "
            f"max_bias={float(clipped.max()):+.3f}"
        )


# ---------------------------------------------------------------------------
# SigLIP reward (fast alternative)
# ---------------------------------------------------------------------------

class SigLIPRewardModel(RewardModel):
    """SigLIP-based reward. Reward = 2 * p(upright | image) - 1."""

    def __init__(
        self,
        model_name: str = "google/siglip-large-patch16-384",
        upright_prompts: Sequence[str] = ("a natural upright photograph",),
        rotated_prompts: Sequence[str] = ("a rotated photograph", "an upside down photo"),
        dtype: str = "float16",
        device: str = "cuda",
    ):
        from transformers import AutoModel, AutoProcessor

        self.device = _resolve_torch_device(device)
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype).to(self.device)
        self.model.eval()

        self.upright_prompts = list(upright_prompts)
        self.rotated_prompts = list(rotated_prompts)
        self._all_prompts = self.upright_prompts + self.rotated_prompts
        self._n_upright = len(self.upright_prompts)

        # Precompute text embeddings -- they don't change.
        with torch.no_grad():
            text_inputs = self.processor(
                text=self._all_prompts,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            text_feat = self.model.get_text_features(**text_inputs)
            text_feat = F.normalize(text_feat, dim=-1)
        self._text_feat = text_feat  # (P, D)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def score(
        self,
        images,
        angles: Optional[np.ndarray] = None,
        image_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        pil_images = _to_pil_list(images)
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        img_feat = self.model.get_image_features(**inputs)
        img_feat = F.normalize(img_feat, dim=-1)            # (B, D)

        sims = img_feat @ self._text_feat.t()               # (B, P)
        # Softmax over prompts; sum probability mass on the "upright" side.
        probs = F.softmax(sims * 100.0, dim=-1)             # SigLIP-ish temperature
        p_upright = probs[:, : self._n_upright].sum(dim=-1)
        reward = (2.0 * p_upright - 1.0).float().cpu().numpy()
        return reward.astype(np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_reward_model(cfg: dict) -> RewardModel:
    rtype = cfg.get("type", "vlm").lower()
    if rtype == "synthetic":
        return SyntheticRewardModel(**cfg.get("synthetic", {}))
    if rtype == "siglip":
        return SigLIPRewardModel(**cfg.get("siglip", {}))
    if rtype == "vlm":
        return VLMRewardModel(**cfg.get("vlm", {}))
    raise ValueError(f"Unknown reward type: {rtype}")
