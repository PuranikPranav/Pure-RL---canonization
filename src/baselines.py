"""Baselines for the canonicalization comparison.

Each baseline implements a single common API::

    final_angle, n_steps, trace = baseline.canonicalize(image_uint8, init_angle)

so that ``scripts/compare_baselines.py`` can run them side by side on
identical (image, initial-rotation) pairs.

We provide four baselines:

1. :class:`BruteForceVLMBaseline`
   Uses a *small* image-text scoring VLM (default ``openai/clip-vit-base-patch16``,
   ~150M params; comparable in size to a DINOv2-base PPO policy ~87M).
   For each test image, scores every candidate angle in ``[-bound, +bound]``
   and picks argmax. **Zero learning, zero iterations.** This is the
   "what does a similar-sized VLM give us out of the box?" baseline.

2. :class:`IterativeVLMBaseline`
   Same scoring model as (1), but used as a step-by-step policy: at each
   step, score the rotated image you'd get from each of the 11 candidate
   actions, take argmax, repeat. Same loop shape as PPO; the only thing
   missing is *learning*. This isolates the contribution of training.

3. :class:`CNNRotationRegressor`
   A small ResNet-18 (~11M params) that regresses ``(cos theta, sin theta)``
   on rotated images. **Trained supervised** on the same image pool used
   by PPO. At test time, predicts the angle, rotates by its negative.
   This is the "classic SL approach" the README warns can be defeated by
   reflect-padding -- the comparison validates that the RL hypothesis
   actually buys us something.

4. :class:`RandomBaseline`
   Random actions. Sanity floor.

All baselines return both the final corrected image's predicted angle
(in the *world* frame, where 0 is upright) and a per-step reward trace
where applicable, so we can compare convergence speed in addition to
accuracy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .rotation import np_to_pil, rotate_image, wrap_angle


# ---------------------------------------------------------------------------
# Common API
# ---------------------------------------------------------------------------

@dataclass
class BaselineTrace:
    """Per-step record so we can compute "steps to converge" etc."""
    angles: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)


class CanonicalizationBaseline(ABC):
    """Common interface so the comparison harness can treat them uniformly."""

    name: str = "abstract"

    @abstractmethod
    def canonicalize(
        self,
        image: np.ndarray,
        initial_angle: float,
    ) -> Tuple[float, int, BaselineTrace]:
        """Canonicalize ``image`` starting from ``initial_angle``.

        Returns
        -------
        final_angle : float
            Predicted residual angle (degrees, wrapped to ``(-180, 180]``).
            ``0`` means perfectly upright. ``|final_angle|`` is the error.
        n_steps : int
            Number of model forward passes used (1 for one-shot baselines).
        trace : BaselineTrace
            Per-step (angle, reward) sequence. Empty for one-shot models.
        """


# ---------------------------------------------------------------------------
# Random control
# ---------------------------------------------------------------------------

class RandomBaseline(CanonicalizationBaseline):
    """Uniformly random action per step. Pure sanity floor."""

    name = "random"

    def __init__(self, action_bound: int = 5, action_step: int = 1, max_steps: int = 64, seed: int = 0):
        self.action_bound = int(action_bound)
        self.action_step = int(action_step)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)

    def canonicalize(self, image: np.ndarray, initial_angle: float):
        angle = float(initial_angle)
        trace = BaselineTrace()
        actions = np.arange(-self.action_bound, self.action_bound + 1, self.action_step)
        for _ in range(self.max_steps):
            trace.angles.append(angle)
            trace.rewards.append(float(np.cos(np.deg2rad(angle))))
            a = float(self.rng.choice(actions))
            angle = wrap_angle(angle + a)
        return wrap_angle(angle), self.max_steps, trace


# ---------------------------------------------------------------------------
# Small-VLM scorers (CLIP / SigLIP)
# ---------------------------------------------------------------------------

class _SmallVLMScorer:
    """Wraps a CLIP-like model into ``score(images, prompts) -> probs(upright)``.

    We use a softmax over [upright_prompts, rotated_prompts] and return the
    summed probability mass on the "upright" side, which gives a smooth
    score in ``[0, 1]``. Sign is flipped to ``[-1, +1]`` for parity with the
    reward model output range (``2*p - 1``).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        device: str = "cuda",
        dtype: str = "float16",
        upright_prompts: Sequence[str] = (
            "a natural upright photograph",
            "a correctly oriented image",
        ),
        rotated_prompts: Sequence[str] = (
            "a rotated photograph",
            "an upside down photo",
            "a sideways photo",
        ),
        batch_size: int = 32,
    ):
        from transformers import AutoModel, AutoProcessor

        self.device = torch.device(
            device if (torch.cuda.is_available() or device == "cpu") else "cpu"
        )
        self.dtype = {
            "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
        }[dtype]
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device)
        self.model.eval()
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.upright_prompts = list(upright_prompts)
        self.rotated_prompts = list(rotated_prompts)
        self._all = self.upright_prompts + self.rotated_prompts
        self._n_upright = len(self.upright_prompts)

        with torch.no_grad():
            text_inputs = self.processor(text=self._all, padding=True, return_tensors="pt").to(self.device)
            text_feat = self.model.get_text_features(**text_inputs)
            text_feat = F.normalize(text_feat, dim=-1)
        self._text_feat = text_feat

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def score_pil(self, pil_images: List[Image.Image]) -> np.ndarray:
        rewards: List[float] = []
        for s in range(0, len(pil_images), self.batch_size):
            batch = pil_images[s : s + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            img_feat = self.model.get_image_features(**inputs)
            img_feat = F.normalize(img_feat, dim=-1)
            sims = img_feat @ self._text_feat.t()
            probs = F.softmax(sims * 100.0, dim=-1)
            p_up = probs[:, : self._n_upright].sum(dim=-1)
            r = (2.0 * p_up - 1.0).float().cpu().numpy()
            rewards.extend(r.tolist())
        return np.asarray(rewards, dtype=np.float32)


class BruteForceVLMBaseline(CanonicalizationBaseline):
    """Score every candidate angle in a grid, pick the highest.

    No iteration, no learning. The single forward pass is the *batch* of
    rotated candidates, so this is fast.
    """

    name = "vlm_bruteforce"

    def __init__(
        self,
        scorer: _SmallVLMScorer,
        candidate_angles: Optional[Sequence[float]] = None,
    ):
        self.scorer = scorer
        if candidate_angles is None:
            candidate_angles = list(range(-90, 91, 2))   # 91 candidates step=2 deg
        self.candidate_angles = [float(a) for a in candidate_angles]

    def canonicalize(self, image: np.ndarray, initial_angle: float):
        # Render the image as currently rotated by ``initial_angle``, then
        # consider candidate *additional* rotations to apply on top of that.
        rotated_now = rotate_image(image, float(initial_angle))
        candidates = [rotate_image(rotated_now, a) for a in self.candidate_angles]
        pil = [np_to_pil(c) for c in candidates]
        scores = self.scorer.score_pil(pil)
        best_idx = int(np.argmax(scores))
        delta = self.candidate_angles[best_idx]
        # The baseline *would have applied* -delta to bring the image upright,
        # so the residual world-frame angle after that correction is:
        residual = wrap_angle(float(initial_angle) + delta)
        trace = BaselineTrace(
            angles=list(self.candidate_angles),
            rewards=list(scores.tolist()),
        )
        return residual, 1, trace


class IterativeVLMBaseline(CanonicalizationBaseline):
    """Step-by-step greedy policy using a small VLM as an in-context scorer.

    At each step, score the 11 rotated candidates implied by the action
    space and take the argmax action. Same loop shape as PPO with the
    crucial difference that no parameters are learned -- this is a fair
    "VLM-as-policy without RL" baseline.
    """

    name = "vlm_iterative"

    def __init__(
        self,
        scorer: _SmallVLMScorer,
        action_bound: int = 5,
        action_step: int = 1,
        max_steps: int = 64,
        tolerance: float = 0.005,
        patience: int = 5,
        reward_threshold: Optional[float] = None,
    ):
        self.scorer = scorer
        self.action_bound = int(action_bound)
        self.action_step = int(action_step)
        self.actions = list(range(-self.action_bound, self.action_bound + 1, self.action_step))
        self.max_steps = int(max_steps)
        self.tolerance = float(tolerance)
        self.patience = int(patience)
        self.reward_threshold = reward_threshold

    def canonicalize(self, image: np.ndarray, initial_angle: float):
        angle = float(initial_angle)
        trace = BaselineTrace()
        last_r: Optional[float] = None
        delta_count = 0
        high_count = 0

        for step in range(self.max_steps):
            current = rotate_image(image, angle)
            current_r = float(self.scorer.score_pil([np_to_pil(current)])[0])
            trace.angles.append(angle)
            trace.rewards.append(current_r)

            # Stopping checks
            if last_r is not None and abs(current_r - last_r) < self.tolerance:
                delta_count += 1
            else:
                delta_count = 0
            if self.reward_threshold is not None and current_r >= self.reward_threshold:
                high_count += 1
            else:
                high_count = 0
            if delta_count >= self.patience or (
                self.reward_threshold is not None and high_count >= self.patience
            ):
                return wrap_angle(angle), step + 1, trace
            last_r = current_r

            # Look ahead: rotate by each candidate action delta and score.
            candidates = [rotate_image(image, wrap_angle(angle + a)) for a in self.actions]
            pil = [np_to_pil(c) for c in candidates]
            scores = self.scorer.score_pil(pil)
            best = int(np.argmax(scores))
            angle = wrap_angle(angle + self.actions[best])

        return wrap_angle(angle), self.max_steps, trace


# ---------------------------------------------------------------------------
# CNN rotation regressor
# ---------------------------------------------------------------------------

class CNNRotationRegressor(nn.Module, CanonicalizationBaseline):
    """ResNet-18 -> (cos theta, sin theta).

    We regress ``(cos, sin)`` (then renormalize) instead of the raw angle
    so the network never has to model the +/-180 wrap-around discontinuity.
    Inference: predict (c, s), recover ``theta = atan2(s, c)`` in degrees,
    output residual after applying ``-theta`` correction.

    Trained supervised by ``scripts/train_baseline_cnn.py`` on rotated
    versions of the same image pool used by PPO -- the same images, the
    same reflect-padded rotation, the same data distribution.
    """

    name = "cnn_regressor"

    def __init__(self, pretrained: bool = True):
        nn.Module.__init__(self)
        import torchvision

        weights = (
            torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        backbone = torchvision.models.resnet18(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),  # (cos, sin)
        )
        # Normalization buffers consistent with ImageNet weights
        self.register_buffer(
            "_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self._device_cache = torch.device("cpu")

    # ----- Module pieces ---------------------------------------------------

    def _preprocess(self, images_uint8: np.ndarray) -> torch.Tensor:
        if images_uint8.ndim == 3:
            images_uint8 = images_uint8[None]
        x = torch.from_numpy(np.ascontiguousarray(images_uint8)).float().div_(255.0)
        x = x.permute(0, 3, 1, 2)            # NHWC -> NCHW
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        return x.to(self._device_cache)

    def forward(self, images_uint8: np.ndarray) -> torch.Tensor:
        x = self._preprocess(images_uint8)
        feat = self.backbone(x)
        out = self.head(feat)
        out = F.normalize(out, dim=-1)        # unit (cos, sin)
        return out

    # ----- Persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, device: torch.device) -> "CNNRotationRegressor":
        ckpt = torch.load(path, map_location=device)
        model = cls(pretrained=False).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        model._device_cache = device
        return model

    # ----- Baseline API ---------------------------------------------------

    @torch.no_grad()
    def canonicalize(self, image: np.ndarray, initial_angle: float):
        rotated = rotate_image(image, float(initial_angle))
        cs = self.forward(rotated[None]).cpu().numpy()[0]
        c, s = float(cs[0]), float(cs[1])
        predicted = float(np.rad2deg(np.arctan2(s, c)))
        residual = wrap_angle(float(initial_angle) - predicted)
        trace = BaselineTrace(
            angles=[float(initial_angle), residual],
            rewards=[float(c)],   # 'c' = cos(predicted) ~ confidence proxy
        )
        return residual, 1, trace
