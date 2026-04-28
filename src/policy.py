"""Actor-critic policy for canonicalization PPO.

Architecture
------------
- Vision backbone: a HuggingFace ``AutoModel`` (DINOv2 by default). We
  read the CLS token from ``last_hidden_state[:, 0]`` -- DINOv2 places
  the CLS embedding there.
- Two heads on top: an actor head over discrete rotation actions, and a
  scalar critic head. Both are 2-layer MLPs with GELU and dropout.

Parameter counts (approximate)
------------------------------
- ``dinov2-small`` ~ 22M  -> total policy ~ 23M
- ``dinov2-base``  ~ 86M  -> total policy ~ 87M  (default)
- ``dinov2-large`` ~ 300M -> total policy ~ 305M (good for the
  "policy = 25% of 2B VLM" hypothesis: 305M / 2B ~ 15%, or pair with
  Qwen2-VL-7B for ~4%; or pair with a 1.2B VLM for ~25%).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoImageProcessor, AutoModel


# ---------------------------------------------------------------------------
# Pixel-value preprocessor (numpy uint8 HWC -> normalized float CHW tensor)
# ---------------------------------------------------------------------------

class ImageEncoderPreprocessor:
    """Wraps a HuggingFace ``AutoImageProcessor`` for batched numpy input."""

    def __init__(self, backbone_name: str):
        self.processor = AutoImageProcessor.from_pretrained(backbone_name)

    def __call__(self, images_uint8_hwc: "torch.Tensor | list | tuple") -> torch.Tensor:
        """Accept ``(N, H, W, 3) uint8`` tensor or list of ndarrays.

        Returns ``(N, 3, H', W') float`` tensor on CPU.
        """
        if isinstance(images_uint8_hwc, torch.Tensor):
            images = list(images_uint8_hwc.detach().cpu().numpy())
        else:
            images = list(images_uint8_hwc)
        out = self.processor(images=images, return_tensors="pt")
        return out["pixel_values"]


# ---------------------------------------------------------------------------
# Actor-critic
# ---------------------------------------------------------------------------

class CanonicalizationPolicy(nn.Module):
    """Vision-backbone actor-critic for discrete rotation actions."""

    def __init__(
        self,
        backbone_name: str,
        num_actions: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_actions = num_actions
        self.freeze_backbone = freeze_backbone

        self.backbone = AutoModel.from_pretrained(backbone_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        feat_dim = self._infer_feat_dim()

        self.actor = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )
        self.critic = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_heads()

    # ------------------------------------------------------------------

    def _infer_feat_dim(self) -> int:
        cfg = self.backbone.config
        for key in ("hidden_size", "embed_dim", "projection_dim"):
            if hasattr(cfg, key):
                return int(getattr(cfg, key))
        raise RuntimeError(
            f"Could not infer feature dim from backbone config: {cfg}"
        )

    def _init_heads(self) -> None:
        for m in list(self.actor.modules()) + list(self.critic.modules()):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------ forward

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
        with ctx:
            out = self.backbone(pixel_values=pixel_values)
            # DINOv2 / ViT -> last_hidden_state[:, 0] is CLS.
            if hasattr(out, "last_hidden_state"):
                feat = out.last_hidden_state[:, 0]
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                feat = out.pooler_output
            else:
                raise RuntimeError("Unsupported backbone output structure")
        return feat

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encode(pixel_values)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        return logits, value

    # ---------------------------------------------------------------- API

    @torch.no_grad()
    def act(
        self,
        pixel_values: torch.Tensor,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or argmax) an action.

        Returns ``(action, log_prob, value)``.
        """
        logits, value = self.forward(pixel_values)
        if greedy:
            action = logits.argmax(dim=-1)
            log_prob = F.log_softmax(logits, dim=-1).gather(-1, action[:, None]).squeeze(-1)
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self,
        pixel_values: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_prob, entropy, value)`` for given actions."""
        logits, value = self.forward(pixel_values)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value

    # ---------------------------------------------------------------- info

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def train(self, mode: bool = True) -> "CanonicalizationPolicy":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
