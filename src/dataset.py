"""Image dataset for canonicalization.

The dataset is intentionally tiny (default 100 images) and held entirely
in memory as preprocessed numpy arrays of shape ``(N, image_size,
image_size, 3)`` uint8. This keeps the rollout loop completely free of
disk I/O -- each rollout step we simply reach into the array, rotate by
the current angle, and feed the result through the policy + reward.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image

from .rotation import pil_to_np, square_resize


_VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImagePool:
    """In-memory pool of preprocessed square RGB images."""

    def __init__(self, images: np.ndarray):
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"Expected (N,H,W,3) uint8 array, got shape {images.shape}"
            )
        if images.dtype != np.uint8:
            raise ValueError(f"Expected uint8, got {images.dtype}")
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    @property
    def image_size(self) -> int:
        return self.images.shape[1]

    def sample_indices(self, n: int, rng: random.Random) -> List[int]:
        return [rng.randrange(len(self.images)) for _ in range(n)]

    def get(self, idx: int) -> np.ndarray:
        return self.images[idx]


# ---------------------------------------------------------------------------
# Loading from disk
# ---------------------------------------------------------------------------

def _list_images(directory: Path) -> List[Path]:
    paths = [p for p in sorted(directory.iterdir()) if p.suffix.lower() in _VALID_EXT]
    return paths


def load_pool_from_dir(
    directory: str | os.PathLike,
    image_size: int,
    max_images: int | None = None,
) -> ImagePool:
    """Load images from a directory into an :class:`ImagePool`."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    paths = _list_images(directory)
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise RuntimeError(f"No images found in {directory}")

    arrays: List[np.ndarray] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = square_resize(img, image_size)
        arrays.append(pil_to_np(img))
    return ImagePool(np.stack(arrays, axis=0))


# ---------------------------------------------------------------------------
# Loading from a HuggingFace dataset (used by scripts/download_data.py)
# ---------------------------------------------------------------------------

def download_pool_from_hf(
    hf_dataset: str,
    split: str,
    num_images: int,
    image_size: int,
    out_dir: str | os.PathLike,
    seed: int = 0,
    hf_config: str | None = None,
) -> ImagePool:
    """Stream ``num_images`` from a HF dataset, save to ``out_dir``, return pool.

    We stream rather than fully download to keep things light. Most HF
    image datasets work with this pattern; we tested with
    ``frgfm/imagenette`` (clean object photos, free).

    Some datasets (e.g. ``frgfm/imagenette``) require a ``hf_config`` such as
    ``'full_size'`` / ``'320px'`` / ``'160px'``. If left ``None``, we default
    to ``'full_size'`` for ``frgfm/imagenette`` and otherwise pass ``None``.
    """
    from datasets import load_dataset  # local import: heavy

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    if hf_config is None and hf_dataset == "frgfm/imagenette":
        hf_config = "full_size"

    if hf_config is None:
        ds = load_dataset(hf_dataset, split=split, streaming=True)
    else:
        ds = load_dataset(hf_dataset, hf_config, split=split, streaming=True)

    arrays: List[np.ndarray] = []
    saved = 0
    for example in ds:
        img = _extract_pil(example)
        if img is None:
            continue
        img = square_resize(img.convert("RGB"), image_size)
        arr = pil_to_np(img)
        arrays.append(arr)
        img.save(out_dir / f"img_{saved:04d}.jpg", quality=95)
        saved += 1
        if saved >= num_images:
            break

    if saved == 0:
        raise RuntimeError(
            f"Could not extract any images from {hf_dataset}:{split}"
        )
    if saved < num_images:
        print(
            f"[dataset] WARNING: only got {saved}/{num_images} images from "
            f"{hf_dataset}:{split}"
        )
    # Light shuffle so the pool isn't all in dataset order
    idx = list(range(saved))
    rng.shuffle(idx)
    arr_stack = np.stack([arrays[i] for i in idx], axis=0)
    return ImagePool(arr_stack)


def _extract_pil(example: dict) -> Image.Image | None:
    """HF datasets store images under a few different key names."""
    for key in ("image", "img", "picture", "jpg"):
        if key in example and example[key] is not None:
            v = example[key]
            if isinstance(v, Image.Image):
                return v
            try:
                return Image.open(v)
            except Exception:
                continue
    return None
