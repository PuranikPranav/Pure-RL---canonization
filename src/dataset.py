"""Image dataset for canonicalization.

The dataset is intentionally tiny (default 100 images) and held entirely
in memory as preprocessed numpy arrays of shape ``(N, image_size,
image_size, 3)`` uint8. This keeps the rollout loop completely free of
disk I/O -- each rollout step we simply reach into the array, rotate by
the current angle, and feed the result through the policy + reward.

Sources
-------
We support three image sources, all dispatched through
:func:`build_pool_from_spec` and combined by :func:`build_combined_pool`:

* ``hf`` -- any HuggingFace ``datasets`` image dataset, streamed.
* ``torchvision`` -- ``CIFAR10``, ``CIFAR100``, ``EMNIST``. Reliable,
  always available, lets the experiments run offline-friendly.
* ``dir`` -- a local folder of images (drop your Chars74K bytes here).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
    shuffle: bool = True,
    seed: int = 0,
) -> ImagePool:
    """Load images from a directory into an :class:`ImagePool`.

    By default we **shuffle** before truncating to ``max_images`` so that
    a directory with class-grouped filenames (e.g. Chars74K's
    ``Sample001_*``, ``Sample002_*``, ...) yields a *diverse* subset
    rather than every image from the first class.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    paths = _list_images(directory)
    if shuffle:
        rng = random.Random(seed)
        paths = list(paths)
        rng.shuffle(paths)
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


# ---------------------------------------------------------------------------
# Loading from torchvision (CIFAR10 / CIFAR100 / EMNIST). Reliable + offline-
# cacheable, used as the workhorse for the combined-dataset experiment.
# ---------------------------------------------------------------------------

_TORCHVISION_REGISTRY = {
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "emnist": "EMNIST",
}


def download_pool_from_torchvision(
    name: str,
    split: str,
    num_images: int,
    image_size: int,
    out_dir: str | os.PathLike,
    seed: int = 0,
    options: Dict[str, Any] | None = None,
) -> ImagePool:
    """Build an :class:`ImagePool` from a torchvision dataset.

    Parameters
    ----------
    name : str
        One of ``cifar10``, ``cifar100``, ``emnist``.
    split : str
        ``train`` or ``test``. (For EMNIST this controls the ``train`` flag.)
    num_images : int
        Number of images to draw (random subset, deterministic given ``seed``).
    image_size : int
        Final square canvas. Small native images (CIFAR=32, EMNIST=28) are
        upscaled to this size with bicubic.
    out_dir : path
        Where to cache the raw torchvision data. Reused across runs.
    seed : int
        For deterministic subset selection.
    options : dict
        Extra options passed through to the torchvision dataset constructor.
        For EMNIST you typically want ``{"emnist_split": "byclass"}``.
    """
    import torchvision  # local import: keep top-of-module cheap

    name = name.lower()
    if name not in _TORCHVISION_REGISTRY:
        raise ValueError(
            f"Unsupported torchvision dataset '{name}'. "
            f"Supported: {sorted(_TORCHVISION_REGISTRY)}"
        )
    cls = getattr(torchvision.datasets, _TORCHVISION_REGISTRY[name])
    options = dict(options or {})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_flag = split.lower().startswith("train")
    kwargs: Dict[str, Any] = {"root": str(out_dir), "download": True}
    if name in {"cifar10", "cifar100"}:
        kwargs["train"] = train_flag
    elif name == "emnist":
        kwargs["train"] = train_flag
        kwargs["split"] = options.get("emnist_split", "byclass")

    ds = cls(**kwargs)

    rng = random.Random(seed)
    n_total = len(ds)
    n = min(int(num_images), n_total)
    indices = rng.sample(range(n_total), n)

    arrays: List[np.ndarray] = []
    saved = 0
    for i in indices:
        item = ds[i]
        # torchvision returns (PIL.Image, label); some datasets give tensors.
        img = item[0] if isinstance(item, tuple) else item
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(np.asarray(img))
            except Exception:
                continue
        img = img.convert("RGB")

        # EMNIST quirk: tensors are stored transposed wrt the visual upright;
        # PIL conversion already gets it right, but the *visual* upright is
        # rotated 90 deg + horizontally flipped relative to what we want.
        # Apply the canonical correction so "upright" actually looks upright.
        if name == "emnist":
            img = img.transpose(Image.TRANSPOSE)

        img = square_resize(img, image_size)
        arrays.append(pil_to_np(img))
        saved += 1
        if saved >= num_images:
            break

    if saved == 0:
        raise RuntimeError(f"Could not extract any images from torchvision:{name}:{split}")
    return ImagePool(np.stack(arrays, axis=0))


# ---------------------------------------------------------------------------
# Unified spec dispatch + combined pool
# ---------------------------------------------------------------------------

def build_pool_from_spec(
    spec: Dict[str, Any],
    image_size: int,
    base_dir: str | os.PathLike,
    seed: int,
) -> ImagePool:
    """Resolve one source spec into an :class:`ImagePool`.

    A spec is a dict with at least ``source`` and ``num_images``. Source-
    specific keys are documented in :func:`build_combined_pool`.
    """
    source = str(spec.get("source", "")).lower()
    num_images = int(spec["num_images"])
    if source == "hf":
        out_dir = Path(base_dir) / spec.get("cache_subdir", spec["name"].replace("/", "_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        return download_pool_from_hf(
            hf_dataset=spec["name"],
            split=spec.get("split", "train"),
            num_images=num_images,
            image_size=image_size,
            out_dir=out_dir,
            seed=seed,
            hf_config=spec.get("hf_config"),
        )
    if source == "torchvision":
        out_dir = Path(base_dir) / spec.get("cache_subdir", spec["name"].lower())
        return download_pool_from_torchvision(
            name=spec["name"],
            split=spec.get("split", "train"),
            num_images=num_images,
            image_size=image_size,
            out_dir=out_dir,
            seed=seed,
            options=spec.get("options"),
        )
    if source == "dir":
        directory = Path(spec["path"])
        return load_pool_from_dir(
            directory=directory,
            image_size=image_size,
            max_images=num_images,
            shuffle=bool(spec.get("shuffle", True)),
            seed=int(seed),
        )
    raise ValueError(
        f"Unknown source '{source}' in spec {spec!r}. "
        f"Expected one of: hf | torchvision | dir."
    )


def build_combined_pool(
    specs: Sequence[Dict[str, Any]],
    image_size: int,
    base_dir: str | os.PathLike,
    seed: int = 0,
    shuffle: bool = True,
) -> ImagePool:
    """Concatenate multiple sub-pools into a single :class:`ImagePool`.

    Each ``spec`` in ``specs`` is one of:

    * ``{"source": "hf", "name": "frgfm/imagenette", "split": "train", \
"num_images": 100, "hf_config": "full_size"}``
    * ``{"source": "torchvision", "name": "cifar10", "split": "train", \
"num_images": 100}``
    * ``{"source": "torchvision", "name": "emnist", "split": "train", \
"num_images": 100, "options": {"emnist_split": "byclass"}}``
    * ``{"source": "dir", "path": "data/chars74k", "num_images": 100}``

    A spec may include ``"seed_offset"`` to give it a different random
    subset than another spec when both pull from the same source.
    Sub-pools are concatenated and (optionally) shuffled together so the
    rollout sampler doesn't see all CIFAR before all Chars74K.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    sub_pools: List[np.ndarray] = []
    for i, spec in enumerate(specs):
        sub_seed = int(seed) + int(spec.get("seed_offset", i))
        pool = build_pool_from_spec(spec, image_size=image_size, base_dir=base_dir, seed=sub_seed)
        sub_pools.append(pool.images)
        print(
            f"[dataset] sub-pool {i}: source={spec.get('source')} "
            f"name={spec.get('name', spec.get('path'))} -> {len(pool)} imgs"
        )
    if not sub_pools:
        raise ValueError("build_combined_pool got an empty `specs` list")

    combined = np.concatenate(sub_pools, axis=0)
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(combined))
        combined = combined[idx]
    return ImagePool(combined)
