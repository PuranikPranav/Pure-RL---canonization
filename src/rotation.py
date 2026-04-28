"""Smart image rotation utilities.

The point: a *naive* rotation leaves black/zero corners in the rotated
image, which is a trivial cue for any supervised classifier ("if there
are 4 zero triangles, it's rotated"). We avoid that here by using
``cv2.BORDER_REFLECT_101`` which reflects pixels at the boundary, so the
corners get filled with a plausible continuation of the image. Combined
with a center crop that hides the small reflected band, the resulting
image contains essentially no obvious rotation artifact -- this is the
regime where SL struggles and the VLM-as-oracle RL setup shines.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Core rotation
# ---------------------------------------------------------------------------

def rotate_image(
    image: np.ndarray,
    angle_deg: float,
    border_mode: int = cv2.BORDER_REFLECT_101,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Rotate ``image`` by ``angle_deg`` degrees about its center.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W, 3)`` uint8 RGB image.
    angle_deg : float
        Rotation angle in degrees. Positive is counter-clockwise (the
        OpenCV convention).
    border_mode : int
        OpenCV border mode. Default ``BORDER_REFLECT_101`` reflects pixels
        at the boundary so corners are filled with plausible content.
    interpolation : int
        OpenCV interpolation flag.

    Returns
    -------
    np.ndarray
        Rotated image, same shape and dtype as input.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected (H,W,3) image, got shape {image.shape}")

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=interpolation,
        borderMode=border_mode,
    )
    return rotated


def rotate_with_center_crop(
    image: np.ndarray,
    angle_deg: float,
    crop_fraction: float = 0.92,
) -> np.ndarray:
    """Rotate then center-crop to hide any sliver of reflected boundary.

    Useful for *visualization* and for evaluation. During training we use
    plain ``rotate_image`` (full size) so the policy must learn from the
    full canvas, including the reflected regions.
    """
    rotated = rotate_image(image, angle_deg)
    h, w = rotated.shape[:2]
    new_h = int(round(h * crop_fraction))
    new_w = int(round(w * crop_fraction))
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    cropped = rotated[y0 : y0 + new_h, x0 : x0 + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------

def wrap_angle(angle_deg: float) -> float:
    """Wrap angle into ``(-180, 180]``."""
    a = float(angle_deg) % 360.0
    if a > 180.0:
        a -= 360.0
    return a


def angle_distance(a: float, b: float) -> float:
    """Smallest absolute angular distance in degrees, in ``[0, 180]``."""
    d = abs(wrap_angle(a - b))
    return d


# ---------------------------------------------------------------------------
# PIL <-> numpy convenience
# ---------------------------------------------------------------------------

def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def square_resize(img: Image.Image, size: int) -> Image.Image:
    """Resize so the *short* side becomes ``size`` and center-crop to square.

    We deliberately do NOT letterbox -- letterboxing would re-introduce
    zero/constant borders that defeat the whole point of reflect padding.
    """
    w, h = img.size
    scale = size / float(min(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


# ---------------------------------------------------------------------------
# Quick visual sanity check
# ---------------------------------------------------------------------------

def make_rotation_grid(
    image: np.ndarray,
    angles: Tuple[float, ...] = (0, 30, 60, 90, 120, 150, 180),
) -> np.ndarray:
    """Build a horizontal strip of the same image at multiple angles."""
    tiles = [rotate_image(image, a) for a in angles]
    return np.concatenate(tiles, axis=1)
