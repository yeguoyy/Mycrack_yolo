"""Turning whatever a segmentation model wrote into a clean binary mask."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi


def to_float(mask: np.ndarray) -> np.ndarray:
    """Return a single-channel crack map scaled to [0, 1].

    Accepts boolean, 0/1 label, 8/16-bit and float masks, grey or colour. A colour mask
    (red on black, say) counts its brightest channel; an RGBA mask with a real
    alpha channel uses the alpha.
    """
    m = np.asarray(mask)
    if m.ndim == 3:
        if m.shape[2] == 4:
            alpha = m[..., 3]
            m = alpha if alpha.min() != alpha.max() else m[..., :3].max(axis=2)
        else:
            m = m.max(axis=2)
    if m.dtype == bool:
        return m.astype(np.float32)
    if np.issubdtype(m.dtype, np.integer):
        top = int(m.max()) if m.size else 0
        if top <= 1:  # a 0/1 label map
            full = 1.0
        elif top <= 255:
            full = 255.0
        else:
            full = float(np.iinfo(m.dtype).max)
        return m.astype(np.float32) / full
    m = m.astype(np.float32)
    top = float(np.nanmax(m)) if m.size else 1.0
    return m / top if top > 1.0 else m


def to_uint8(image: np.ndarray) -> np.ndarray:
    """An 8-bit copy of a photo of any depth: 16-bit and float images are
    scaled by their range rather than cast (which would wrap or go black)."""
    img = np.asarray(image)
    if img.dtype == np.uint8:
        return img
    if img.dtype == bool:
        return img.astype(np.uint8) * 255
    f = img.astype(np.float32)
    if np.issubdtype(img.dtype, np.integer):
        top = 255.0 if f.max() <= 255 else float(np.iinfo(img.dtype).max)
        return np.clip(f / top * 255.0, 0, 255).astype(np.uint8)
    lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
    if lo >= 0.0 and hi <= 1.0:
        lo, hi = 0.0, 1.0
    scale = 255.0 / (hi - lo) if hi > lo else 0.0
    return np.clip((np.nan_to_num(f) - lo) * scale, 0, 255).astype(np.uint8)


def binarize(
    mask: np.ndarray,
    threshold: float = 0.5,
    hysteresis: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Binary crack mask from a mask or probability map."""
    m = to_float(mask)
    if hysteresis is not None:
        low, high = hysteresis
        weak = m >= low
        strong = m >= high
        labels, n = ndi.label(weak, structure=np.ones((3, 3)))
        if n == 0:
            return np.zeros(m.shape, bool)
        keep = np.zeros(n + 1, bool)
        keep[np.unique(labels[strong])] = True
        keep[0] = False
        return keep[labels]
    return m >= threshold


def distance(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance from each crack pixel to the nearest background
    pixel, with the image border counted as background."""
    padded = np.pad(mask.astype(np.uint8), 1)
    dt = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dt[1:-1, 1:-1]


def width_estimate(mask: np.ndarray, dt: Optional[np.ndarray] = None) -> float:
    """Median crack width in pixels, from the distance ridge of the mask."""
    if not mask.any():
        return 0.0
    if dt is None:
        dt = distance(mask)
    ridge = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6)
    vals = dt[ridge]
    if vals.size == 0:
        vals = dt[mask]
    return float(max(1.0, 2.0 * np.median(vals) - 1.0))


def remove_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels]


def fill_small_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
    if max_area <= 0:
        return mask
    m = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return mask
    out = m.copy()
    for c, h in zip(contours, hierarchy[0]):
        if h[3] >= 0 and cv2.contourArea(c) <= max_area:  # an inner contour: a hole
            cv2.drawContours(out, [c], -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def rough_width(mask: np.ndarray) -> float:
    """Quick width guess from area and outline: a long band of width w and
    length L has area wL and perimeter about 2L."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    per = sum(cv2.arcLength(c, True) for c in contours)
    return float(max(1.0, 2.0 * mask.sum() / per)) if per > 0 else 1.0


def clean(mask: np.ndarray, min_area: int = 20, fill_holes: float = -1) -> Tuple[np.ndarray, float]:
    """Drop specks and fill pinholes. Returns the mask and a rough width."""
    m = remove_specks(mask.astype(bool), min_area)
    if not m.any():
        return m, 0.0
    width = rough_width(m)
    if fill_holes < 0:
        fill_holes = max(16.0, (1.5 * width) ** 2)
    m = fill_small_holes(m, fill_holes)
    return m, width


__all__ = [
    "to_float",
    "to_uint8",
    "binarize",
    "distance",
    "width_estimate",
    "remove_specks",
    "fill_small_holes",
    "clean",
]

