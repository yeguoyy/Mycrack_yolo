"""Polyline helpers: arc length, resampling, smoothing, normals, widths."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates


def arclength(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return np.zeros(len(pts))
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])


def length(pts: np.ndarray) -> float:
    return float(arclength(pts)[-1]) if len(pts) > 1 else 0.0


def resample(
    pts: np.ndarray, step: float, flags: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Evenly spaced points along a polyline, keeping both ends.

    ``flags`` (one per segment) are carried over by the segment each new
    segment's midpoint falls in.
    """
    s = arclength(pts)
    total = s[-1] if len(s) else 0.0
    if total <= 1e-9:
        out = pts[:1].copy() if len(pts) else pts.copy()
        return out, (np.zeros(0, bool) if flags is not None else None)
    n = max(2, int(round(total / step)) + 1)
    t = np.linspace(0.0, total, n)
    keep = np.concatenate([[True], np.diff(s) > 1e-12])
    s_u, p_u = s[keep], pts[keep]
    out = np.column_stack([np.interp(t, s_u, p_u[:, 0]), np.interp(t, s_u, p_u[:, 1])])
    new_flags = None
    if flags is not None:
        mid = 0.5 * (t[1:] + t[:-1])
        seg = np.clip(np.searchsorted(s, mid, side="right") - 1, 0, len(flags) - 1)
        new_flags = flags[seg] if len(flags) else np.zeros(n - 1, bool)
    return out, new_flags


def smooth(pts: np.ndarray, sigma: float, closed: bool = False) -> np.ndarray:
    """Gaussian smoothing along an evenly sampled polyline.

    The ends are extended by point reflection before filtering, so they stay
    put and a straight end stays straight instead of curling inwards.
    """
    n = len(pts)
    if sigma <= 0 or n < 3:
        return pts.copy()
    if closed:
        return gaussian_filter1d(pts, sigma, axis=0, mode="wrap")
    k = min(n - 1, int(np.ceil(3 * sigma)))
    head = 2 * pts[0] - pts[k:0:-1]
    tail = 2 * pts[-1] - pts[-2 : -k - 2 : -1]
    ext = np.vstack([head, pts, tail])
    out = gaussian_filter1d(ext, sigma, axis=0, mode="nearest")[k : k + n]
    out[0], out[-1] = pts[0], pts[-1]
    return out


def tangents(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return np.tile([1.0, 0.0], (len(pts), 1))
    d = np.gradient(pts, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(n, 1e-12)


def normals(pts: np.ndarray) -> np.ndarray:
    t = tangents(pts)
    return np.column_stack([-t[:, 1], t[:, 0]])


def cross_widths(
    mask: np.ndarray,
    pts: np.ndarray,
    nrm: np.ndarray,
    reach: np.ndarray,
    chunk: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Width of the mask across the crack at each point, to sub-pixel.

    Samples the mask along each normal (bilinear, so the 0.5 crossing falls
    on the pixel edge) and finds where it leaves the crack on either side.
    Returns ``(width, offset)``: ``offset`` is how far along the normal the
    middle of the crack sits from the point. NaN where the profile is not
    closed within ``reach``.
    """
    m = np.asarray(mask, np.float32)
    N = len(pts)
    width = np.full(N, np.nan)
    offset = np.full(N, np.nan)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        R = float(np.clip(reach[a:b].max(), 3.0, 250.0))
        ds = max(0.25, R / 400.0)
        s = np.arange(-R, R + ds / 2, ds)
        K = len(s)
        P = pts[a:b]
        V = nrm[a:b]
        xs = P[:, 0, None] + s[None, :] * V[:, 0, None]
        ys = P[:, 1, None] + s[None, :] * V[:, 1, None]
        f = map_coordinates(m, [ys.ravel(), xs.ravel()], order=1, mode="grid-constant", cval=0.0)
        f = f.reshape(len(P), K)
        inside = f >= 0.5
        J = np.arange(K)[None, :]
        # start from the nearest inside sample to the point (within 2 px)
        near = np.abs(s) <= 2.0
        dist = np.where(inside & near[None, :], np.abs(s)[None, :], np.inf)
        cidx = np.argmin(dist, axis=1)
        ok = np.isfinite(dist[np.arange(len(P)), cidx])
        cc = cidx[:, None]
        right = (~inside) & (J > cc)
        left = (~inside) & (J < cc)
        has_r = right.any(axis=1)
        has_l = left.any(axis=1)
        jr = np.argmax(right, axis=1)  # first outside sample to the right
        jl = K - 1 - np.argmax(left[:, ::-1], axis=1)  # last outside sample to the left
        ok &= has_r & has_l
        rows = np.arange(len(P))
        # right crossing between jr-1 (inside) and jr (outside)
        fi = f[rows, np.clip(jr - 1, 0, K - 1)]
        fo = f[rows, jr]
        tr = np.clip((fi - 0.5) / np.maximum(fi - fo, 1e-6), 0, 1)
        sr = s[np.clip(jr - 1, 0, K - 1)] + tr * ds
        fi = f[rows, np.clip(jl + 1, 0, K - 1)]
        fo = f[rows, jl]
        tl = np.clip((fi - 0.5) / np.maximum(fi - fo, 1e-6), 0, 1)
        sl = s[np.clip(jl + 1, 0, K - 1)] - tl * ds
        w = np.where(ok, sr - sl, np.nan)
        o = np.where(ok, 0.5 * (sr + sl), np.nan)
        width[a:b] = w
        offset[a:b] = o
    return width, offset


def principal_angle(pts: np.ndarray) -> float:
    """Direction of a point set, degrees in [0, 180) counter-clockwise from
    the image x axis (image y points down, so it is flipped here)."""
    if len(pts) < 2:
        return 0.0
    c = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    vx, vy = vt[0]
    return float(np.degrees(np.arctan2(-vy, vx)) % 180.0)


__all__ = [
    "arclength",
    "length",
    "resample",
    "smooth",
    "tangents",
    "normals",
    "cross_widths",
    "principal_angle",
]

