"""Synthetic cracks with exact ground truth.

Real crack datasets label pixels, not centrelines, so they cannot say how
long a crack really is or how wide it is at each point. These generated
cracks can: every image comes with the centreline it was drawn from, the
width along it, its junctions and its ends.

Each crack is a meandering path with fine zig-zag, a width that wanders and
tapers to its tips, and sometimes branches. The mask is rendered from that
path, then (depending on the preset) roughened, broken, speckled and holed
the way a segmentation model's output tends to be. A concrete-like photo is
rendered from the true crack, not the damaged mask.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


@dataclass(frozen=True)
class SynthSpec:
    size: int = 768
    cracks: Tuple[int, int] = (1, 3)
    width: Tuple[float, float] = (1.5, 18.0)
    branch_p: float = 0.6
    ragged: float = 0.0  # boundary noise, as a fraction of local width
    gaps: int = 0  # up to this many breaks per crack
    specks: int = 0  # up to this many false blobs per image
    holes: bool = False


PRESETS: Dict[str, SynthSpec] = {
    "clean": SynthSpec(),
    "ragged": SynthSpec(ragged=0.4),
    "broken": SynthSpec(gaps=4),
    "realistic": SynthSpec(ragged=0.3, gaps=2, specks=12, holes=True),
}


def _noise1d(rng, n, sigma):
    """Smooth noise with unit standard deviation in expectation.

    Scaled by the filter's known gain rather than the sample's own spread,
    so a short path does not get its few wiggles blown up."""
    sigma = max(sigma, 0.5)
    v = gaussian_filter1d(rng.normal(size=n), sigma, mode="reflect")
    return v * np.sqrt(2.0 * np.sqrt(np.pi) * sigma)


def _walk(rng, p0, heading, length, w0, size, step=0.5):
    n = max(4, int(length / step))
    low = _noise1d(rng, n, 120 / step) * 0.35
    mid = _noise1d(rng, n, max(6.0, 1.5 * w0) / step) * 0.22
    theta = heading + low + mid
    d = np.column_stack([np.cos(theta), np.sin(theta)]) * step
    pts = p0 + np.vstack([[0, 0], np.cumsum(d, axis=0)])
    inside = (pts >= 1.0).all(axis=1) & (pts <= size - 2.0).all(axis=1)
    if not inside.all():
        cut = int(np.argmin(inside))
        pts = pts[: max(cut, 2)]
        at_border = True
    else:
        at_border = False
    return pts, theta[: len(pts)], at_border


def _width_profile(rng, n, w0, step, blunt_start, blunt_end, taper_len):
    w = w0 * (1.0 + 0.28 * _noise1d(rng, n, 40 / step))
    w = np.maximum(w, 0.6 * w0)
    s = np.arange(n) * step
    L = s[-1] if n else 0.0
    t = np.ones(n)
    if not blunt_start:
        t = np.minimum(t, np.clip(s / taper_len, 0.12, 1.0) ** 0.6)
    if not blunt_end:
        t = np.minimum(t, np.clip((L - s) / taper_len, 0.12, 1.0) ** 0.6)
    return np.maximum(w * t, 1.5)


def _render(paths, size, ss=4):
    """Coverage map [0, 1] of discs along every path, supersampled."""
    S = size * ss
    canvas = np.zeros((S, S), np.uint8)
    for pts, w in paths:
        for (x, y), wi in zip(pts, w):
            c = (int(round((x + 0.5) * ss * 16 - 8)), int(round((y + 0.5) * ss * 16 - 8)))
            # cv2 fills a disc half a pixel past its radius; take that back
            r = int(round((wi / 2 * ss - 0.5) * 16))
            cv2.circle(canvas, c, max(r, 1), 255, -1, cv2.LINE_8, shift=4)
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def _extend_tips(pts, w, start, end, size):
    """Carry a ground-truth line out to the tip of its drawn mask.

    Discs of radius w/2 swept along the line reach w/2 past its last point,
    so the crack, as drawn, ends there."""
    pts, w = np.asarray(pts, float), np.asarray(w, float)

    def tip(p_end, p_in, r):
        d = p_end - p_in
        n = np.linalg.norm(d)
        if n < 1e-9:
            return None
        q = p_end + d / n * r
        return np.clip(q, -0.5, size - 0.5)

    if end and len(pts) > 4:
        q = tip(pts[-1], pts[-5], w[-1] / 2)
        if q is not None:
            pts, w = np.vstack([pts, q]), np.append(w, w[-1])
    if start and len(pts) > 4:
        q = tip(pts[0], pts[4], w[0] / 2)
        if q is not None:
            pts, w = np.vstack([q, pts]), np.insert(w, 0, w[0])
    return pts, w


def _dt(mask):
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def damage(
    truth: np.ndarray,
    paths: List[Tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    ragged: float = 0.0,
    gaps: int = 0,
    holes: bool = False,
    specks: int = 0,
    step: float = 0.5,
) -> Tuple[np.ndarray, List[List[float]]]:
    """Spoil a clean mask the way a segmentation model tends to: ragged
    edges, breaks across the crack, pinholes in wide parts, and specks.

    ``paths`` are the crack centrelines with their widths, points ``step``
    px apart; breaks and holes are placed along them.
    """
    mask = truth.copy()
    H, W = mask.shape
    if ragged > 0 and mask.any():
        din, dout = _dt(mask), _dt(~mask)
        phi = din - dout
        local = cv2.dilate(din, np.ones((9, 9), np.uint8), iterations=2)
        amp = np.clip(ragged * local, 0.3, 2.0)
        n = gaussian_filter(rng.normal(size=mask.shape), 1.6) * (2.0 * np.sqrt(np.pi) * 1.6)
        mask = (phi + amp * n) > 0
    where: List[List[float]] = []
    if gaps:
        for pts, w in paths:
            for _ in range(int(rng.integers(0, gaps + 1))):
                if len(pts) < 40:
                    continue
                k = int(rng.uniform(0.15, 0.85) * len(pts))
                g = rng.uniform(3, min(6 + 2.5 * w[k], 28))
                k2 = min(len(pts) - 1, k + int(g / step))
                seg = np.round(pts[k : k2 + 1] * 16).astype(np.int32).reshape(-1, 1, 2)
                th = int(np.ceil(w[k : k2 + 1].max() + 6))
                m8 = mask.astype(np.uint8)
                cv2.polylines(m8, [seg], False, 0, th, cv2.LINE_8, shift=4)
                mask = m8.astype(bool)
                where.append(pts[k].tolist())
    if holes:
        for pts, w in paths:
            for _ in range(int(rng.integers(0, 4))):
                k = int(rng.integers(0, len(pts)))
                if w[k] < 7:
                    continue
                ax = (int(rng.uniform(1, 0.22 * w[k])), int(rng.uniform(1, 0.22 * w[k])))
                m8 = mask.astype(np.uint8)
                cv2.ellipse(m8, tuple(np.round(pts[k]).astype(int)), ax, rng.uniform(0, 180), 0, 360, 0, -1)
                mask = m8.astype(bool)
    if specks:
        far = _dt(~truth) > 15 if truth.any() else np.ones_like(truth)
        for _ in range(int(rng.integers(0, specks + 1))):
            x, y = int(rng.integers(5, W - 5)), int(rng.integers(5, H - 5))
            if not far[y, x]:
                continue
            ax = (int(rng.uniform(1, 5)), int(rng.uniform(1, 4)))
            m8 = mask.astype(np.uint8)
            cv2.ellipse(m8, (x, y), ax, rng.uniform(0, 180), 0, 360, 1, -1)
            mask = m8.astype(bool)
    return mask, where


def make_sample(rng: np.random.Generator, spec: SynthSpec) -> Dict[str, Any]:
    size = spec.size
    step = 0.5
    cracks: List[Dict[str, Any]] = []
    occupied = np.zeros((size, size), bool)
    n_cracks = int(rng.integers(spec.cracks[0], spec.cracks[1] + 1))
    lo, hi = np.log(spec.width[0]), np.log(spec.width[1])

    for _ in range(n_cracks):
        for _attempt in range(25):
            w0 = float(np.exp(rng.uniform(lo, hi)))
            length = rng.uniform(0.45, 1.1) * size
            p0 = rng.uniform(0.1, 0.9, 2) * size
            heading = rng.uniform(0, 2 * np.pi)
            pts, theta, at_border = _walk(rng, p0, heading, length, w0, size, step)
            if len(pts) * step < 0.2 * size:
                continue
            taper = min(0.2 * len(pts) * step, 10 * w0)
            w = _width_profile(rng, len(pts), w0, step, rng.random() < 0.3, at_border or rng.random() < 0.3, taper)
            paths = [(pts, w)]
            junctions, endpoints = [], [pts[0]] + [pts[-1]]
            if rng.random() < spec.branch_p:
                for _b in range(int(rng.integers(1, 3))):
                    k = int(rng.uniform(0.25, 0.75) * len(pts))
                    side = rng.choice([-1, 1])
                    h = theta[k] + side * np.radians(rng.uniform(25, 60))
                    wb0 = float(w[k] * rng.uniform(0.45, 0.8))
                    bl = rng.uniform(0.2, 0.45) * len(pts) * step
                    bpts, _, b_border = _walk(rng, pts[k], h, bl, wb0, size, step)
                    if len(bpts) * step < max(25.0, 3 * w[k]):
                        continue
                    bw = _width_profile(
                        rng, len(bpts), wb0, step, True, b_border, min(0.3 * len(bpts) * step, 10 * wb0)
                    )
                    # a branch has to leave its parent's band and stay clear of
                    # the other branches, or no mask could show it
                    far = bpts[int(3 * (w[k] + 4) / step) :]
                    if len(far) < 10:
                        continue
                    clash = False
                    for q, qw in paths:
                        sub = q[::4]
                        if np.min(np.linalg.norm(far[::2, None] - sub[None], axis=2)) < 0.5 * (qw.max() + wb0) + 4:
                            clash = True
                    if clash:
                        continue
                    paths.append((bpts, bw))
                    junctions.append(pts[k])
                    endpoints.append(bpts[-1])
            cov = _render(paths, size) >= 0.5
            if (cov & cv2.dilate(occupied.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)).any():
                continue
            occupied |= cov
            cracks.append(dict(paths=paths, junctions=junctions, endpoints=endpoints))
            break

    all_paths = [p for c in cracks for p in c["paths"]]
    coverage = _render(all_paths, size) if all_paths else np.zeros((size, size), np.float32)
    truth = coverage >= 0.5
    mask = truth.copy()

    mask, gaps = damage(truth, all_paths, rng, spec.ragged, spec.gaps, spec.holes, spec.specks, step)

    photo = render_photo(rng, coverage)
    for c in cracks:
        c["truth"] = [
            _extend_tips(p[::2], w[::2], start=(k == 0), end=True, size=size) for k, (p, w) in enumerate(c["paths"])
        ]

    gt = {
        "width": size,
        "height": size,
        "cracks": [
            {
                "paths": [
                    {
                        "points": np.round(tp, 3).tolist(),
                        "width": np.round(tw, 3).tolist(),
                    }
                    for tp, tw in c["truth"]
                ],
                "junctions": np.round(np.array(c["junctions"]).reshape(-1, 2), 3).tolist(),
                "endpoints": np.round(
                    np.array([tp[-1] for tp, _ in c["truth"]] + [c["truth"][0][0][0]]).reshape(-1, 2), 3
                ).tolist(),
            }
            for c in cracks
        ],
        "gaps": gaps,
        "spec": asdict(spec),
    }
    return dict(mask=mask, truth=truth, photo=photo, gt=gt)


def render_photo(rng: np.random.Generator, coverage: np.ndarray) -> np.ndarray:
    """A concrete-like BGR photo with the crack darkened into it."""
    H, W = coverage.shape
    tex = np.zeros((H, W), np.float32)
    for sigma, amp in ((0.6, 0.045), (1.8, 0.05), (5, 0.04), (20, 0.05), (80, 0.07)):
        n = gaussian_filter(rng.normal(size=(H, W)).astype(np.float32), sigma)
        tex += amp * n / max(n.std(), 1e-6)
    pits = (gaussian_filter(rng.random((H, W)).astype(np.float32), 1.0) > 0.62).astype(np.float32)
    tex -= 0.06 * gaussian_filter(pits, 0.7)
    tone = rng.uniform(0.45, 0.62)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = rng.uniform(0, 2 * np.pi)
    light = 1.0 + 0.16 * (((xx / W) - 0.5) * np.cos(a) + ((yy / H) - 0.5) * np.sin(a))
    img = (tone + tex) * light
    soft = gaussian_filter(coverage, 0.8)
    depth = rng.uniform(0.55, 0.8)
    img *= 1.0 - depth * np.clip(soft * 1.2, 0, 1)
    img += rng.normal(0, 0.012, size=img.shape)
    tint = np.array([0.95, 1.0, 1.05]) * rng.uniform(0.97, 1.03, 3)
    bgr = np.clip(img[..., None] * tint[None, None, :] * 255, 0, 255).astype(np.uint8)
    return bgr


def generate(
    out_dir: str, n: int = 20, preset: str = "realistic", seed: int = 0, size: Optional[int] = None
) -> List[str]:
    """Write ``n`` samples: ``photos/``, ``masks/``, ``truth/`` (JSON)."""
    spec = PRESETS[preset]
    if size:
        spec = replace(spec, size=size)
    rng = np.random.default_rng(seed)
    for sub in ("photos", "masks", "truth"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    names = []
    for i in range(n):
        s = make_sample(rng, spec)
        name = f"{preset}-{i:03d}"
        cv2.imwrite(os.path.join(out_dir, "photos", name + ".jpg"), s["photo"], [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(out_dir, "masks", name + ".png"), s["mask"].astype(np.uint8) * 255)
        with open(os.path.join(out_dir, "truth", name + ".json"), "w") as f:
            json.dump(s["gt"], f)
        names.append(name)
    return names


__all__ = ["SynthSpec", "PRESETS", "make_sample", "damage", "render_photo", "generate"]

