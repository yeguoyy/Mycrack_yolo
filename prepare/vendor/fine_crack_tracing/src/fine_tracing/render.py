"""Drawing traced cracks over the photo or the mask."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .mask import to_uint8
from .result import TraceResult

# thin -> wide: mint green, yellow, magenta (RGB)
_STOPS = np.array([0.0, 0.5, 1.0])
_COLORS = np.array([[94, 224, 160], [242, 209, 75], [255, 61, 154]], float)


def width_color(v: float) -> Tuple[int, int, int]:
    """BGR colour for a width as a fraction of the scale maximum."""
    v = float(np.clip(v if np.isfinite(v) else 0.0, 0.0, 1.0))
    rgb = [np.interp(v, _STOPS, _COLORS[:, k]) for k in range(3)]
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def _poly(img, pts, color, thick, closed=False):
    if len(pts) < 2:
        return
    p = np.round(np.asarray(pts) * 4).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], closed, color, thick, cv2.LINE_AA, shift=2)


def _dashed(img, a, b, color, thick, dash):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 1e-6:
        return
    d = (b - a) / L
    s = 0.0
    while s < L:
        e = min(L, s + dash)
        _poly(img, [a + s * d, a + e * d], color, thick)
        s += 2 * dash


def _base(result: TraceResult, image: Optional[np.ndarray], show_mask: bool, dim: float) -> np.ndarray:
    H, W = result.shape
    if image is not None:
        img = to_uint8(np.asarray(image))
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img[..., :3].astype(np.float32)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        img *= dim
        if show_mask and result.mask is not None:
            m = result.mask
            img[m] = 0.72 * img[m] + 0.28 * np.array([235, 235, 235])
        return np.clip(img, 0, 255).astype(np.uint8)
    img = np.full((H, W, 3), (16, 15, 13), np.uint8)
    if show_mask and result.mask is not None:
        img[result.mask] = (78, 66, 74)
    return img


def overlay(
    result: TraceResult,
    image: Optional[np.ndarray] = None,
    show_mask: bool = True,
    labels: bool = True,
    legend: bool = True,
    line: Optional[int] = None,
    dim: float = 0.8,
    wmax: Optional[float] = None,
) -> np.ndarray:
    """The traced cracks drawn over the photo (or the mask on black).

    Lines are coloured by the crack's width at that point, from mint (thin)
    through yellow to magenta (the widest in the image). Dashed white runs
    cross gaps in the mask; white rings mark junctions.
    """
    img = _base(result, image, show_mask, dim)
    H, W = result.shape
    lw = line or max(2, int(round(max(H, W) / 650)))
    scale = wmax or result._color_scale()

    for c in result.cracks:
        for p in c.paths:
            pts = p.points
            if len(pts) < 2:
                continue
            _poly(img, pts, (0, 0, 0), lw + 2)
        for p in c.paths:
            pts = p.points
            if len(pts) < 2:
                continue
            wf = p.width_filled()
            wm = 0.5 * (wf[:-1] + wf[1:])
            for i in range(len(pts) - 1):
                if p.bridged[i]:
                    continue
                col = width_color(wm[i] / scale if np.isfinite(wm[i]) else np.nan)
                _poly(img, pts[i : i + 2], col, lw)
            # bridged runs, dashed
            i = 0
            while i < len(p.bridged):
                if p.bridged[i]:
                    j = i
                    while j < len(p.bridged) and p.bridged[j]:
                        j += 1
                    _dashed(img, pts[i], pts[j], (255, 255, 255), lw, 3 * lw)
                    i = j
                else:
                    i += 1
        for j in c.junctions:
            cv2.circle(
                img,
                tuple(np.round(j * 4).astype(int)),
                int(4 * 2.4 * lw),
                (255, 255, 255),
                max(1, lw // 2),
                cv2.LINE_AA,
                shift=2,
            )
        for e in c.endpoints:
            cv2.circle(
                img, tuple(np.round(e * 4).astype(int)), int(4 * 1.3 * lw), (255, 255, 255), -1, cv2.LINE_AA, shift=2
            )

    if labels:
        _labels(img, result, lw)
    if legend and result.cracks:
        _legend(img, scale, result, lw)
    return img


def _fmt_len(result: TraceResult, v: float) -> str:
    if result.px_size is not None:
        return f"{v * result.px_size:.1f} {result.unit}"
    return f"{v:.0f} px"


def _fmt_w(result: TraceResult, v: float) -> str:
    if not np.isfinite(v):
        return "-"
    if result.px_size is not None:
        return f"{v * result.px_size:.2f} {result.unit}"
    return f"{v:.1f} px"


def _text(img, txt, org, fs, th, placed=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, tht), base = cv2.getTextSize(txt, font, fs, th)
    x, y = org
    H, W = img.shape[:2]
    pad = max(3, th * 2)
    x = int(np.clip(x, pad + 2, max(pad + 2, W - tw - 2 * pad - 4)))
    y = int(np.clip(y, tht + 6, H - base - 4))
    if placed is not None:
        # step down past any label already drawn here
        step = tht + base + 2 * pad + 2
        for _ in range(40):
            box = (x - pad, y - tht - pad, x + tw + pad, y + base + pad)
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
                break
            y += step
            if y > H - base - 4:
                y = tht + 6
        placed.append((x - pad, y - tht - pad, x + tw + pad, y + base + pad))
    sub = img[y - tht - pad : y + base + pad, x - pad : x + tw + pad]
    if sub.size:
        sub[:] = (0.35 * sub).astype(np.uint8)
    cv2.putText(img, txt, (x, y), font, fs, (245, 245, 245), th, cv2.LINE_AA)


def _labels(img, result: TraceResult, lw: int) -> None:
    H, W = result.shape
    fs = max(0.45, max(H, W) / 2200)
    th = max(1, int(round(fs * 1.6)))
    placed: list = []
    for c in result.cracks:
        if c.length < 0.04 * max(H, W) and len(result.cracks) > 6:
            continue
        ws = c.width_stats()
        t = c.trunk.points
        anchor = t[len(t) // 2]
        txt = f"#{c.id}  {_fmt_len(result, c.length)}  max w {_fmt_w(result, ws['max'])}"
        _text(img, txt, (int(anchor[0] + 6 * lw), int(anchor[1])), fs, th, placed)


def _legend(img, scale: float, result: TraceResult, lw: int) -> None:
    H, W = img.shape[:2]
    bw = int(min(W * 0.28, 360 * max(1.0, max(H, W) / 1600)))
    bh = max(8, 3 * lw)
    x0 = int(W * 0.03)
    fs = max(0.4, max(H, W) / 2600)
    th = max(1, int(round(fs * 1.6)))
    y0 = int(H - H * 0.03 - bh - 36 * fs)
    box = img[max(0, y0 - int(40 * fs) - 10) : min(H, y0 + bh + int(40 * fs) + 8), max(0, x0 - 10) : x0 + bw + 10]
    box[:] = (0.3 * box).astype(np.uint8)
    for i in range(bw):
        col = width_color(i / max(bw - 1, 1))
        cv2.line(img, (x0 + i, y0), (x0 + i, y0 + bh), col, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "width across crack", (x0, y0 - int(14 * fs) - 4), font, fs, (230, 230, 230), th, cv2.LINE_AA)
    lab = _fmt_w(result, scale) + ("+" if scale < result.max_width - 1e-6 else "")
    (tw, _), _ = cv2.getTextSize(lab, font, fs, th)
    below = y0 + bh + int(30 * fs) + 4
    if below < H - 2:
        cv2.putText(img, "0", (x0, below), font, fs, (230, 230, 230), th, cv2.LINE_AA)
        cv2.putText(img, lab, (x0 + bw - tw, below), font, fs, (230, 230, 230), th, cv2.LINE_AA)
    else:
        cv2.putText(img, lab, (x0 + bw - tw, y0 - int(14 * fs) - 4), font, fs, (230, 230, 230), th, cv2.LINE_AA)


def centreline_image(result: TraceResult, thickness: int = 1) -> np.ndarray:
    """The traced centrelines as a white-on-black image, for other tools."""
    H, W = result.shape
    img = np.zeros((H, W), np.uint8)
    for c in result.cracks:
        for p in c.paths:
            if len(p.points) >= 2:
                q = np.round(p.points).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [q], False, 255, thickness, cv2.LINE_8)
    return img


__all__ = ["overlay", "centreline_image", "width_color"]

