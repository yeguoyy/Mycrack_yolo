"""Scoring traced centrelines against a reference.

Centreline scores follow the usual buffer method for thin structures (as in
road-network extraction): a traced point is correct when it lies within
``tol`` px of the reference line, and a reference point is found when a traced
point lies within ``tol`` px of it.

- precision: share of the traced line that lies on the reference
- recall:    share of the reference line that the trace covers
- f1:        harmonic mean of the two
- mean_dist: mean distance from traced points to the reference, px
- hd95:      95th percentile of the distances both ways, px
- length_error: (traced length - reference length) / reference length
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .geometry import length as poly_length
from .geometry import resample


def densify(polys: Iterable[np.ndarray], step: float = 1.0, widths: Optional[Iterable[np.ndarray]] = None):
    """Evenly spaced points along every polyline (and widths with them)."""
    pts, ws = [], []
    wl = list(widths) if widths is not None else None
    for k, p in enumerate(polys):
        p = np.asarray(p, float)
        if len(p) == 0:
            continue
        if len(p) == 1:
            pts.append(p)
            if wl is not None:
                ws.append(np.asarray(wl[k], float)[:1])
            continue
        q, _ = resample(p, step)
        pts.append(q)
        if wl is not None:
            w = np.asarray(wl[k], float)
            s_old = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
            s_new = np.linspace(0, s_old[-1], len(q))
            good = np.isfinite(w)
            ws.append(np.interp(s_new, s_old[good], w[good]) if good.sum() >= 2 else np.full(len(q), np.nan))
    P = np.vstack(pts) if pts else np.zeros((0, 2))
    if wl is None:
        return P
    return P, (np.concatenate(ws) if ws else np.zeros(0))


def centreline_scores(pred: np.ndarray, gt: np.ndarray, tol: float = 3.0) -> Dict[str, float]:
    pred = np.asarray(pred, float).reshape(-1, 2)
    gt = np.asarray(gt, float).reshape(-1, 2)
    if len(gt) == 0 and len(pred) == 0:
        return dict(precision=1.0, recall=1.0, f1=1.0, mean_dist=0.0, hd95=0.0)
    if len(pred) == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0, mean_dist=float("nan"), hd95=float("nan"))
    if len(gt) == 0:
        return dict(precision=0.0, recall=1.0, f1=0.0, mean_dist=float("nan"), hd95=float("nan"))
    d_pg, _ = cKDTree(gt).query(pred)
    d_gp, _ = cKDTree(pred).query(gt)
    precision = float(np.mean(d_pg <= tol))
    recall = float(np.mean(d_gp <= tol))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return dict(
        precision=precision,
        recall=recall,
        f1=float(f1),
        mean_dist=float(np.mean(d_pg)),
        hd95=float(np.percentile(np.concatenate([d_pg, d_gp]), 95)),
    )


def length_error(pred_len: float, gt_len: float) -> float:
    return float((pred_len - gt_len) / gt_len) if gt_len > 0 else float("nan")


def point_scores(pred: np.ndarray, gt: np.ndarray, tol) -> Dict[str, float]:
    """Greedy one-to-one matching of points (junctions, ends) within ``tol``
    (one value, or one per reference point)."""
    pred = np.asarray(pred, float).reshape(-1, 2)
    gt = np.asarray(gt, float).reshape(-1, 2)
    if len(gt) == 0 and len(pred) == 0:
        return dict(precision=1.0, recall=1.0)
    if len(gt) == 0 or len(pred) == 0:
        return dict(precision=float(len(pred) == 0), recall=float(len(gt) == 0))
    D = np.linalg.norm(pred[:, None] - gt[None], axis=2)
    T = np.broadcast_to(np.asarray(tol, float).reshape(1, -1), D.shape) if np.ndim(tol) else tol
    pairs = sorted(zip(*np.nonzero(D <= T)), key=lambda ij: D[ij])
    used_p, used_g, tp = set(), set(), 0
    for i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    return dict(precision=tp / len(pred), recall=tp / len(gt))


def width_error(
    pred_pts: np.ndarray, pred_w: np.ndarray, gt_pts: np.ndarray, gt_w: np.ndarray, tol: float = 3.0
) -> Dict[str, float]:
    ok = np.isfinite(pred_w)
    if ok.sum() == 0 or len(gt_pts) == 0:
        return dict(width_mae=float("nan"), width_bias=float("nan"))
    d, j = cKDTree(gt_pts).query(pred_pts[ok])
    m = d <= tol
    if not m.any():
        return dict(width_mae=float("nan"), width_bias=float("nan"))
    err = pred_w[ok][m] - gt_w[j[m]]
    return dict(width_mae=float(np.mean(np.abs(err))), width_bias=float(np.mean(err)))


# ---------------------------------------------------------------------------
# reading traces and references
# ---------------------------------------------------------------------------


def load_lines(path: str) -> Dict[str, Any]:
    """Polylines (and widths, junctions, ends when known) from a trace JSON,
    a synthetic ground-truth JSON or a centreline image."""
    if str(path).lower().endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        return lines_from_dict(d)
    import cv2
    from skimage.morphology import skeletonize

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    from .geometry import smooth
    from .graph import from_skeleton

    sk = skeletonize(img > 127)
    ys, xs = np.nonzero(sk)
    pts = np.column_stack([xs, ys]).astype(float)
    # length along each run of the line, smoothed like a trace is, so a
    # diagonal is neither cut short (pixel count) nor staircased
    g = from_skeleton(sk)
    total = 0.0
    for e in g.edges.values():
        q, _ = resample(e.pts, 1.0)
        total += poly_length(smooth(q, 2.0, closed=e.u == e.v and len(q) > 6))
    return dict(points=pts, widths=None, junctions=None, endpoints=None, length=total, polylines=None)


def lines_from_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    polys, widths, junctions, endpoints = [], [], [], []
    for c in d.get("cracks", []):
        for p in c.get("paths", []):
            polys.append(np.asarray(p["points"], float))
            w = p.get("width", p.get("width_px"))
            widths.append(np.array([np.nan if v is None else v for v in w], float) if w is not None else None)
        # trace JSON keeps counts in "junctions" and points in "junction_points";
        # ground-truth JSON keeps the points in "junctions"
        for key, acc in (("junction", junctions), ("endpoint", endpoints)):
            pts = c.get(f"{key}_points", c.get(f"{key}s"))
            if isinstance(pts, list):
                acc += pts
    has_w = bool(widths) and all(w is not None for w in widths)
    if has_w:
        pts, ws = densify(polys, 1.0, widths)
    else:
        pts, ws = densify(polys, 1.0), None
    return dict(
        points=pts,
        widths=ws,
        junctions=np.asarray(junctions, float).reshape(-1, 2),
        endpoints=np.asarray(endpoints, float).reshape(-1, 2),
        length=float(sum(poly_length(p) for p in polys)),
        polylines=polys,
        cracks=len(d.get("cracks", [])),
    )


def evaluate(pred: Dict[str, Any], gt: Dict[str, Any], tol: float = 3.0, node_tol: float = 8.0) -> Dict[str, float]:
    """Every score that both sides have the data for.

    Junctions count as found within ``node_tol`` px or two local crack
    widths, whichever is larger."""
    out: Dict[str, float] = {}
    out.update(centreline_scores(pred["points"], gt["points"], tol))
    out["length_error"] = length_error(pred["length"], gt["length"])
    if pred.get("widths") is not None and gt.get("widths") is not None:
        out.update(width_error(pred["points"], pred["widths"], gt["points"], gt["widths"], tol))
    if pred.get("junctions") is not None and gt.get("junctions") is not None:
        # a branch leaves a wide crack's band some way past where its own
        # line starts, so a junction counts as found within two crack widths
        jt = node_tol
        if gt.get("widths") is not None and len(gt["junctions"]) and len(gt["points"]):
            _, k = cKDTree(gt["points"]).query(gt["junctions"])
            jt = np.maximum(node_tol, 2.0 * np.nan_to_num(gt["widths"][k]))
        js = point_scores(pred["junctions"], gt["junctions"], jt)
        out["junction_precision"], out["junction_recall"] = js["precision"], js["recall"]
    if pred.get("cracks") is not None and gt.get("cracks"):
        out["pieces_per_crack"] = pred["cracks"] / gt["cracks"]
    return out


def summarize(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float))})
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and r[k] is not None], float)
        v = v[np.isfinite(v)]
        if v.size:
            out[k] = float(np.mean(v))
    return out


__all__ = [
    "densify",
    "centreline_scores",
    "length_error",
    "point_scores",
    "width_error",
    "load_lines",
    "lines_from_dict",
    "evaluate",
    "summarize",
]

