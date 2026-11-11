"""The tracing pipeline: mask in, measured centrelines out.

1. Clean    binarise the mask, drop specks, fill pinholes
2. Thin     skeletonise each crack (or place corners, for the fallback)
3. Graph    turn the skeleton into ends, junctions and the paths between them
4. Prune    cut the spurs that ragged mask edges leave on a skeleton
5. Bridge   join pieces a segmentation break split apart
6. Extend   carry free ends out to the tip of the mask
7. Split    each crack into a trunk and its branches
8. Smooth   resample at even spacing, smooth, centre across the crack
9. Measure  width across the crack at every point, length, shape
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from . import graph as G
from .config import TraceConfig
from .corners import corner_graph
from .geometry import cross_widths, normals, resample, smooth
from .mask import binarize, clean, distance, to_uint8
from .result import Crack, Path, TraceResult

ArrayLike = Union[np.ndarray, str]
SMOOTH_PER_RADIUS = 1.0
RECENTRE_SMOOTH = 2.0


def _radius_fn(dt: np.ndarray):
    H, W = dt.shape

    def radius(xy: np.ndarray) -> float:
        x, y = int(round(xy[0])), int(round(xy[1]))
        x0, x1 = max(0, x - 1), min(W, x + 2)
        y0, y1 = max(0, y - 1), min(H, y + 2)
        if x0 >= x1 or y0 >= y1:
            return 1.0
        return float(max(dt[y0:y1, x0:x1].max(), 0.5))

    return radius


def _to_gray(image: Optional[np.ndarray], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if image is None:
        return None
    img = to_uint8(np.asarray(image))
    if img.ndim == 3:
        img = cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_BGR2GRAY)
    if img.shape != shape:
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return img


def coverage(g: G.Graph, comp: np.ndarray, dt: np.ndarray) -> float:
    """Share of a mask component that discs along the traced path rebuild
    (each disc as wide as the crack where it sits)."""
    area = int(comp.sum())
    if area == 0:
        return 1.0
    H, W = comp.shape
    pts = [e.pts for e in g.edges.values()] + [xy[None] for xy in g.nodes.values()]
    if not pts:
        return 0.0
    P = np.round(np.vstack(pts)).astype(int)
    P = P[(P[:, 0] >= 0) & (P[:, 0] < W) & (P[:, 1] >= 0) & (P[:, 1] < H)]
    P = np.unique(P, axis=0)
    canvas = np.zeros((H, W), np.uint8)
    for (x, y), r in zip(P, dt[P[:, 1], P[:, 0]]):
        cv2.circle(canvas, (int(x), int(y)), int(np.ceil(r + 1.0)), 1, -1)
    return float((canvas.astype(bool) & comp).sum() / area)


def trace_component(
    comp: np.ndarray,
    dt: np.ndarray,
    gray: Optional[np.ndarray],
    cfg: TraceConfig,
) -> Tuple[G.Graph, str]:
    radius = _radius_fn(dt)
    if cfg.method in ("auto", "skeleton"):
        skel = skeletonize(comp)
        g = G.from_skeleton(skel)
        G.prune(g, radius, cfg.prune, cfg.prune_min)
        if cfg.method == "skeleton":
            return g, "skeleton"
        area = int(comp.sum())
        good = bool(g.edges) or area < 4 * max(cfg.min_area, 1)
        if good and (area < 200 or coverage(g, comp, dt) >= cfg.fallback_coverage):
            return g, "skeleton"
    g = corner_graph(comp, dt, gray, cfg.corner_quality, cfg.corner_min_distance)
    G.prune(g, radius, cfg.prune, cfg.prune_min)
    return g, "corners"


def _loops(g: G.Graph, nodes, edges) -> int:
    return max(0, len(edges) - len(nodes) + 1)


def _finish_path(
    pts: np.ndarray,
    flags: np.ndarray,
    mask: np.ndarray,
    dt: np.ndarray,
    cfg: TraceConfig,
    closed: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample, smooth and centre one path; measure its width."""
    p1, f1 = resample(pts, 1.0, flags)
    H, W = mask.shape
    if cfg.smoothing > 0 and len(p1) >= 3:
        # ragged edges wiggle a wide crack's skeleton on the scale of its
        # width, so wide cracks are smoothed more (never less than the setting)
        ix = np.clip(np.round(p1[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.round(p1[:, 1]).astype(int), 0, H - 1)
        r_med = float(np.median(dt[iy, ix]))
        sigma = max(cfg.smoothing, SMOOTH_PER_RADIUS * r_med)
        p1 = smooth(p1, sigma, closed=closed and len(p1) > 6)
    else:
        sigma = 0.0
    if len(p1) >= 2:
        nrm = normals(p1)
        ix = np.clip(np.round(p1[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.round(p1[:, 1]).astype(int), 0, H - 1)
        reach = 2.5 * dt[iy, ix] + 4.0
        width, offset = cross_widths(mask, p1, nrm, reach)
        on_bridge = np.zeros(len(p1), bool)
        if len(f1):
            on_bridge[:-1] |= f1
            on_bridge[1:] |= f1
        if cfg.recentre:
            ok = np.isfinite(offset) & ~on_bridge & (np.abs(offset) <= 0.5 * np.nan_to_num(width))
            shift = np.where(ok, offset, 0.0)
            if len(shift) > 2:
                # spread the shift smoothly, keep the ends and junctions fixed
                # only the slow part: a skeleton's half-pixel lean, not the
                # ragged edge it would copy back in
                shift = ndi.gaussian_filter1d(shift, max(1.0, RECENTRE_SMOOTH * sigma), mode="nearest")
                shift[0] = shift[-1] = 0.0
            p1 = p1 + shift[:, None] * nrm
        width = np.where(on_bridge, np.nan, width)
    else:
        width = np.full(len(p1), np.nan)
    if cfg.step != 1.0 and len(p1) >= 2:
        out, fo = resample(p1, cfg.step, f1)
        s_old = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p1, axis=0), axis=1))])
        s_new = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(out, axis=0), axis=1))])
        wv = np.where(np.isfinite(width), width, np.nan)
        good = np.isfinite(wv)
        w_out = np.interp(s_new, s_old[good], wv[good]) if good.sum() >= 2 else np.full(len(out), np.nan)
        fo = fo if fo is not None else np.zeros(len(out) - 1, bool)
        # a bridge crosses a gap in the mask: it has no width to report
        gap = np.zeros(len(out), bool)
        gap[:-1] |= fo
        gap[1:] |= fo
        w_out = np.where(gap, np.nan, w_out)
        return out, w_out, fo
    return p1, width, f1 if f1 is not None else np.zeros(max(len(p1) - 1, 0), bool)


def _nearest_on(poly: np.ndarray, p: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(poly - p, axis=1)
    return poly[int(np.argmin(d))]


def trace(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    config: Optional[TraceConfig] = None,
    name: str = "",
    **overrides: Any,
) -> TraceResult:
    """Trace every crack in a mask.

    ``mask`` is any crack mask or probability map (see :func:`mask.to_float`).
    ``image``, the photo the mask came from, is optional: the corner tracer
    looks for corners in it and the overlays draw on it.
    """
    cfg = config or TraceConfig()
    if overrides:
        cfg = cfg.with_(**overrides)
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    m = mask if (isinstance(mask, np.ndarray) and mask.dtype == bool) else binarize(mask, cfg.threshold, cfg.hysteresis)
    m, med_w = clean(m, cfg.min_area, cfg.fill_holes)
    H, W = m.shape
    dt = distance(m)
    gray = _to_gray(image, (H, W))
    timings["clean"] = time.perf_counter() - t0

    # ---- trace each mask component ------------------------------------
    t1 = time.perf_counter()
    n, labels = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    slices = ndi.find_objects(labels)
    areas = np.bincount(labels.ravel(), minlength=n)
    g = G.Graph()
    for k, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        y0, y1 = max(0, sl[0].start - 2), min(H, sl[0].stop + 2)
        x0, x1 = max(0, sl[1].start - 2), min(W, sl[1].stop + 2)
        comp = labels[y0:y1, x0:x1] == k
        gc = gray[y0:y1, x0:x1] if gray is not None else None
        sub, _ = trace_component(comp, dt[y0:y1, x0:x1] * comp, gc, cfg)
        g.absorb(sub, (x0, y0))
    timings["trace"] = time.perf_counter() - t1

    # ---- join, extend, split ------------------------------------------
    t2 = time.perf_counter()
    radius = _radius_fn(dt)
    if g.edges:
        P = np.round(np.vstack([e.pts for e in g.edges.values()])).astype(int)
        P[:, 0] = np.clip(P[:, 0], 0, W - 1)
        P[:, 1] = np.clip(P[:, 1], 0, H - 1)
        r_vals = dt[P[:, 1], P[:, 0]]
        r_vals = r_vals[r_vals > 0]
        if r_vals.size:
            med_w = float(max(1.0, 2.0 * np.median(r_vals) - 1.0))
    bridges: List[Tuple[np.ndarray, np.ndarray]] = []
    if cfg.bridge:
        bridges = G.bridge(g, radius, cfg.bridge_max_gap, cfg.bridge_max_angle, cfg.min_length)
        g.merge_degree2()
    for nodes, edges in g.components():
        traced = sum(
            float(np.linalg.norm(np.diff(g.edges[e].pts, axis=0), axis=1)[~g.edges[e].bridged].sum())
            for e in edges
            if len(g.edges[e].pts) > 1
        )
        if not edges or traced < cfg.min_length:
            for nid in nodes:
                g.remove_node(nid)
    if cfg.extend_ends:
        G.extend_ends(g, m, radius)
    timings["graph"] = time.perf_counter() - t2

    # ---- paths and measurements ---------------------------------------
    t3 = time.perf_counter()
    cracks: List[Crack] = []
    comps = sorted(g.components(), key=lambda ne: -sum(g.edges[e].length for e in ne[1]))
    for nodes, edges in comps:
        if not edges:
            continue
        specs = G.decompose(g, nodes, edges)
        # widths are read on a crop around this crack, not the whole image
        allp = np.vstack([g.edges[e].pts for e in edges])
        ix = np.clip(np.round(allp[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.round(allp[:, 1]).astype(int), 0, H - 1)
        margin = int(np.ceil(2.5 * float(dt[iy, ix].max()) + 8))
        cx0, cy0 = max(0, int(allp[:, 0].min()) - margin), max(0, int(allp[:, 1].min()) - margin)
        cx1, cy1 = min(W, int(allp[:, 0].max()) + margin + 1), min(H, int(allp[:, 1].max()) + margin + 1)
        mf = m[cy0:cy1, cx0:cx1].astype(np.float32)
        dtc = dt[cy0:cy1, cx0:cx1]
        off = np.array([cx0, cy0], float)
        methods = set()
        for e in edges:
            methods |= set(g.edges[e].methods) - {"bridge"}
        method = "+".join(sorted(methods)) if methods else "skeleton"
        paths: List[Path] = []
        for pid, spec in enumerate(specs):
            pts, flags = G.assemble(g, spec)
            closed = len(spec.steps) == 1 and g.edges[spec.steps[0][0]].u == g.edges[spec.steps[0][0]].v
            if spec.parent is not None and paths:
                # start the branch on its parent's finished line
                pts = pts.copy()
                pts[0] = _nearest_on(paths[spec.parent].points, pts[0])
            P, Wd, F = _finish_path(pts - off, flags, mf, dtc, cfg, closed)
            P = P + off
            if spec.parent is not None and len(P):
                P[0] = _nearest_on(paths[spec.parent].points, P[0])
            paths.append(Path(pid, spec.parent, spec.order, P, Wd, F))
        junctions = [g.nodes[nid] for nid in nodes if g.degree(nid) >= 3]
        endpoints = [g.nodes[nid] for nid in nodes if g.degree(nid) == 1]
        # junction neighbourhoods: the width there is the junction, not a crack
        if junctions:
            J = np.array(junctions)
            rj = np.array([radius(j) for j in J])
            for p in paths:
                if not len(p.points):
                    continue
                gap = np.linalg.norm(p.points[:, None, :] - J[None], axis=2) - (1.5 * rj[None, :] + 2.0)
                p.width = np.where(gap.min(axis=1) <= 0, np.nan, p.width)
        pix = np.vstack([np.round(p.points).astype(int) for p in paths])
        pix[:, 0] = np.clip(pix[:, 0], 0, W - 1)
        pix[:, 1] = np.clip(pix[:, 1], 0, H - 1)
        labs = np.unique(labels[pix[:, 1], pix[:, 0]])
        labs = labs[labs > 0]
        area = int(areas[labs].sum()) if len(labs) else 0
        cracks.append(
            Crack(
                id=len(cracks) + 1,
                method=method,
                paths=paths,
                junctions=np.array(junctions).reshape(-1, 2),
                endpoints=np.array(endpoints).reshape(-1, 2),
                loops=_loops(g, nodes, edges),
                area=area,
            )
        )
    timings["measure"] = time.perf_counter() - t3
    timings["total"] = time.perf_counter() - t0
    return TraceResult(
        shape=(H, W),
        cracks=cracks,
        px_size=cfg.px_size,
        unit=cfg.unit,
        width_classes=tuple(cfg.width_classes),
        timings=timings,
        median_width=med_w,
        bridges=bridges,
        mask=m,
        name=name,
        config=cfg.to_dict(),
    )


__all__ = ["trace", "trace_component", "coverage"]

