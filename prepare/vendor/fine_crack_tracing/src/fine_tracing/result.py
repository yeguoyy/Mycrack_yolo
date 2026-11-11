"""What a tracing run returns, and how it is written out."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import principal_angle


def _r(a, nd: int = 2):
    return np.round(np.asarray(a, float), nd).tolist()


def _num(v, nd: int = 3):
    """A finite number rounded, or None (JSON has no NaN)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if np.isfinite(f) else None


def finite(obj):
    """Replace every NaN or infinity in a JSON-bound structure with None."""
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


@dataclass
class Path:
    """One run of a crack: the trunk, or a branch off another path."""

    id: int
    parent: Optional[int]
    order: int
    points: np.ndarray  # (N, 2) x, y
    width: np.ndarray  # (N,) px across the crack; NaN where not measured
    bridged: np.ndarray  # (N-1,) True on segments that cross a gap in the mask

    @property
    def seg_lengths(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.points, axis=0), axis=1)

    @property
    def length(self) -> float:
        return float(self.seg_lengths.sum()) if len(self.points) > 1 else 0.0

    @property
    def gap_length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(self.seg_lengths[self.bridged].sum())

    def width_filled(self) -> np.ndarray:
        """The width profile with unmeasured points (junctions, bridges)
        interpolated from their neighbours, for drawing."""
        w = self.width
        ok = np.isfinite(w)
        if ok.all() or not ok.any():
            return np.nan_to_num(w)
        s = np.concatenate([[0.0], np.cumsum(self.seg_lengths)])
        return np.interp(s, s[ok], w[ok])


@dataclass
class Crack:
    """One connected crack: a trunk and its branches."""

    id: int
    method: str
    paths: List[Path]
    junctions: np.ndarray
    endpoints: np.ndarray
    loops: int
    area: int

    # ---- size ---------------------------------------------------------
    @property
    def length(self) -> float:
        return float(sum(p.length for p in self.paths))

    @property
    def gap_length(self) -> float:
        return float(sum(p.gap_length for p in self.paths))

    @property
    def gaps(self) -> int:
        """How many breaks in the mask this crack was joined across."""
        n = 0
        for p in self.paths:
            b = p.bridged.astype(int)
            n += int(np.sum(np.diff(np.concatenate([[0], b])) == 1))
        return n

    @property
    def trunk(self) -> Path:
        return self.paths[0]

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        pts = np.vstack([p.points for p in self.paths])
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)

    # ---- width --------------------------------------------------------
    def _width_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        ws, wt = [], []
        for p in self.paths:
            if len(p.points) < 2:
                continue
            seg = p.seg_lengths * ~p.bridged
            weight = np.zeros(len(p.points))
            weight[:-1] += seg / 2
            weight[1:] += seg / 2
            ok = np.isfinite(p.width) & (weight > 0)
            ws.append(p.width[ok])
            wt.append(weight[ok])
        if not ws:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(ws), np.concatenate(wt)

    def width_stats(self) -> Dict[str, float]:
        w, wt = self._width_samples()
        if w.size == 0 or wt.sum() <= 0:
            return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
        order = np.argsort(w)
        cw = np.cumsum(wt[order]) / wt.sum()

        def q(f: float) -> float:
            return float(w[order][min(int(np.searchsorted(cw, f)), len(w) - 1)])

        # the max is read from a lightly filtered profile so one stray sample
        # at a ragged mask edge does not set it
        peaks = []
        for p in self.paths:
            v = p.width[np.isfinite(p.width)]
            if v.size >= 5:
                k = 5
                pad = np.pad(v, k // 2, mode="edge")
                win = np.lib.stride_tricks.sliding_window_view(pad, k)
                peaks.append(np.median(win, axis=1).max())
            elif v.size:
                peaks.append(np.median(v))
        return {
            "mean": float(np.average(w, weights=wt)),
            "median": q(0.5),
            "p95": q(0.95),
            "max": float(max(peaks)) if peaks else float(w.max()),
        }

    def width_class_lengths(self, edges_px: Sequence[float]) -> List[float]:
        """Crack length whose width falls in each class [e_i, e_i+1).

        Bridged stretches are left out; junction zones take the width of the
        crack on either side."""
        bounds = [0.0, *edges_px, np.inf]
        out = [0.0] * (len(bounds) - 1)
        for p in self.paths:
            if len(p.points) < 2:
                continue
            seg = p.seg_lengths
            # junction zones are not measured; class them by the width either side
            wf = p.width_filled() if np.isfinite(p.width).any() else p.width
            wmid = 0.5 * (wf[:-1] + wf[1:])
            for k in range(len(out)):
                sel = (~p.bridged) & np.isfinite(wmid) & (wmid >= bounds[k]) & (wmid < bounds[k + 1])
                out[k] += float(seg[sel].sum())
        return out

    # ---- shape --------------------------------------------------------
    @property
    def orientation(self) -> float:
        return principal_angle(self.trunk.points)

    @property
    def direction(self) -> str:
        a = self.orientation
        if a < 22.5 or a >= 157.5:
            return "horizontal"
        if 67.5 <= a < 112.5:
            return "vertical"
        return "diagonal"

    @property
    def tortuosity(self) -> float:
        t = self.trunk.points
        chord = float(np.linalg.norm(t[-1] - t[0])) if len(t) > 1 else 0.0
        return float(self.trunk.length / chord) if chord > 1e-6 else float("nan")

    @property
    def pattern(self) -> str:
        if self.loops > 0:
            return "network"
        return "branched" if len(self.paths) > 1 else "single"


@dataclass
class TraceResult:
    """Everything traced in one image."""

    shape: Tuple[int, int]
    cracks: List[Crack]
    px_size: Optional[float] = None
    unit: str = "mm"
    width_classes: Tuple[float, ...] = ()
    timings: Dict[str, float] = field(default_factory=dict)
    median_width: float = 0.0
    bridges: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    # ---- totals ------------------------------------------------------
    @property
    def length(self) -> float:
        return float(sum(c.length for c in self.cracks))

    @property
    def max_width(self) -> float:
        vals = [c.width_stats()["max"] for c in self.cracks]
        vals = [v for v in vals if np.isfinite(v)]
        return float(max(vals)) if vals else float("nan")

    def all_points(self, step: Optional[float] = None) -> np.ndarray:
        if not self.cracks:
            return np.zeros((0, 2))
        return np.vstack([p.points for c in self.cracks for p in c.paths])

    def _scale(self, v: float) -> Optional[float]:
        if self.px_size is None or v is None or not np.isfinite(v):
            return None
        return float(v * self.px_size)

    # ---- export ------------------------------------------------------
    def crack_row(self, c: Crack) -> Dict[str, Any]:
        ws = c.width_stats()
        row: Dict[str, Any] = {
            "image": self.name,
            "crack": c.id,
            "method": c.method,
            "pattern": c.pattern,
            "length_px": round(c.length, 2),
            "gaps": c.gaps,
            "gap_px": round(c.gap_length, 2),
            "mean_width_px": _num(ws["mean"]),
            "median_width_px": _num(ws["median"]),
            "p95_width_px": _num(ws["p95"]),
            "max_width_px": _num(ws["max"]),
            "branches": len(c.paths) - 1,
            "junctions": len(c.junctions),
            "endpoints": len(c.endpoints),
            "loops": c.loops,
            "orientation_deg": round(c.orientation, 1),
            "direction": c.direction,
            "tortuosity": round(c.tortuosity, 4) if np.isfinite(c.tortuosity) else None,
            "area_px": c.area,
        }
        if self.px_size is not None:
            u = self.unit
            row[f"length_{u}"] = round(self._scale(c.length), 4)
            for k in ("mean", "median", "p95", "max"):
                v = self._scale(ws[k])
                row[f"{k}_width_{u}"] = round(v, 4) if v is not None else None
        if self.width_classes:
            scale = self.px_size or 1.0
            edges_px = [e / scale for e in self.width_classes]
            labels = self._class_labels()
            for lab, L in zip(labels, c.width_class_lengths(edges_px)):
                row[f"length_{lab}"] = round(L * scale, 4)
        return row

    def _class_labels(self) -> List[str]:
        u = self.unit if self.px_size is not None else "px"
        e = list(self.width_classes)
        labels = [f"w<{e[0]:g}{u}"]
        labels += [f"w{a:g}-{b:g}{u}" for a, b in zip(e[:-1], e[1:])]
        labels.append(f"w>={e[-1]:g}{u}")
        return labels

    def rows(self) -> List[Dict[str, Any]]:
        return [self.crack_row(c) for c in self.cracks]

    def to_dict(self, geometry: bool = True) -> Dict[str, Any]:
        from . import __version__

        H, W = self.shape
        cracks = []
        for c in self.cracks:
            d = self.crack_row(c)
            d.pop("image", None)
            d["id"] = d.pop("crack")
            d["bbox"] = _r(c.bbox, 1)
            d["junction_points"] = _r(c.junctions)
            d["endpoint_points"] = _r(c.endpoints)
            if geometry:
                d["paths"] = [
                    {
                        "id": p.id,
                        "parent": p.parent,
                        "order": p.order,
                        "length_px": round(p.length, 2),
                        "points": _r(p.points),
                        "width_px": [None if not np.isfinite(v) else round(float(v), 2) for v in p.width],
                        "bridged": p.bridged.astype(int).tolist(),
                    }
                    for p in c.paths
                ]
            cracks.append(d)
        summary: Dict[str, Any] = {
            "cracks": len(self.cracks),
            "length_px": round(self.length, 2),
            "max_width_px": round(self.max_width, 3) if np.isfinite(self.max_width) else None,
            "median_mask_width_px": round(self.median_width, 2),
            "bridged_gaps": sum(c.gaps for c in self.cracks),
        }
        if self.px_size is not None:
            summary[f"length_{self.unit}"] = round(self._scale(self.length), 4)
            mw = self._scale(self.max_width)
            summary[f"max_width_{self.unit}"] = round(mw, 4) if mw is not None else None
        return {
            "image": self.name,
            "width": W,
            "height": H,
            "px_size": self.px_size,
            "unit": self.unit if self.px_size is not None else "px",
            "summary": summary,
            "cracks": cracks,
            "timing_ms": {k: round(v * 1000, 2) for k, v in self.timings.items()},
            "config": self.config,
            "version": __version__,
        }

    def to_json(self, path: Optional[str] = None, geometry: bool = True) -> str:
        s = json.dumps(finite(self.to_dict(geometry)), indent=1, allow_nan=False)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    def to_geojson(self) -> Dict[str, Any]:
        """Paths as LineStrings in image pixel coordinates (x right, y down)."""
        feats = []
        for c in self.cracks:
            for p in c.paths:
                w = p.width[np.isfinite(p.width)]
                props = {
                    "crack": c.id,
                    "path": p.id,
                    "parent": p.parent,
                    "order": p.order,
                    "method": c.method,
                    "length_px": round(p.length, 2),
                    "gap_px": round(p.gap_length, 2),
                    "mean_width_px": round(float(w.mean()), 3) if w.size else None,
                    "max_width_px": round(float(w.max()), 3) if w.size else None,
                }
                if self.px_size is not None:
                    props[f"length_{self.unit}"] = round(p.length * self.px_size, 4)
                feats.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": _r(p.points)},
                        "properties": props,
                    }
                )
            for j in c.junctions:
                feats.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": _r(j)},
                        "properties": {"crack": c.id, "kind": "junction"},
                    }
                )
        return {"type": "FeatureCollection", "name": self.name, "features": feats}

    def to_svg(self, background: Optional[str] = None, line: float = 0.0) -> str:
        from .render import width_color

        H, W = self.shape
        lw = line or max(1.5, max(H, W) / 700)
        wmax = self._color_scale()
        out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
        ]
        if background:
            out.append(f'<image href="{background}" x="0" y="0" width="{W}" height="{H}"/>')
        for c in self.cracks:
            out.append(f'<g id="crack-{c.id}" fill="none" stroke-linecap="round" stroke-linejoin="round">')
            for p in c.paths:
                for seg_pts, bridged, wv in _runs(p, wmax):
                    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in seg_pts)
                    if bridged:
                        out.append(
                            f'<polyline points="{pts}" stroke="#ffffff" stroke-width="{lw:.2f}" '
                            f'stroke-dasharray="{3 * lw:.1f} {2 * lw:.1f}" opacity="0.85"/>'
                        )
                    else:
                        b, g_, r = width_color(wv)
                        out.append(f'<polyline points="{pts}" stroke="rgb({r},{g_},{b})" stroke-width="{lw:.2f}"/>')
            for x, y in c.junctions:
                out.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.2 * lw:.1f}" fill="none" '
                    f'stroke="#ffffff" stroke-width="{0.6 * lw:.2f}"/>'
                )
            out.append("</g>")
        out.append("</svg>")
        return "\n".join(out)

    def _color_scale(self) -> float:
        """Top of the colour scale: the 97th percentile of width, so one wide
        spalled patch does not wash every other crack out to the thin end."""
        ws = [c._width_samples() for c in self.cracks]
        w = np.concatenate([a for a, _ in ws]) if ws else np.zeros(0)
        wt = np.concatenate([b for _, b in ws]) if ws else np.zeros(0)
        if w.size == 0 or wt.sum() <= 0:
            return 1.0
        order = np.argsort(w)
        cw = np.cumsum(wt[order]) / wt.sum()
        top = float(w[order][min(int(np.searchsorted(cw, 0.97)), len(w) - 1)])
        return top if top > 0 else 1.0


def _runs(p: Path, wmax: float, bins: int = 24):
    """Split a path into runs of one colour bin (or one bridge) for SVG."""
    if len(p.points) < 2:
        return
    wf = p.width_filled()
    wm = 0.5 * (wf[:-1] + wf[1:])
    ok = np.isfinite(wm)
    q = np.full(len(wm), -1, dtype=int)
    q[ok] = np.clip(np.floor(wm[ok] / wmax * bins), 0, bins).astype(int)
    key = np.where(p.bridged, -2, q)
    start = 0
    for i in range(1, len(key) + 1):
        if i == len(key) or key[i] != key[start]:
            seg = p.points[start : i + 1]
            if key[start] == -2:
                yield seg, True, 0.0
            else:
                vals = wm[start:i]
                v = float(np.nanmean(vals)) / wmax if np.isfinite(vals).any() else 0.0
                yield seg, False, v
            start = i


__all__ = ["Path", "Crack", "TraceResult"]

