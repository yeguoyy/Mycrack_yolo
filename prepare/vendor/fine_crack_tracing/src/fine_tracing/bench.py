"""Benchmark: every tracer on the same synthetic cracks, scored against the
centrelines they were drawn from.

Writes ``results.csv`` (one row per image and tracer), ``summary.csv`` and
``summary.md`` (means per preset), ``width_sweep.csv`` (single cracks at set
widths) and, when matplotlib is installed, charts of both.
"""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .config import TraceConfig
from .metrics import densify, evaluate, lines_from_dict
from .result import finite

TRACERS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "v0.1 ORB, y-order": ("legacy", dict(detection="orb", sort="classic")),
    "v0.1 Shi-Tomasi + MST": ("legacy", dict(detection="shitomasi", sort="MST")),
    "plain skeleton": (
        "v1",
        dict(
            method="skeleton",
            prune=0.0,
            prune_min=0.0,
            bridge=False,
            extend_ends=False,
            smoothing=0.0,
            recentre=False,
            min_length=0.0,
            fill_holes=0,
        ),
    ),
    "v1 corners": ("v1", dict(method="corners")),
    "v1 auto": ("v1", dict()),
}

SCORES = (
    "f1",
    "precision",
    "recall",
    "mean_dist",
    "hd95",
    "length_error",
    "abs_length_error",
    "width_mae",
    "junction_recall",
    "junction_precision",
    "pieces_per_crack",
    "ms",
)


def run_tracer(kind: str, opts: Dict[str, Any], mask: np.ndarray, photo: np.ndarray) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    if kind == "legacy":
        from . import legacy

        polys = [p for p in legacy.trace(mask, photo, **opts) if len(p)]
        ms = (time.perf_counter() - t0) * 1000
        pts = densify(polys)
        L = float(sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in polys if len(p) > 1))
        return dict(points=pts, widths=None, junctions=None, length=L, cracks=len(polys)), ms
    from .pipeline import trace

    res = trace(mask, photo, TraceConfig().with_(**opts))
    ms = (time.perf_counter() - t0) * 1000
    d = res.to_dict()
    return lines_from_dict(d), ms


def _score_sample(job) -> List[Dict[str, Any]]:
    from .synth import PRESETS, SynthSpec, make_sample

    preset, idx, seed, size, spec_over = job
    spec = PRESETS.get(preset, SynthSpec())
    spec = replace(spec, size=size, **(spec_over or {}))
    rng = np.random.default_rng(seed)
    s = make_sample(rng, spec)
    gt = lines_from_dict(s["gt"])
    mask = s["mask"].astype(np.uint8) * 255
    rows = []
    for name, (kind, opts) in TRACERS.items():
        pred, ms = run_tracer(kind, opts, mask, s["photo"])
        sc = evaluate(pred, gt, tol=3.0, node_tol=8.0)
        sc["abs_length_error"] = abs(sc["length_error"])
        sc["ms"] = ms
        if pred.get("junctions") is None:
            sc.pop("junction_recall", None)
            sc.pop("junction_precision", None)
        rows.append(dict(preset=preset, image=idx, tracer=name, **sc, **(spec_over or {})))
    return rows


def _mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    v = np.array([r[key] for r in rows if key in r and r[key] is not None], float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def run(
    out: str,
    n: int = 25,
    size: int = 768,
    seed: int = 7,
    presets: Sequence[str] = ("clean", "ragged", "broken", "realistic"),
    workers: int = 1,
    sweep_widths: Sequence[float] = (1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32),
    sweep_n: int = 8,
) -> Dict[str, Any]:
    os.makedirs(out, exist_ok=True)
    jobs = [(p, i, seed * 100003 + k * 1009 + i, size, None) for k, p in enumerate(presets) for i in range(n)]
    sweep_jobs = [
        (
            "ragged",
            i,
            seed * 7919 + int(w * 100) * 31 + i,
            size,
            dict(width=(float(w), float(w)), cracks=(1, 1), branch_p=0.0),
        )
        for w in sweep_widths
        for i in range(sweep_n)
    ]
    rows: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_score_sample, jobs):
                rows += r
            for r in ex.map(_score_sample, sweep_jobs):
                sweep_rows += r
    else:
        for j in jobs:
            rows += _score_sample(j)
        for j in sweep_jobs:
            sweep_rows += _score_sample(j)
    for r in sweep_rows:
        r["sweep_width"] = r.pop("width")[0]
        r.pop("cracks", None)
        r.pop("branch_p", None)

    _write_csv(os.path.join(out, "results.csv"), rows)
    _write_csv(os.path.join(out, "width_sweep.csv"), sweep_rows)

    tracers = list(TRACERS)
    summary = []
    for p in list(presets) + ["all"]:
        for t in tracers:
            sel = [r for r in rows if r["tracer"] == t and (p == "all" or r["preset"] == p)]
            summary.append(dict(preset=p, tracer=t, **{k: _mean(sel, k) for k in SCORES}))
    _write_csv(os.path.join(out, "summary.csv"), summary)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(_markdown(summary, presets, n, size))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(
            finite(
                dict(
                    images_per_preset=n,
                    size=size,
                    seed=seed,
                    seconds=round(time.perf_counter() - t0, 1),
                    summary=summary,
                )
            ),
            f,
            indent=1,
            allow_nan=False,
        )
    try:
        _charts(out, summary, sweep_rows, presets, tracers)
    except ImportError:  # pragma: no cover
        pass
    return dict(summary=summary, rows=rows, sweep=sweep_rows)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})


def _fmt(v: float, pct: bool = False, nd: int = 2) -> str:
    if not np.isfinite(v):
        return "–"
    return f"{v * 100:+.1f}%" if pct else f"{v:.{nd}f}"


def _markdown(summary, presets, n, size) -> str:
    lines = [
        f"Synthetic benchmark: {n} images per preset, {size} x {size} px, tolerance 3 px.",
        "",
        "| preset | tracer | F1 | mean dist (px) | length error | width error (px) | junctions found | ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        lines.append(
            f"| {r['preset']} | {r['tracer']} | {_fmt(r['f1'])} | {_fmt(r['mean_dist'])} | "
            f"{_fmt(r['length_error'], pct=True)} | {_fmt(r['width_mae'])} | "
            f"{_fmt(r['junction_recall'] * 100, nd=0) + '%' if np.isfinite(r['junction_recall']) else '–'} | "
            f"{_fmt(r['ms'], nd=0)} |"
        )
    return "\n".join(lines) + "\n"


def _charts(out, summary, sweep_rows, presets, tracers) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#9aa0a6", "#c5a3ff", "#7fb3d5", "#f2d14b", "#3ddc97"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(presets))
    wbar = 0.8 / len(tracers)
    for k, t in enumerate(tracers):
        f1 = [next(r["f1"] for r in summary if r["preset"] == p and r["tracer"] == t) for p in presets]
        le = [next(r["abs_length_error"] for r in summary if r["preset"] == p and r["tracer"] == t) for p in presets]
        axes[0].bar(x + k * wbar - 0.4 + wbar / 2, f1, wbar, label=t, color=colors[k % len(colors)])
        axes[1].bar(x + k * wbar - 0.4 + wbar / 2, np.array(le) * 100, wbar, label=t, color=colors[k % len(colors)])
    axes[0].set_title("Centreline F1 at 3 px (higher is better)")
    axes[1].set_title("Length error, % (lower is better)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(presets)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "by_preset.png"), dpi=150)
    plt.close(fig)

    if sweep_rows:
        widths = sorted({r["sweep_width"] for r in sweep_rows})
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for k, t in enumerate(tracers):
            md = [
                _mean([r for r in sweep_rows if r["tracer"] == t and r["sweep_width"] == w], "mean_dist")
                for w in widths
            ]
            le = [
                _mean([r for r in sweep_rows if r["tracer"] == t and r["sweep_width"] == w], "abs_length_error")
                for w in widths
            ]
            axes[0].plot(widths, md, "o-", label=t, color=colors[k % len(colors)])
            axes[1].plot(widths, np.array(le) * 100, "o-", label=t, color=colors[k % len(colors)])
        axes[0].set_title("Distance from the true centreline, px")
        axes[1].set_title("Length error, %")
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlabel("crack width, px")
            ax.spines[["top", "right"]].set_visible(False)
        axes[1].set_yscale("log")
        axes[0].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "width_sweep.png"), dpi=150)
        plt.close(fig)


__all__ = ["TRACERS", "run", "run_tracer"]

