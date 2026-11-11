"""The ``fine-tracing`` command.

fine-tracing trace masks/ --photos photos/ -o results/ --px-size 0.1
fine-tracing eval results/json truth/ -o scores/
fine-tracing synth data/synthetic -n 40 --preset realistic
fine-tracing bench -o bench/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .config import METHODS, TraceConfig
from .result import finite

FORMATS = ("overlay", "json", "csv", "geojson", "svg", "centreline")

EXAMPLES = """examples:
  fine-tracing trace masks/ --photos photos/ -o results/
  fine-tracing trace masks/ -o results/ --px-size 0.08 --width-classes 0.1,0.3,1
  fine-tracing eval results/json reference/ --tol 3 -o scores/
  fine-tracing synth data/synthetic -n 40 --preset realistic
  fine-tracing bench -o bench/ -j 8

docs: https://github.com/Kronbii/fine-crack-tracing#readme
"""


def _floats(s: str) -> List[float]:
    return [float(v) for v in s.replace(" ", "").split(",") if v]


def config_from_args(a: argparse.Namespace) -> TraceConfig:
    base: Dict[str, Any] = {}
    if getattr(a, "config", None):
        with open(a.config) as f:
            base = json.load(f)
    cfg = TraceConfig.from_dict(base)
    changes: Dict[str, Any] = {}
    for name, attr in (
        ("method", "method"),
        ("px_size", "px_size"),
        ("unit", "unit"),
        ("threshold", "threshold"),
        ("bridge_max_gap", "max_gap"),
        ("bridge_max_angle", "max_angle"),
        ("smoothing", "smoothing"),
        ("min_length", "min_length"),
        ("min_area", "min_area"),
        ("prune", "prune"),
        ("step", "step"),
    ):
        v = getattr(a, attr, None)
        if v is not None:
            changes[name] = v
    if getattr(a, "hysteresis", None):
        lo, hi = _floats(a.hysteresis)
        changes["hysteresis"] = (lo, hi)
    if getattr(a, "width_classes", None):
        changes["width_classes"] = tuple(_floats(a.width_classes))
    if getattr(a, "no_bridge", False):
        changes["bridge"] = False
    if getattr(a, "no_extend", False):
        changes["extend_ends"] = False
    if getattr(a, "no_recentre", False):
        changes["recentre"] = False
    return cfg.with_(**changes) if changes else cfg


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------


def _trace_one(job) -> Dict[str, Any]:
    import cv2

    from .io import read_image, read_mask
    from .pipeline import trace
    from .render import centreline_image, overlay

    name, mask_path, photo_path, out, cfg_d, formats = job
    cfg = TraceConfig.from_dict(cfg_d)
    mask = read_mask(mask_path)
    photo = read_image(photo_path) if photo_path else None
    res = trace(mask, photo, cfg, name=name)
    if "json" in formats:
        res.to_json(os.path.join(out, "json", name + ".json"))
    if "geojson" in formats:
        with open(os.path.join(out, "geojson", name + ".geojson"), "w") as f:
            json.dump(finite(res.to_geojson()), f, allow_nan=False)
    if "svg" in formats:
        with open(os.path.join(out, "svg", name + ".svg"), "w") as f:
            f.write(res.to_svg())
    if "overlay" in formats:
        img = overlay(res, photo)
        ext = ".jpg" if photo is not None else ".png"
        cv2.imwrite(os.path.join(out, "overlays", name + ext), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if "centreline" in formats:
        cv2.imwrite(os.path.join(out, "centrelines", name + ".png"), centreline_image(res))
    return dict(
        timings={k: round(v * 1000, 1) for k, v in res.timings.items()},
        name=name,
        rows=res.rows(),
        summary=res.to_dict(geometry=False)["summary"],
        ms=res.timings["total"] * 1000,
        unit=res.unit if res.px_size is not None else "px",
        px=res.px_size,
    )


def cmd_trace(a: argparse.Namespace) -> int:
    from .io import pair_inputs

    cfg = config_from_args(a)
    formats = set(_floats_or_names(a.formats))
    bad = formats - set(FORMATS)
    if bad:
        print(f"unknown format(s): {', '.join(sorted(bad))}", file=sys.stderr)
        return 2
    jobs_in = pair_inputs(a.masks, a.photos)
    if not jobs_in:
        print(f"no masks found in {a.masks}", file=sys.stderr)
        return 1
    out = a.output
    for sub, fmt in (
        ("json", "json"),
        ("geojson", "geojson"),
        ("svg", "svg"),
        ("overlays", "overlay"),
        ("centrelines", "centreline"),
    ):
        if fmt in formats:
            os.makedirs(os.path.join(out, sub), exist_ok=True)
    os.makedirs(out, exist_ok=True)
    jobs = [(n, str(m), str(p) if p else None, out, cfg.to_dict(), formats) for n, m, p in jobs_in]
    missing = sum(1 for _, _, p in jobs_in if a.photos and p is None)
    if missing:
        print(f"note: {missing} mask(s) have no matching photo", file=sys.stderr)

    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    workers = a.workers or 1
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_trace_one, jobs):
                results.append(r)
                _report(r, a.quiet, getattr(a, "verbose", 0))
    else:
        for j in jobs:
            r = _trace_one(j)
            results.append(r)
            _report(r, a.quiet, getattr(a, "verbose", 0))
    wall = time.perf_counter() - t0

    rows = [row for r in results for row in r["rows"]]
    if "csv" in formats:
        path = os.path.join(out, "cracks.csv")
        keys: List[str] = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys or ["image"])
            w.writeheader()
            for row in rows:
                w.writerow(row)
    summary = {
        "version": __version__,
        "images": len(results),
        "cracks": len(rows),
        "seconds": round(wall, 3),
        "mean_ms_per_image": round(sum(r["ms"] for r in results) / max(len(results), 1), 1),
        "config": cfg.to_dict(),
        "per_image": {r["name"]: r["summary"] for r in results},
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(finite(summary), f, indent=1, allow_nan=False)
    if not a.quiet:
        print(
            f"\n{len(results)} image(s), {len(rows)} crack(s), {summary['mean_ms_per_image']:.0f} ms per image -> {out}"
        )
    return 0


def _floats_or_names(s: str) -> List[str]:
    return [v.strip() for v in s.split(",") if v.strip()]


def _report(r: Dict[str, Any], quiet: bool, verbose: int = 0) -> None:
    if quiet:
        return
    if verbose:
        print("  " + "  ".join(f"{k} {v:.0f} ms" for k, v in r["timings"].items()), file=sys.stderr)
    s = r["summary"]
    unit = r["unit"]
    if r["px"] is not None:
        L = s.get(f"length_{unit}", 0.0)
        mw = s.get(f"max_width_{unit}")
        tail = f"{L:9.1f} {unit} long   max width {mw if mw is not None else float('nan'):.2f} {unit}"
    else:
        mw = s.get("max_width_px")
        tail = f"{s['length_px']:9.0f} px long   max width {mw if mw is not None else float('nan'):.1f} px"
    print(f"{r['name']:<28} {s['cracks']:3d} crack(s) {tail}   {r['ms']:6.0f} ms")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def cmd_eval(a: argparse.Namespace) -> int:
    from .metrics import evaluate, load_lines, summarize

    pred_dir, gt_dir = Path(a.pred), Path(a.truth)
    exts = (".json", ".png", ".jpg", ".bmp", ".tif")
    gts = {p.stem: p for p in sorted(gt_dir.iterdir()) if p.suffix.lower() in exts}
    preds = {p.stem: p for p in sorted(pred_dir.iterdir()) if p.suffix.lower() in exts}
    common = sorted(set(gts) & set(preds))
    if not common:
        print("no file names in common between the two folders", file=sys.stderr)
        return 1
    rows = []
    for name in common:
        sc = evaluate(load_lines(str(preds[name])), load_lines(str(gts[name])), tol=a.tol)
        rows.append({"image": name, **{k: round(v, 4) for k, v in sc.items()}})
    mean = summarize(rows)
    os.makedirs(a.output, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(a.output, "scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.output, "scores.json"), "w") as f:
        json.dump(finite({"tolerance_px": a.tol, "mean": mean, "images": rows}), f, indent=1, allow_nan=False)
    print(f"{len(rows)} image(s) scored at {a.tol:g} px")
    for k, v in mean.items():
        print(f"  {k:<20} {v:8.4f}")
    return 0


# ---------------------------------------------------------------------------
# synth / bench
# ---------------------------------------------------------------------------


def cmd_synth(a: argparse.Namespace) -> int:
    from .synth import generate

    names = generate(a.output, a.n, a.preset, a.seed, a.size)
    print(f"{len(names)} sample(s) -> {a.output} (photos/, masks/, truth/)")
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    from .bench import run

    run(a.output, n=a.n, size=a.size, seed=a.seed, presets=_floats_or_names(a.presets), workers=a.workers)
    return 0


def cmd_run_legacy(a: argparse.Namespace) -> int:
    """v0.1 spelling: ``run --frames-dir --masks-dir --output-dir``."""
    ns = argparse.Namespace(**vars(a))
    ns.masks, ns.photos, ns.output = a.masks_dir, a.frames_dir, a.output_dir
    ns.formats = ",".join(FORMATS)
    ns.quiet, ns.workers, ns.config = False, 1, None
    print("note: `run` is the v0.1 spelling; use `fine-tracing trace`", file=sys.stderr)
    return cmd_trace(ns)


def cmd_metrics_legacy(a: argparse.Namespace) -> int:
    """v0.1 spelling: ``metrics --ground-truth --generated --output-dir``."""
    print("note: `metrics` is the v0.1 spelling; use `fine-tracing eval`", file=sys.stderr)
    ns = argparse.Namespace(pred=a.generated, truth=a.ground_truth, output=a.output_dir, tol=3.0)
    return cmd_eval(ns)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fine-tracing",
        description="Turn crack masks into measured centrelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    p.add_argument("--version", action="version", version=f"fine-tracing {__version__}")
    # v0.1 took -v/-vv before the subcommand; it is accepted and shows stage timings
    p.add_argument("-v", "--verbose", action="count", default=0, help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="command", required=True, metavar="{trace,eval,synth,bench}")

    t = sub.add_parser("trace", help="trace the cracks in a mask, or a folder of masks")
    t.add_argument("masks", help="mask file or folder")
    t.add_argument("--photos", help="photo file or folder, matched to masks by file name")
    t.add_argument("-o", "--output", default="results", help="output folder (default: results)")
    t.add_argument(
        "--method",
        choices=METHODS,
        help="skeleton, corners, or auto: skeleton with corners for any crack it cannot describe (default)",
    )
    t.add_argument("--px-size", type=float, help="size of one pixel in --unit, e.g. 0.1")
    t.add_argument("--unit", help="unit for --px-size (default: mm)")
    t.add_argument("--width-classes", help="width class edges in output units, e.g. 0.1,0.3,1")
    t.add_argument("--threshold", type=float, help="mask threshold as a fraction (default 0.5)")
    t.add_argument("--hysteresis", help="low,high thresholds for probability maps, e.g. 0.3,0.6")
    t.add_argument("--no-bridge", action="store_true", help="do not join pieces across gaps")
    t.add_argument("--max-gap", type=float, help="longest gap to join, px (default: from crack width)")
    t.add_argument("--max-angle", type=float, help="how far each end may turn to meet, degrees")
    t.add_argument("--smoothing", type=float, help="smoothing along the path, px (0 = off)")
    t.add_argument("--min-length", type=float, help="drop cracks shorter than this, px")
    t.add_argument("--min-area", type=int, help="drop mask specks smaller than this, px")
    t.add_argument("--prune", type=float, help="cut side branches shorter than this many half-widths")
    t.add_argument("--step", type=float, help="spacing of output points, px")
    t.add_argument("--no-extend", action="store_true", help="leave skeleton ends where thinning put them")
    t.add_argument("--no-recentre", action="store_true", help="skip centring points across the crack")
    t.add_argument("--config", help="JSON file of settings (flags override it)")
    t.add_argument("--formats", default=",".join(FORMATS), help=f"outputs to write (default: {','.join(FORMATS)})")
    t.add_argument("-j", "--workers", type=int, default=1, help="images to trace in parallel")
    t.add_argument("-q", "--quiet", action="store_true")
    t.set_defaults(func=cmd_trace)

    e = sub.add_parser("eval", help="score traces against reference centrelines")
    e.add_argument("pred", help="folder of trace JSON files or centreline images")
    e.add_argument("truth", help="folder of reference JSON files or centreline drawings")
    e.add_argument("--tol", type=float, default=3.0, help="distance tolerance, px (default 3)")
    e.add_argument("-o", "--output", default="scores")
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("synth", help="generate synthetic cracks with exact ground truth")
    s.add_argument("output")
    s.add_argument("-n", type=int, default=20)
    s.add_argument("--preset", default="realistic", choices=["clean", "ragged", "broken", "realistic"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--size", type=int, default=None)
    s.set_defaults(func=cmd_synth)

    b = sub.add_parser("bench", help="compare tracers on synthetic cracks")
    b.add_argument("-o", "--output", default="bench")
    b.add_argument("-n", type=int, default=25, help="images per preset")
    b.add_argument("--size", type=int, default=768)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--presets", default="clean,ragged,broken,realistic")
    b.add_argument("-j", "--workers", type=int, default=1)
    b.set_defaults(func=cmd_bench)

    # v0.1 spellings, kept working but out of the help
    r = sub.add_parser("run")
    r.add_argument("--frames-dir", required=True)
    r.add_argument("--masks-dir", required=True)
    r.add_argument("--output-dir", required=True)
    r.set_defaults(func=cmd_run_legacy)

    mt = sub.add_parser("metrics")
    mt.add_argument("--ground-truth", required=True)
    mt.add_argument("--generated", required=True)
    mt.add_argument("--output-dir", default="scores")
    mt.set_defaults(func=cmd_metrics_legacy)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    old_env = sorted(k for k in os.environ if k.startswith("FINE_TRACING_"))
    if old_env:
        print(
            f"note: {', '.join(old_env)} no longer change anything; use command-line flags or --config",
            file=sys.stderr,
        )
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

