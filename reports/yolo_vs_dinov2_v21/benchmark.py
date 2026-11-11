"""Compare the two early-stopped checkpoints on the same V2.1 test set."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import ultralytics
import ultralytics.nn.tasks as tasks
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from dinov2.dinov2_backbone import DINOv2Pyramid

tasks.DINOv2Pyramid = DINOv2Pyramid
RUNS = {"YOLO26n-OBB": "crack_yolo26n_448_v21-2", "DINOv2-OBB": "crack_yolo26n_dinov2_obb_v21"}
DATASET = ROOT / "data/crack_obb_auto_v21"
PREDICT = dict(imgsz=448, device=0, batch=1, quantize=32, rect=False, conf=.25,
               iou=.7, max_det=300, verbose=False, save=False, augment=False)
EVAL = dict(imgsz=448, batch=8, conf=.01, iou=.7, quantize=32, rect=False)
METRICS = dict(precision="metrics/precision(B)", recall="metrics/recall(B)",
               map50="metrics/mAP50(B)", map50_95="metrics/mAP50-95(B)")


def training_log(run):
    with (run / "results.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    best = max(rows, key=lambda r: r[METRICS["map50_95"]])
    def summarize(row):
        return {"epoch": int(row["epoch"]), "seconds": row["time"],
                **{k: row[v] for k, v in METRICS.items()}}
    return {"best": summarize(best), "last": summarize(rows[-1]),
            "median_seconds_per_epoch": float(np.median(np.diff([r["time"] for r in rows])))}


def describe(samples):
    return {"mean": float(np.mean(samples)), "p50": float(np.median(samples)),
            "p95": float(np.percentile(samples, 95))}


def evaluate(model, run, weight):
    metrics = model.val(data=str(DATASET / "data.yaml"), split="test", device=0,
                        workers=0, plots=False, verbose=False, augment=False,
                        project=str(HERE / "eval"), name=run.name, exist_ok=True, **EVAL)
    instances = int(np.asarray(metrics.nt_per_class).sum())
    assert instances == 1890
    return {"weights": str(weight), "checkpoint_bytes": weight.stat().st_size,
            "training_log": training_log(run),
            "test": {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                     "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
                     "instances": instances, "val_batch_speed_ms_per_image": dict(metrics.speed)}}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU benchmark.")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    paths = sorted(p for p in (DATASET / "images/test").iterdir() if p.is_file())
    assert len(paths) == 822
    chosen = [paths[i] for i in np.linspace(0, len(paths) - 1, 64, dtype=int)]
    images = [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) for p in chosen]
    assert all(image is not None for image in images)
    report = {"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
              "ultralytics": ultralytics.__version__, "cuda": torch.version.cuda,
              "data_yaml": str(DATASET / "data.yaml"), "split": "test", "test_images": len(paths),
              "eval_settings": EVAL,
              "prediction_settings": PREDICT, "timing_samples": [p.name for p in chosen],
              "timing_note": "RAM BGR input to Results; includes preprocessing, GPU transfer, model and postprocess; no disk I/O, display or saving.",
              "models": {}}
    report_path = HERE / "benchmark.json"
    models, timings = {}, {}
    for name, run_name in RUNS.items():
        run = ROOT / "runs" / run_name
        weight = run / "weights/best.pt"
        model = YOLO(str(weight), task="obb")
        nparams = sum(p.numel() for p in model.model.parameters())
        print(f"Evaluating {name} on all {len(paths)} test images", flush=True)
        entry = evaluate(model, run, weight)
        report["models"][name] = {**entry, "params": nparams}
        models[name] = model
        timings[name] = {k: [] for k in ["end_to_end", "inference", "preprocess", "postprocess"]}

    names = list(models)
    for name in names:
        for image in images[:20]:
            models[name].predict(image, **PREDICT)
    print("Measuring warmed batch-1 latency: 64 images, two passes, alternating model order", flush=True)
    for repeat in range(2):
        for i, image in enumerate(images):
            for name in (names if (i + repeat) % 2 == 0 else names[::-1]):
                torch.cuda.synchronize()
                start = perf_counter()
                result = models[name].predict(image, **PREDICT)[0]
                torch.cuda.synchronize()
                timings[name]["end_to_end"].append((perf_counter() - start) * 1000)
                for k in ["inference", "preprocess", "postprocess"]:
                    timings[name][k].append(float(result.speed[k]))
    for name in names:
        report["models"][name]["latency_ms"] = {k: describe(v) for k, v in timings[name].items()}
        report["models"][name]["latency_samples_ms"] = timings[name]
        print(name, json.dumps({k: v for k, v in report["models"][name].items() if k != "latency_samples_ms"}), flush=True)
    reference_run = ROOT / "runs/crack_yolo26n_448_v21"
    reference_weight = reference_run / "weights/best.pt"
    print("Evaluating the earlier plain YOLO26n-OBB checkpoint for reference", flush=True)
    reference = YOLO(str(reference_weight), task="obb")
    report["earlier_plain_reference"] = evaluate(reference, reference_run, reference_weight)
    print("Earlier plain checkpoint:", json.dumps(report["earlier_plain_reference"]), flush=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
