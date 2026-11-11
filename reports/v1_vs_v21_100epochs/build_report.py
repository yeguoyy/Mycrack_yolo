"""从现有训练记录生成报告数据和插图，不训练模型、不改数据集。"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import markdown
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = HERE / "assets"
FONT = Path("C:/Windows/Fonts/msyh.ttc")
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(FONT)).get_name(),
                     "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
COLORS = {"V1": "#526e90", "V2.1": "#008a80"}
METRICS = {"Precision": "metrics/precision(B)", "Recall": "metrics/recall(B)",
           "mAP50": "metrics/mAP50(B)", "mAP50–95": "metrics/mAP50-95(B)"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


def read_image(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def labels(path, width, height):
    return [np.fromstring(line, sep=" ")[1:].reshape(4, 2) * [width, height]
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect():
    output = {"date": "2026-11-10", "timezone": "Asia/Shanghai", "runs": {}, "sources": {}}
    for version, name in [("V1", "crack_yolo26n_448"), ("V2.1", "crack_yolo26n_448_v21")]:
        run = ROOT / "runs" / name
        rows = [{k: float(v) for k, v in r.items()} for r in read_csv(run / "results.csv")]
        assert [r["epoch"] for r in rows] == list(range(1, 101))
        args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
        ck = torch.load(run / "weights/best.pt", map_location="cpu", weights_only=False)
        best = max(rows, key=lambda r: r[METRICS["mAP50–95"]])
        assert all(abs(ck["train_metrics"][key] - best[key]) < 1e-8 for key in METRICS.values())
        output["runs"][version] = {"directory": str(run), "args_yaml": args,
            "checkpoint_effective_args": {k: ck["train_args"].get(k) for k in
                ["optimizer", "lr0", "momentum", "imgsz", "batch", "seed", "resume", "model"]},
            "parameter_count": sum(p.numel() for p in ck["model"].parameters()),
            "model_yaml": ck["model"].yaml, "ultralytics_version": ck["version"],
            "epoch100": rows[-1], "best_in_100": best, "rows": rows}
        for filename in ["results.csv", "args.yaml", "weights/best.pt", "weights/last.pt"]:
            p = run / filename
            output["sources"][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

    a, b = output["runs"]["V1"], output["runs"]["V2.1"]
    assert a["model_yaml"] == b["model_yaml"]
    output["learning_rate_sequences_identical"] = all(
        all(x[k] == y[k] for k in ["lr/pg0", "lr/pg1", "lr/pg2"])
        for x, y in zip(a["rows"], b["rows"]))
    output["delta_percentage_points"] = {scope: {label: (b[scope][key] - a[scope][key]) * 100
        for label, key in METRICS.items()} for scope in ["epoch100", "best_in_100"]}

    rows1 = read_csv(ROOT / "data/crack_obb_auto_v1/manifest.csv")
    rows21 = {r["sample_id"]: r for r in read_csv(ROOT / "data/crack_obb_auto_v21/manifest.csv")}
    assert set(rows21) == {r["sample_id"] for r in rows1}
    assert all((r["split"], r["source_group"]) == (rows21[r["sample_id"]]["split"], rows21[r["sample_id"]]["source_group"]) for r in rows1)
    output["sample_ids_and_group_splits_identical"] = True
    output["dataset_geometry"] = {}
    for version in ["v1", "v21"]:
        long, short, counts, per_split, nonrect = [], [], [], Counter(), 0
        for r in rows1:
            q = rows21[r["sample_id"]]
            w, h = (512, 512) if version == "v1" else (int(q["output_width"]), int(q["output_height"]))
            boxes = labels(ROOT / f"data/crack_obb_auto_{version}/labels" / r["split"] / (r["sample_id"] + ".txt"), w, h)
            counts.append(len(boxes))
            for box in boxes:
                e = np.roll(box, -1, axis=0) - box
                lens = np.linalg.norm(e, axis=1)
                error = max(abs(float(e[i] @ e[(i + 1) % 4])) / max(float(lens[i] * lens[(i + 1) % 4]), 1e-9) for i in range(4))
                nonrect += error > 0.001
                _, (x, y), _ = cv2.minAreaRect(box.astype(np.float32))
                long.append(max(x, y) * 448 / max(w, h))
                short.append(min(x, y) * 448 / max(w, h))
                per_split[r["split"]] += 1
        output["dataset_geometry"][version] = {"images": len(counts), "boxes": sum(counts),
            "boxes_per_split": dict(per_split), "boxes_per_image_mean": float(np.mean(counts)),
            "boxes_per_image_median": float(np.median(counts)), "max_boxes_per_image": max(counts),
            "long_median_at448": float(np.median(long)), "short_median_at448": float(np.median(short)),
            "long_under16_pct": float(np.mean(np.array(long) < 16) * 100),
            "short_under4_pct": float(np.mean(np.array(short) < 4) * 100),
            "nonorthogonal_quads": int(nonrect), "nonorthogonal_quad_pct": nonrect / sum(counts) * 100}
    for rel in ["data/crack_obb_auto_v1/summary.json", "data/crack_obb_auto_v21/summary.json",
                "data/crack_obb_auto_v21/validation.json", "data/crack512/manifest.csv",
                "prepare/prepare_crack_obb.py", "prepare/prepare_crack_obb_v2.py", "output/val_metrics.txt"]:
        p = ROOT / rel
        output["sources"][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    output["independent_v21_validation"] = (ROOT / "output/val_metrics.txt").read_text(encoding="utf-8")
    (HERE / "comparison_data.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output, rows21


def charts(data):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.1), constrained_layout=True)
    for ax, (label, key) in zip(axes.flat, METRICS.items()):
        for version, run in data["runs"].items():
            ax.plot([r["epoch"] for r in run["rows"]], [100 * r[key] for r in run["rows"]],
                    color=COLORS[version], label=version, linewidth=1.5)
        ax.set(title=label, xlabel="训练轮数", ylabel="%", xlim=(1, 100))
        ax.grid(alpha=.17)
        ax.legend(frameon=False, loc="lower right")
    fig.suptitle("相同模型训练 100 轮：各自版本标签上的验证曲线", fontsize=15)
    fig.savefig(ASSETS / "training_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    x = np.arange(len(METRICS))
    for offset, version in [(-.19, "V1"), (.19, "V2.1")]:
        ys = [data["runs"][version]["epoch100"][key] * 100 for key in METRICS.values()]
        bars = ax.bar(x + offset, ys, .34, label=version, color=COLORS[version])
        ax.bar_label(bars, labels=[f"{y:.2f}" for y in ys], padding=4, fontsize=10)
    for i, label in enumerate(METRICS):
        ax.text(i, 77, f"+{data['delta_percentage_points']['epoch100'][label]:.2f} 个百分点", ha="center", color="#326b63", fontsize=10)
    ax.set(xticks=x, xticklabels=list(METRICS), ylim=(0, 86), ylabel="%", title="第 100 轮验证指标（使用各自版本的 GT）")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, .84))
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    fig.savefig(ASSETS / "epoch100_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set(xlim=(0, 12), ylim=(0, 4.5))
    ax.axis("off")
    lanes = [(3.0, COLORS["V1"], ["绿色 mask\n过滤小区域", "8 连通区域\n每区视为一目标", "轮廓最小面积矩形\n短边至少 2 px", "逐点裁剪角点\n归一化输出"]),
             (1.0, COLORS["V2.1"], ["绿色 mask\n骨架与主干/分支", "平滑走向\n30° 转向切大段", "像素分配与同向合并\n拟合旋转矩形", "补灰边保留方向\n标签与几何校验"])]
    for y, color, texts in lanes:
        ax.text(.15, y + .75, "V1" if y > 2 else "V2.1", color=color, weight="bold", fontsize=13)
        for i, text in enumerate(texts):
            left = .15 + i * 3
            ax.add_patch(FancyBboxPatch((left, y - .35), 2.55, .95, boxstyle="round,pad=0.1", linewidth=1.1, edgecolor=color, facecolor="#f2f6f7"))
            ax.text(left + 1.275, y + .1, text, ha="center", va="center", fontsize=10)
            if i < 3:
                ax.add_patch(FancyArrowPatch((left + 2.65, y + .1), (left + 2.9, y + .1), arrowstyle="-|>", mutation_scale=12, color=color))
    fig.suptitle("两种标签构建流程", fontsize=15)
    fig.savefig(ASSETS / "construction_flow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def label_examples(rows21):
    font = ImageFont.truetype(str(FONT), 19)
    small = ImageFont.truetype(str(FONT), 16)
    canvas = Image.new("RGB", (1260, 1350), "#eef1f4")
    for rowidx, sample in enumerate(["IMG_6637_35", "IMG_6637_18", "IMG_8010_50"]):
        r = rows21[sample]
        source = read_image(Path(r["image_path"]))
        mask = read_image(Path(r["mask_path"]))
        left, top, right, bottom = json.loads(r["padding_ltrb"])
        original = cv2.copyMakeBorder(source, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        green = np.all(mask == [0, 128, 0], axis=-1)
        overlay = source.copy()
        overlay[green] = (.25 * overlay[green] + .75 * np.array([70, 220, 80])).astype(np.uint8)
        overlay = cv2.copyMakeBorder(overlay, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        old = labels(ROOT / "data/crack_obb_auto_v1/labels" / r["split"] / (sample + ".txt"), 512, 512)
        old = [p + [left, top] for p in old]
        new = labels(ROOT / "data/crack_obb_auto_v21/labels" / r["split"] / (sample + ".txt"), original.shape[1], original.shape[0])
        a, b = original.copy(), original.copy()
        for img, boxes, color in [(a, old, (224, 119, 49)), (b, new, (110, 190, 0))]:
            for j, box in enumerate(boxes, 1):
                cv2.polylines(img, [np.rint(box).astype(np.int32)], True, color, 2, cv2.LINE_AA)
                center = tuple(np.rint(box.mean(axis=0)).astype(int))
                cv2.putText(img, str(j), center, cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, str(j), center, cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        for col, (img, title) in enumerate([(overlay, "原始 mask"), (a, f"V1：{len(old)} 个框"), (b, f"V2.1：{len(new)} 个框")]):
            panel = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            panel.thumbnail((400, 390), Image.Resampling.LANCZOS)
            x, y = col * 420, rowidx * 450
            canvas.paste(panel, (x + (420 - panel.width) // 2, y + 47 + (390 - panel.height) // 2))
            d = ImageDraw.Draw(canvas)
            d.text((x + 12, y + 4), title, font=font, fill="#172c3f")
            d.text((x + 12, y + 27), sample, font=small, fill="#566574")
    canvas.save(ASSETS / "label_examples.jpg", quality=94)


def html_report():
    body = markdown.markdown((HERE / "report.md").read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "toc"])
    for image in ASSETS.iterdir():
        mime = "image/png" if image.suffix == ".png" else "image/jpeg"
        body = body.replace(image.as_posix(), f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode()}")
    css = """body{max-width:1080px;margin:40px auto;padding:0 30px 70px;color:#243442;background:#fff;font:16px/1.85 'Microsoft YaHei',sans-serif}h1{font-size:29px;line-height:1.5;color:#123b50}h2{font-size:23px;margin-top:42px;border-bottom:2px solid #dbe6ec;padding-bottom:6px}h3{font-size:18px;color:#20556b;margin-top:28px}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}th,td{padding:9px 12px;border:1px solid #dce4e9;text-align:left}th{background:#edf4f7}tr:nth-child(even){background:#f8fafb}img{max-width:100%;height:auto;display:block;margin:24px auto}blockquote{margin:20px 0;padding:12px 20px;border-left:4px solid #008a80;background:#f0f8f6}pre{padding:15px;overflow:auto;background:#f2f5f7;border-radius:6px;font-size:13px}code{overflow-wrap:anywhere}a{color:#087781}li{margin:5px 0}@media print{body{max-width:none;margin:0;padding:0;font-size:11pt}h2,h3{break-after:avoid}tr,blockquote,img{break-inside:avoid}a{color:inherit;text-decoration:none}img{max-height:22cm;object-fit:contain}pre{white-space:pre-wrap}}"""
    page = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V1 与 V2.1 裂缝 OBB 100 轮训练对比报告</title><style>{css}</style></head><body>{body}</body></html>'
    (HERE / "report.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    data, rows21 = collect()
    charts(data)
    label_examples(rows21)
    html_report()
    print(json.dumps({"report": str(HERE / 'report.md'), "html": str(HERE / 'report.html'),
                      "epoch100_delta_pp": data['delta_percentage_points']['epoch100'],
                      "best_delta_pp": data['delta_percentage_points']['best_in_100'],
                      "lr_sequences_identical": data['learning_rate_sequences_identical']}, ensure_ascii=False, indent=2))
