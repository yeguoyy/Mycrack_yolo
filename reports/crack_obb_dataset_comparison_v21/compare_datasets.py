"""只读比较磁盘上的 V1/V2/V2.1 标签，输出报告所用统计与示例图。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHINESE_FONT = Path("C:/Windows/Fonts/msyh.ttc")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "prepare"))
from prepare_crack_obb_v2 import (
    ROOT, audit_boxes, crack_mask, imread, imwrite, read_rows,
)

VERSIONS = {"V1": "crack_obb_auto_v1", "V2": "crack_obb_auto_v2", "V2.1": "crack_obb_auto_v21"}


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_boxes(path, width, height):
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        z = np.fromstring(line, sep=" ")
        if len(z) != 9 or z[0] != 0 or not np.isfinite(z).all():
            raise ValueError(f"Invalid label: {path}")
        if ((z[1:] < 0) | (z[1:] > 1)).any():
            raise ValueError(f"Out-of-bounds label: {path}")
        boxes.append(z[1:].reshape(4, 2) * [width, height])
    return boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/crack_obb_dataset_comparison_v21")
    args = parser.parse_args()
    out = args.output.resolve()
    data_root = (ROOT / "data").resolve()
    if out == data_root or data_root in out.parents:
        parser.error("报告输出必须位于数据集目录之外")
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(ROOT / "data/crack512/manifest.csv")
    manifests = {v: {r["sample_id"]: r for r in load_csv(ROOT / "data" / folder / "manifest.csv")}
                 for v, folder in VERSIONS.items()}
    for manifest in manifests.values():
        assert set(manifest) == {r["sample_id"] for r in rows}
        for row in rows:
            assert all(manifest[row["sample_id"]][k] == row[k] for k in ["split", "source_group"])
    accum = {v: {"boxes": 0, "splits": Counter(), "background": 0, "hits": 0,
                 "long": [], "short": [], "nonrectangular": 0, "nonrectangular_images": 0,
                 "coverage_below_99_5_images": 0, "label_sha256": hashlib.sha256()}
             for v in VERSIONS}
    per_image = []
    total_kept = total_ignored = 0
    example_ids = {"IMG_6637_35", "IMG_6926_50", "IMG_8010_50"}
    for i, row in enumerate(rows, 1):
        binary = crack_mask(imread(Path(row["mask_path"])))
        h, w = binary.shape
        _, labels, areas, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        allowed = areas[:, cv2.CC_STAT_AREA] >= 12
        allowed[0] = False
        kept = allowed[labels]
        total_kept += int(kept.sum())
        total_ignored += int(binary.sum() - kept.sum())
        r21 = manifests["V2.1"][row["sample_id"]]
        left, top, right, bottom = json.loads(r21["padding_ltrb"])
        assert (w, h) == (int(r21["original_width"]), int(r21["original_height"]))
        item = {"sample_id": row["sample_id"], "split": row["split"], "source_group": row["source_group"]}
        display_boxes = {}
        for version, folder in VERSIONS.items():
            native_w, native_h = (w + left + right, h + top + bottom) if version == "V2.1" else (w, h)
            path = ROOT / "data" / folder / "labels" / row["split"] / f"{Path(row['image_path']).stem}.txt"
            boxes = read_boxes(path, native_w, native_h)
            a = accum[version]
            a["label_sha256"].update((row["split"] + "/" + path.name + "\n").encode())
            a["label_sha256"].update(path.read_bytes())
            a["boxes"] += len(boxes)
            a["splits"][row["split"]] += len(boxes)
            offset = np.array([left, top])
            on_original = [b - offset for b in boxes] if version == "V2.1" else boxes
            display_boxes[version] = boxes if version == "V2.1" else [b + offset for b in boxes]
            audit = audit_boxes(on_original, kept)
            hits = round(audit["coverage"] * int(kept.sum()))
            a["hits"] += hits
            a["background"] += audit["background_pixels"]
            a["coverage_below_99_5_images"] += audit["coverage"] < 0.995
            item[f"{version}_boxes"] = len(boxes)
            item[f"{version}_coverage"] = audit["coverage"]
            item[f"{version}_background_pixels"] = audit["background_pixels"]
            broken = 0
            for b in boxes:
                _, (bw, bh), _ = cv2.minAreaRect(b.astype(np.float32))
                scale = 448 / max(native_w, native_h)
                a["long"].append(max(bw, bh) * scale)
                a["short"].append(min(bw, bh) * scale)
                e = np.roll(b, -1, axis=0) - b
                lens = np.linalg.norm(e, axis=1)
                cosine = np.abs(np.sum(e * np.roll(e, -1, axis=0), axis=1)) / np.maximum(lens * np.roll(lens, -1), 1e-9)
                broken += int(lens.min() < 1e-5 or cosine.max() > 0.001)
            a["nonrectangular"] += broken
            a["nonrectangular_images"] += broken > 0
        per_image.append(item)
        if row["sample_id"] in example_ids:
            image = imread(ROOT / "data/crack_obb_auto_v21/images" / row["split"] / Path(row["image_path"]).name)
            panels = []
            for version, boxes in display_boxes.items():
                panel = image.copy()
                for j, b in enumerate(boxes, 1):
                    cv2.polylines(panel, [np.rint(b).astype(np.int32)], True, (0, 190, 255), 2, cv2.LINE_AA)
                    center = tuple(np.rint(b.mean(axis=0)).astype(int))
                    cv2.putText(panel, str(j), center, cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(panel, str(j), center, cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
                panel = cv2.copyMakeBorder(panel, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 32, 36))
                title_image = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
                ImageDraw.Draw(title_image).text(
                    (12, 10), f"{version}：{len(boxes)} 个框 | {row['sample_id']}",
                    font=ImageFont.truetype(str(CHINESE_FONT), 18), fill=(245, 245, 245),
                )
                panel = cv2.cvtColor(np.asarray(title_image), cv2.COLOR_RGB2BGR)
                panels.append(panel)
            imwrite(out / "images" / f"{row['sample_id']}_three_versions.jpg", np.hstack(panels))
        if i % 1000 == 0 or i == len(rows):
            print(f"已比较标签与掩码：{i}/{len(rows)}", flush=True)
    stats = {}
    for version, a in accum.items():
        lengths, widths = np.array(a["long"]), np.array(a["short"])
        stats[version] = {
            "boxes": a["boxes"], "boxes_per_image": a["boxes"] / len(rows), "boxes_by_split": dict(a["splits"]),
            "coverage_on_filtered_mask": a["hits"] / total_kept,
            "background_pixels_inside_original_images": a["background"],
            "coverage_below_99_5_images": a["coverage_below_99_5_images"],
            "nonrectangular_label_polygons": a["nonrectangular"], "images_with_nonrectangular_polygons": a["nonrectangular_images"],
            "at_448_fit_long_edge": {"long_median_px": float(np.median(lengths)),
                "long_under16_pct": float(np.mean(lengths < 16) * 100),
                "long_under32_pct": float(np.mean(lengths < 32) * 100),
                "short_median_px": float(np.median(widths)),
                "short_under4_pct": float(np.mean(widths < 4) * 100)},
            "labels_sha256_in_manifest_order": a["label_sha256"].hexdigest(),
        }
    result = {"samples": len(rows), "image_splits": dict(Counter(r["split"] for r in rows)),
              "source_groups": len({r["source_group"] for r in rows}), "kept_mask_pixels": total_kept,
              "ignored_small_component_pixels": total_ignored, "versions": stats,
              "notes": ["覆盖率与框并集背景仅在共同原图范围内统计。",
                        "V2.1 在栅格化前先扣除补边引入的坐标平移。",
                        "448 尺寸按各版本自身图片估算，不包含随机增强与额外步幅对齐补边。",
                        "V1 裁剪后的四边形先用 minAreaRect 重拟合，再统计长宽。",
                        "掩码覆盖率为几何覆盖，不代表精确率、召回率或标注正确率。"]}
    (out / "statistics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "per_image_geometry.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    colors = ["#8796a5", "#d99759", "#2a9d8f"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=160)
    values = [[stats[v]["boxes"] for v in VERSIONS],
              [stats[v]["at_448_fit_long_edge"]["long_under16_pct"] for v in VERSIONS]]
    for ax, vals, title, suffix in zip(axes, values, ["标注框总数（5,462 张图像）", "缩放至 448 后，长边不足 16 像素的框"], ["", "%"]):
        bars = ax.bar(list(VERSIONS), vals, color=colors, width=.58)
        ax.set_title(title, fontsize=11, pad=16)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            label = f"{val:,.0f}" if not suffix else f"{val:.2f}%"
            ax.text(bar.get_x() + bar.get_width()/2, val + max(vals)*.025, label, ha="center", fontsize=10)
    fig.tight_layout()
    (out / "images").mkdir(exist_ok=True)
    fig.savefig(out / "images/dataset_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
