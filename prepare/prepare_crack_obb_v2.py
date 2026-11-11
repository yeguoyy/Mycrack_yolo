"""用 fine-crack-tracing 提取中心线，按明显转向生成大段 YOLO OBB。

默认使用 relaxed（V2.1），只生成 36 张预览；--full 转换完整数据集。
原始图片、mask 和现有 V1/V2 数据集均只读；--profile v2 可重现旧规则。
运行环境和示例见同目录 README_crack_obb_v2.md。
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
VENDOR = Path(__file__).resolve().parent / "vendor" / "fine_crack_tracing"
sys.path.insert(0, str(VENDOR / "src"))
try:
    from fine_tracing import trace
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少转换依赖。请用 prepare/.venv/Scripts/python.exe 运行，"
        "或先安装同目录 requirements_crack_obb_v2.txt。"
    ) from exc

PIN = "da32dd1ae80f2b6b1a58758033b183e16b6259a2"
GREEN_BGR = np.array([0, 128, 0], dtype=np.uint8)
RED_BGR = np.array([0, 0, 128], dtype=np.uint8)
COLORS = [(60, 200, 255), (255, 160, 60), (170, 70, 255), (70, 240, 110), (230, 110, 210)]


@dataclass
class Settings:
    min_component_pixels: int = 12
    min_box_side: float = 2.0
    min_segment_length: float = 40.0
    max_deviation: float = 8.0
    width_tolerance: float = 1.0
    prune: float = 2.0
    prune_min: float = 3.0
    profile: str = "v2"
    padding: float = 0.0
    merge_angle: float = 25.0
    merge_area_ratio: float = 1.35
    tiny_length: float = 32.0
    split_angle: float = 30.0
    direction_window: float = 48.0
    turn_persistence: float = 16.0


def make_settings(profile: str, max_deviation=None, min_segment_length=None) -> Settings:
    cfg = Settings()
    if profile == "relaxed":
        cfg.profile = "relaxed"
        cfg.min_box_side = 8.0
        cfg.padding = 2.0
        cfg.min_segment_length = 64.0
        cfg.max_deviation = 14.0
        cfg.width_tolerance = 1.5
    if max_deviation is not None:
        cfg.max_deviation = max_deviation
    if min_segment_length is not None:
        cfg.min_segment_length = min_segment_length
    return cfg


def imread(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Cannot encode {path}")
    encoded.tofile(path)


def crack_mask(mask: np.ndarray) -> np.ndarray:
    known = np.all(mask == 0, axis=2) | np.all(mask == GREEN_BGR, axis=2) | np.all(mask == RED_BGR, axis=2)
    if not known.all():
        raise ValueError(f"Mask contains {int((~known).sum())} pixels of unknown colors")
    return np.all(mask == GREEN_BGR, axis=2)


def split_curve(points: np.ndarray, tolerance: float, min_length: float) -> list[np.ndarray]:
    """按到首尾线段的最大偏离递归拆分；最短弧长限制避免噪声造成过度碎片化。"""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return [p]
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]
    p = p[keep]
    if len(p) < 3:
        return [p]
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    valid = np.flatnonzero((lengths >= min_length) & (lengths <= lengths[-1] - min_length))
    if not len(valid):
        return [p]
    vector = p[-1] - p[0]
    scale = float(vector @ vector)
    t = np.clip(((p - p[0]) @ vector) / max(scale, 1e-12), 0, 1)
    distances = np.linalg.norm(p - (p[0] + t[:, None] * vector), axis=1)
    cut = int(valid[np.argmax(distances[valid])])
    if distances[cut] <= tolerance:
        return [p]
    return split_curve(p[: cut + 1], tolerance, min_length) + split_curve(p[cut:], tolerance, min_length)


def path_pieces(crack, cfg: Settings) -> list[np.ndarray]:
    pieces = []
    # 微小毛刺不应把主干切成许多短框；其 mask 像素仍会分配给最近的有效段。
    useful = [p for p in crack.paths if p.parent is None or p.length >= cfg.min_segment_length]
    junctions = [p.points[0] for p in useful if p.parent is not None and len(p.points)]
    for path in useful:
        p = path.points
        if len(p) < 2:
            continue
        finite = path.width[np.isfinite(path.width)]
        local_width = float(np.median(finite)) if len(finite) else 2.0
        tolerance = max(cfg.max_deviation, cfg.width_tolerance * local_width)
        cuts = {0, len(p) - 1}
        for junction in junctions:
            distances = np.linalg.norm(p - junction, axis=1)
            index = int(np.argmin(distances))
            if distances[index] <= max(4.0, local_width):
                cuts.add(index)
        ordered = sorted(cuts)
        for start, end in zip(ordered, ordered[1:]):
            if end > start:
                pieces.extend(split_curve(p[start : end + 1], tolerance, cfg.min_segment_length))
    return [p for p in pieces if len(p) >= 2]


def relaxed_path_pieces(crack, cfg: Settings) -> list[np.ndarray]:
    """每个主干/可见分支按方向变化分大段，不使用离直线的像素偏离阈值。"""
    pieces = []
    for path in crack.paths:
        if len(path.points) < 2:
            continue
        widths = path.width[np.isfinite(path.width)]
        width = float(np.median(widths)) if len(widths) else 2.0
        if path.parent is not None and path.length < max(24.0, 3.0 * width):
            continue
        pieces.extend(split_by_angle(path.points, cfg))
    return [p for p in pieces if len(p) >= 2]


def split_by_angle(points: np.ndarray, cfg: Settings) -> list[np.ndarray]:
    p = np.asarray(points, float)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]
    p = p[keep]
    if len(p) < 3:
        return [p]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if arc[-1] < 2 * cfg.min_segment_length:
        return [p]
    s = np.unique(np.r_[np.arange(0, arc[-1], 2.0), arc[-1]])
    q = np.c_[np.interp(s, arc, p[:, 0]), np.interp(s, arc, p[:, 1])]
    before = np.maximum(0, s - cfg.direction_window / 2)
    after = np.minimum(arc[-1], s + cfg.direction_window / 2)
    v = np.c_[np.interp(after, arc, p[:, 0]) - np.interp(before, arc, p[:, 0]),
              np.interp(after, arc, p[:, 1]) - np.interp(before, arc, p[:, 1])]
    heading = np.degrees(np.unwrap(np.arctan2(v[:, 1], v[:, 0])))

    def reference(start):
        selected = (s >= s[start] + cfg.direction_window / 2) & (s <= s[start] + cfg.direction_window)
        return float(np.median(heading[selected])) if selected.any() else float(heading[start])

    start, run, index = 0, None, 1
    direction = reference(start)
    pieces = []
    while index < len(q) - 1:
        if s[index] - s[start] < cfg.min_segment_length:
            index += 1
            continue
        if s[-1] - s[index] < cfg.min_segment_length:
            break
        delta = abs((heading[index] - direction + 180) % 360 - 180)
        if delta >= cfg.split_angle:
            if run is None:
                run = index
            if s[index] - s[run] >= cfg.turn_persistence:
                pieces.append(q[start:run + 1])
                start = run
                direction = reference(start)
                index, run = start + 1, None
                continue
        else:
            run = None
        index += 1
    pieces.append(q[start:])
    return pieces


def group_geometry(points: np.ndarray, cfg: Settings):
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(hull)
    direction = angle if w >= h else angle + 90.0
    extent = abs(math.cos(math.radians(angle))) + abs(math.sin(math.radians(angle)))
    ew, eh = max(w + extent + 2 * cfg.padding, cfg.min_box_side), max(h + extent + 2 * cfg.padding, cfg.min_box_side)
    return {"hull": hull, "direction": direction, "length": max(w, h) + extent, "area": ew * eh,
            "minimum": points.min(axis=0), "maximum": points.max(axis=0)}


def merge_pixel_groups(groups: list[np.ndarray], cfg: Settings) -> tuple[list[np.ndarray], int]:
    """仅合并同一连通区域内相接的段；面积膨胀限制防止把明显分叉重新装进大框。"""
    if len(groups) < 2:
        return groups, 0
    all_pixels = np.vstack(groups)
    origin = all_pixels.min(axis=0)
    size = all_pixels.max(axis=0) - origin + 1
    owner = np.zeros((int(size[1]), int(size[0])), np.int32)
    for i, pixels in enumerate(groups):
        q = pixels - origin
        owner[q[:, 1], q[:, 0]] = i + 1
    edges = set()
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        if owner.shape[0] <= dy or owner.shape[1] <= abs(dx):
            continue
        if dx >= 0:
            a = owner[:owner.shape[0] - dy, :owner.shape[1] - dx]
            b = owner[dy:, dx:]
        else:
            a = owner[:-dy, 1:]
            b = owner[dy:, :-1]
        valid = (a > 0) & (b > 0) & (a != b)
        if valid.any():
            for u, v in np.unique(np.sort(np.column_stack([a[valid] - 1, b[valid] - 1]), axis=1), axis=0):
                edges.add((int(u), int(v)))
    active = dict(enumerate(groups))
    geometry = {i: group_geometry(p, cfg) for i, p in active.items()}
    merged = 0
    while edges:
        best = None
        for i, j in sorted(edges):
            x, y = geometry[i], geometry[j]
            delta = abs((x["direction"] - y["direction"] + 90) % 180 - 90)
            tiny = min(x["length"], y["length"]) < cfg.tiny_length
            if not tiny and delta > cfg.merge_angle:
                continue
            union = group_geometry(np.vstack([x["hull"], y["hull"]]), cfg)
            ratio = union["area"] / max(x["area"] + y["area"], 1)
            limit = 1.8 if tiny else cfg.merge_area_ratio
            if ratio > limit:
                continue
            score = (0 if tiny else 1, ratio, i, j)
            if best is None or score < best[0]:
                best = (score, i, j, union)
        if best is None:
            break
        _, i, j, union = best
        active[i] = np.vstack([active[i], active.pop(j)])
        geometry[i] = union
        del geometry[j]
        edges = {tuple(sorted((i if u == j else u, i if v == j else v))) for u, v in edges
                 if (i if u == j else u) != (i if v == j else v)}
        merged += 1
    return list(active.values()), merged


def fit_directional_box(pixels: np.ndarray, cfg: Settings) -> np.ndarray:
    """保留大段的真实倾斜方向；越界问题由输出图片加背景边解决，不改变框方向。"""
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(pixels.astype(np.float32) + 0.5)
    theta = math.radians(angle)
    extent = abs(math.cos(theta)) + abs(math.sin(theta))
    rw = max(rw + extent + 2 * cfg.padding, cfg.min_box_side)
    rh = max(rh + extent + 2 * cfg.padding, cfg.min_box_side)
    return cv2.boxPoints(((cx, cy), (rw, rh), angle)).astype(float)


def pad_for_boxes(image: np.ndarray, binary: np.ndarray, result: dict, old: list[np.ndarray]):
    """只在输出副本加 114 灰色边，平移全部坐标，保留倾斜框和原图内容。"""
    h, w = binary.shape
    if not result['boxes']:
        return image, binary, [0, 0, 0, 0]
    points = np.vstack(result['boxes'])
    lo, hi = points.min(axis=0), points.max(axis=0)
    left, top = int(max(0, math.ceil(-lo[0]))), int(max(0, math.ceil(-lo[1])))
    right, bottom = int(max(0, math.ceil(hi[0] - w))), int(max(0, math.ceil(hi[1] - h)))
    padding = [left, top, right, bottom]
    if any(padding):
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        binary = np.pad(binary, ((top, bottom), (left, right)))
        result['kept_mask'] = np.pad(result['kept_mask'], ((top, bottom), (left, right)))
        offset = np.array([left, top])
        result['boxes'] = [b + offset for b in result['boxes']]
        result['pieces'] = [p + offset for p in result['pieces']]
        old[:] = [b + offset for b in old]
    return image, binary, padding


def boxes_for_group(pixels: np.ndarray, cfg: Settings, shape: tuple[int, int], depth: int = 0):
    """分离贴图像边缘的窄带，避免一个越界角使整条长裂缝退回大水平框。"""
    box, outside = fit_box(pixels, shape, cfg.min_box_side)
    if outside and depth < 3 and len(pixels) >= 2 * cfg.min_component_pixels:
        (_, _), (rw, rh), _ = cv2.minAreaRect(pixels.astype(np.float32))
        band = max(4.0, min(24.0, 1.25 * min(rw + 1, rh + 1)))
        h, w = shape
        near = (pixels[:, 0] < band) | (pixels[:, 0] >= w - band) | (pixels[:, 1] < band) | (pixels[:, 1] >= h - band)
        if near.any() and (~near).any():
            groups = [pixels[~near]]
            edge_pixels = pixels[near]
            x0, y0 = edge_pixels.min(axis=0)
            edge_mask = np.zeros((int(edge_pixels[:, 1].max() - y0 + 1), int(edge_pixels[:, 0].max() - x0 + 1)), np.uint8)
            edge_mask[edge_pixels[:, 1] - y0, edge_pixels[:, 0] - x0] = 1
            count, ids = cv2.connectedComponents(edge_mask, connectivity=8)
            membership = ids[edge_pixels[:, 1] - y0, edge_pixels[:, 0] - x0]
            groups.extend(edge_pixels[membership == i] for i in range(1, count))
            fitted, fallbacks = [], 0
            for group in groups:
                sub, nf = boxes_for_group(group, cfg, shape, depth + 1)
                fitted.extend(sub)
                fallbacks += nf
            return fitted, fallbacks
    return [box], int(outside)


def fit_box(pixels_xy: np.ndarray, shape: tuple[int, int], min_side: float) -> tuple[np.ndarray, bool]:
    """像素中心拟合后扩展至像素边界；越界时退回本段的轴对齐框，避免裁角破坏矩形。"""
    h, w = shape
    centers = pixels_xy.astype(np.float32) + 0.5
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(centers)
    theta = math.radians(angle)
    pixel_extent = abs(math.cos(theta)) + abs(math.sin(theta))
    rw, rh = max(rw + pixel_extent, min_side), max(rh + pixel_extent, min_side)
    corners = cv2.boxPoints(((cx, cy), (rw, rh), angle)).astype(np.float64)
    outside = bool(
        (corners[:, 0] < -1e-4).any() or (corners[:, 0] > w + 1e-4).any()
        or (corners[:, 1] < -1e-4).any() or (corners[:, 1] > h + 1e-4).any()
    )
    if outside:
        x0, y0 = pixels_xy.min(axis=0).astype(float)
        x1, y1 = pixels_xy.max(axis=0).astype(float) + 1
        if x1 - x0 < min_side:
            x0 = max(0.0, min(x0 - (min_side - (x1 - x0)) / 2, w - min_side))
            x1 = min(float(w), x0 + min_side)
        if y1 - y0 < min_side:
            y0 = max(0.0, min(y0 - (min_side - (y1 - y0)) / 2, h - min_side))
            y1 = min(float(h), y0 + min_side)
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    else:
        corners = np.clip(corners, [0, 0], [w, h])
    return corners, outside


def convert_binary(binary: np.ndarray, cfg: Settings) -> dict:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    eligible = stats[:, cv2.CC_STAT_AREA] >= cfg.min_component_pixels
    eligible[0] = False
    kept = eligible[labels]
    traced = trace(
        kept, method="skeleton", bridge=False, fill_holes=0, min_area=1,
        prune=cfg.prune, prune_min=cfg.prune_min, min_length=0,
    )
    by_component = defaultdict(list)
    for crack in traced.cracks:
        pieces = relaxed_path_pieces(crack, cfg) if cfg.profile == "relaxed" else path_pieces(crack, cfg)
        for piece in pieces:
            xy = np.rint(piece).astype(int)
            xy[:, 0] = np.clip(xy[:, 0], 0, binary.shape[1] - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, binary.shape[0] - 1)
            ids = labels[xy[:, 1], xy[:, 0]]
            ids = ids[(ids > 0) & eligible[ids]]
            if len(ids):
                component = int(np.bincount(ids).argmax())
                by_component[component].append(piece)

    boxes, all_pieces = [], []
    boundary_fallbacks = no_path_components = merged_groups = boundary_adjustments = 0
    slices = ndi.find_objects(labels)
    for component in range(1, n):
        if not eligible[component]:
            continue
        sl = slices[component - 1]
        yy, xx = np.nonzero(labels[sl] == component)
        pixels = np.column_stack([xx + sl[1].start, yy + sl[0].start])
        pieces = by_component[component]
        if pieces:
            points = np.vstack(pieces)
            ids = np.repeat(np.arange(len(pieces)), [len(p) for p in pieces])
            assigned = ids[cKDTree(points).query(pixels, workers=1)[1]]
            counts = np.bincount(assigned, minlength=len(pieces))
            good = np.flatnonzero(counts >= cfg.min_component_pixels)
            if len(good):
                tiny = ~np.isin(assigned, good)
                if tiny.any():
                    q = np.vstack([pieces[i] for i in good])
                    qids = np.repeat(good, [len(pieces[i]) for i in good])
                    assigned[tiny] = qids[cKDTree(q).query(pixels[tiny], workers=1)[1]]
                groups = [pixels[assigned == i] for i in good]
            else:
                groups = [pixels]
            all_pieces.extend(pieces)
        else:
            no_path_components += 1
            groups = [pixels]
        if cfg.profile == "relaxed":
            groups, merged = merge_pixel_groups(groups, cfg)
            merged_groups += merged
            for group in groups:
                boxes.append(fit_directional_box(group, cfg))
        else:
            for group in groups:
                fitted, fallbacks = boxes_for_group(group, cfg, binary.shape)
                boundary_fallbacks += fallbacks
                boxes.extend(fitted)
    ignored = (stats[1:, cv2.CC_STAT_AREA] < cfg.min_component_pixels)
    return {
        "boxes": boxes, "pieces": all_pieces, "kept_mask": kept,
        "components": int(eligible.sum()), "ignored_components": int(ignored.sum()),
        "ignored_pixels": int(stats[1:, cv2.CC_STAT_AREA][ignored].sum()),
        "boundary_fallbacks": boundary_fallbacks, "no_path_components": no_path_components,
        "junctions": sum(len(c.junctions) for c in traced.cracks), "trace": traced,
        "merged_groups": merged_groups, "boundary_adjustments": boundary_adjustments,
    }


def raster_boxes(boxes: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(shape, np.uint8)
    for box in boxes:
        # 框坐标采用像素边界；栅格坐标采用像素中心。
        polygon = np.rint((box - 0.5) * 256).astype(np.int32)
        cv2.fillConvexPoly(canvas, polygon, 1, shift=8)
    return canvas.astype(bool)


def audit_boxes(boxes: list[np.ndarray], kept: np.ndarray) -> dict:
    covered = raster_boxes(boxes, kept.shape)
    hit = int((covered & kept).sum())
    return {
        "coverage": hit / max(1, int(kept.sum())) if kept.any() else 1.0,
        "box_union_pixels": int(covered.sum()),
        "background_pixels": int((covered & ~kept).sum()),
        "box_area_sum": sum(float(abs(cv2.contourArea(b.astype(np.float32)))) for b in boxes),
    }


def baseline_boxes(binary: np.ndarray, cfg: Settings) -> list[np.ndarray]:
    """从当前 mask 重算 V1 基础方法，只作对照，不读取旧数据集标签。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    h, w = binary.shape
    boxes = []
    for component in range(1, n):
        x, y, width, height, area = stats[component]
        if area < cfg.min_component_pixels:
            continue
        region = (labels[y:y + height, x:x + width] == component).astype(np.uint8)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        box = cv2.boxPoints(((cx, cy), (max(rw, cfg.min_box_side), max(rh, cfg.min_box_side)), angle))
        boxes.append(np.clip(box.astype(np.float64) + [x, y], [0, 0], [w - 1, h - 1]))
    return boxes


def save_labels(path: Path, boxes: list[np.ndarray], shape: tuple[int, int]) -> None:
    h, w = shape
    lines = ["0 " + " ".join(f"{x:.6f}" for x in (box / [w, h]).ravel()) for box in boxes]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def draw_comparison(image: np.ndarray, mask: np.ndarray, result: dict, old: list[np.ndarray], sample: str,
                    before_name="V1", after_name="V2") -> np.ndarray:
    trace_panel = image.copy()
    trace_panel[mask] = (0.35 * trace_panel[mask] + 0.65 * np.array([65, 225, 80])).astype(np.uint8)
    for i, points in enumerate(result["pieces"]):
        cv2.polylines(trace_panel, [np.rint(points).astype(np.int32)], False, COLORS[i % len(COLORS)], 2, cv2.LINE_AA)
    old_panel, new_panel = image.copy(), image.copy()
    for box in old:
        cv2.polylines(old_panel, [np.rint(box).astype(np.int32)], True, (0, 165, 255), 2, cv2.LINE_AA)
    for i, box in enumerate(result["boxes"]):
        cv2.polylines(new_panel, [np.rint(box).astype(np.int32)], True, COLORS[i % len(COLORS)], 2, cv2.LINE_AA)
        anchor = np.rint(box.mean(axis=0)).astype(int)
        cv2.putText(new_panel, str(i + 1), tuple(anchor), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(new_panel, str(i + 1), tuple(anchor), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    panels = []
    for panel, title in zip([trace_panel, old_panel, new_panel], ["MASK + CENTERLINES", f"{before_name}: {len(old)} BOXES", f"{after_name}: {len(result['boxes'])} BOXES"]):
        panel = cv2.copyMakeBorder(panel, 48, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 31, 35))
        cv2.putText(panel, title, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(panel, sample, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 185, 190), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)


def choose_rows(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if limit >= len(rows):
        return rows
    selected = {}
    # 加入上一轮验证图中的实例，方便直接比较。
    wanted = ["IMG_6637_35", "IMG_6637_36", "IMG_6637_18", "IMG_6637_33", "IMG_6637_25", "IMG_6637_34"]
    lookup = {r["sample_id"]: r for r in rows}
    for name in wanted:
        if name in lookup:
            selected[name] = lookup[name]
    for split in ["train", "val", "test"]:
        candidates = [r for r in rows if r["split"] == split]
        positives = [r for r in candidates if int(r["boxes"]) > 0]
        orders = [
            sorted(positives, key=lambda r: float(r["max_aspect"])),
            sorted(positives, key=lambda r: -float(r["max_aspect"])),
            sorted(positives, key=lambda r: -int(r["boxes"])),
            [r for r in candidates if int(r["boxes"]) == 0],
        ]
        for order in orders:
            for row in order[:2]:
                selected.setdefault(row["sample_id"], row)
    rng = random.Random(seed)
    for row in rng.sample(rows, len(rows)):
        selected.setdefault(row["sample_id"], row)
        if len(selected) >= limit:
            break
    return list(selected.values())[:limit]


def read_rows(manifest: Path) -> list[dict]:
    with manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"sample_id", "split", "source_group", "image_path", "mask_path", "boxes", "max_aspect"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Expected an input manifest containing image_path, mask_path, split and source_group")
    splits = defaultdict(set)
    destinations = set()
    for row in rows:
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {row['split']}")
        splits[row["source_group"]].add(row["split"])
        key = (row["split"], Path(row["image_path"]).stem)
        if key in destinations:
            raise ValueError(f"Duplicate output stem: {key}")
        destinations.add(key)
    if any(len(v) != 1 for v in splits.values()):
        raise ValueError("Source-photo groups overlap train/val/test")
    return rows


def resolve_sources(rows: list[dict], source_root: Path | None) -> list[dict]:
    """可将独立清单中的原图和 mask 路径重定位；创建输出目录前检查全部输入。"""
    resolved, missing = [], []
    if source_root is not None and not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    for original in rows:
        row = dict(original)
        if source_root is not None:
            for key, folder in [("image_path", "outputs_RGB"), ("mask_path", "outputs_Mask")]:
                old = Path(row[key])
                row[key] = str(source_root / old.parent.parent.name / folder / old.name)
        for key in ["image_path", "mask_path"]:
            if not Path(row[key]).is_file():
                missing.append(row[key])
        resolved.append(row)
    if missing:
        raise FileNotFoundError(f"{len(missing)} source files are missing; examples: {missing[:4]}")
    return resolved


def link_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def write_reviews(out: Path, records: list[dict], before_name="V1", after_name="V2") -> None:
    records = [r for r in records if r["preview"]]
    cards = []
    # 突出有分叉的例子，同时保留容易、无裂缝的样本。
    featured = sorted(records, key=lambda r: (-r["junctions"], -r["background_reduction"], r["sample_id"]))
    featured = featured[: min(6, len(featured))]
    for record in featured:
        view = imread(out / record["preview"])
        scale = 900 / view.shape[1]
        cards.append(cv2.resize(view, (900, round(view.shape[0] * scale)), interpolation=cv2.INTER_AREA))
    if cards:
        imwrite(out / "review" / "overview.jpg", np.vstack(cards))
    for r in records:
        cardsrc = html.escape(r["preview"])
        cardtitle = html.escape(r["sample_id"])
        r["_card"] = (
            f'<article><h2>{cardtitle} · {r["split"]}</h2>'
            f'<p>框数 {r["old_boxes"]} → {r["new_boxes"]}；'
            f'保留 mask 覆盖率 {r["new_coverage"]:.1%}；'
            f'框内背景像素 {r["old_background_pixels"]} → {r["new_background_pixels"]}</p>'
            f'<a href="{cardsrc}"><img src="{cardsrc}" loading="lazy"></a></article>'
        )
    page = (
        f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>裂缝 OBB {after_name} 标注对照</title>'
        '<style>body{max-width:1500px;margin:30px auto;padding:0 20px;background:#eef0f2;color:#202428;'
        'font-family:system-ui,"Microsoft YaHei",sans-serif}article{background:white;padding:18px;margin:24px 0;'
        'border-radius:8px}img{width:100%;height:auto}h2{font-size:18px}p{line-height:1.7}</style>'
        f'<h1>裂缝 OBB {after_name} 标注对照（{len(records)} 张抽样预览）</h1><p>左：原 mask 与拆段中心线；中：{before_name}；右：{after_name}。'
        '这是自动标注候选；请结合原 mask 检查大段方向、明显转弯和分支。'
        '贴边倾斜框通过输出图片补灰边保留方向。点击图片查看原尺寸。</p>'
        + ''.join(r.pop("_card") for r in records) + '</html>'
    )
    (out / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/crack512/manifest.csv")
    parser.add_argument("--source-root", type=Path, help="crack512 根目录，包含 labelme/V7 下的 outputs_RGB 和 outputs_Mask")
    parser.add_argument("--profile", choices=["v2", "relaxed"], default="relaxed", help="默认 relaxed 生成放宽版 V2.1；v2 可重现旧规则")
    parser.add_argument("--compare-root", type=Path, help="对照标签目录的上级目录，默认 relaxed 与现有 V2 对比")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review-count", type=int, default=36, help="完整转换时保存的对照图和 trace 数量；全部标签始终输出")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=36)
    selection.add_argument("--full", action="store_true")
    parser.add_argument("--samples", nargs="+", help="只转换指定 sample_id，顺序与参数一致")
    parser.add_argument("--max-deviation", type=float, help="仅旧 --profile v2 使用的曲线偏离阈值；V2.1 请用 --split-angle")
    parser.add_argument("--min-segment-length", type=float)
    parser.add_argument("--split-angle", type=float, default=30.0, help="大段方向变化达到该角度并持续一段距离才切分，默认 30 度")
    args = parser.parse_args()
    if args.limit < 1 or args.review_count < 1 or (args.max_deviation is not None and args.max_deviation <= 0) or (args.min_segment_length is not None and args.min_segment_length <= 0):
        parser.error("limit、max-deviation、min-segment-length 必须为正数")
    if args.profile == "relaxed" and args.max_deviation is not None:
        parser.error("V2.1 按转角切分，不使用 max-deviation；请设置 --split-angle")
    rows = read_rows(args.manifest)
    if args.samples:
        lookup = {r["sample_id"]: r for r in rows}
        missing = set(args.samples) - lookup.keys()
        if missing:
            parser.error(f"找不到样本: {sorted(missing)}")
        rows = [lookup[s] for s in dict.fromkeys(args.samples)]
    elif not args.full:
        rows = choose_rows(rows, args.limit, seed=20261110)
    rows = resolve_sources(rows, args.source_root)
    full = args.full and not args.samples
    relaxed = args.profile == "relaxed"
    version = "crack_obb_auto_v21" if relaxed else "crack_obb_auto_v2"
    before_name, after_name = ("V2", "V2.1") if relaxed else ("V1", "V2")
    compare_root = args.compare_root or (ROOT / "data/crack_obb_auto_v2" if relaxed else None)
    review_ids = {r["sample_id"] for r in choose_rows(rows, args.review_count, seed=20261110)} if full else {r["sample_id"] for r in rows}
    out = (args.output or ROOT / (f"data/{version}" if full else f"output/{version.replace('auto_', '')}_preview")).resolve()
    protected_v2 = (ROOT / "data/crack_obb_auto_v2").resolve()
    if relaxed and (out == protected_v2 or protected_v2 in out.parents):
        parser.error("放宽版必须输出到独立目录，不能覆盖正在训练的 V2")
    if out == args.manifest.resolve().parent or args.manifest.resolve().parent in out.parents:
        parser.error("输出目录必须位于输入数据目录之外")
    if out.exists() and any(out.iterdir()):
        parser.error(f"输出目录已非空，拒绝覆盖；请指定新的 --output: {out}")
    cfg = make_settings(args.profile, args.max_deviation, args.min_segment_length)
    if not 0 < args.split_angle < 180:
        parser.error("split-angle 必须位于 0 到 180 度之间")
    cfg.split_angle = args.split_angle
    cfg.merge_angle = max(0, args.split_angle - 5)
    out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    records = []
    new_long, new_short, old_long, old_short = [], [], [], []
    for index, row in enumerate(rows, 1):
        image_path, mask_path = Path(row["image_path"]), Path(row["mask_path"])
        image, color_mask = imread(image_path), imread(mask_path)
        if image.shape[:2] != color_mask.shape[:2]:
            raise ValueError(f"Image/mask size mismatch: {row['sample_id']}")
        binary = crack_mask(color_mask)
        result = convert_binary(binary, cfg)
        if compare_root is None:
            old = baseline_boxes(binary, Settings())
        else:
            old_path = compare_root / "labels" / row["split"] / f"{image_path.stem}.txt"
            old = []
            for line in old_path.read_text(encoding="utf-8").splitlines():
                z = np.fromstring(line, sep=" ")
                if len(z) != 9:
                    raise ValueError(f"Invalid comparison label: {old_path}")
                old.append(z[1:].reshape(4, 2) * [binary.shape[1], binary.shape[0]])
        original_shape = binary.shape
        padding = [0, 0, 0, 0]
        if relaxed:
            image, binary, padding = pad_for_boxes(image, binary, result, old)
        before, after = audit_boxes(old, result["kept_mask"]), audit_boxes(result["boxes"], result["kept_mask"])
        split, stem = row["split"], image_path.stem
        save_labels(out / "labels" / split / f"{stem}.txt", result["boxes"], binary.shape)
        if any(padding):
            imwrite(out / "images" / split / image_path.name, image)
        else:
            link_image(image_path, out / "images" / split / image_path.name)
        preview = ""
        if row["sample_id"] in review_ids:
            preview = f"review/{split}/{stem}.jpg"
            imwrite(out / preview, draw_comparison(image, binary, result, old, row["sample_id"], before_name, after_name))
            trace_dir = out / "traces" / split
            trace_dir.mkdir(parents=True, exist_ok=True)
            result["trace"].to_json(trace_dir / f"{stem}.json")
            trace_path = trace_dir / f"{stem}.json"
            trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
            trace_data["coordinate_space"] = "original_source_image"
            trace_data["training_image_padding_ltrb"] = padding
            trace_path.write_text(json.dumps(trace_data, ensure_ascii=False), encoding="utf-8")
        flags = []
        if after["coverage"] < 0.995:
            flags.append("coverage_below_99.5pct")
        if result["no_path_components"]:
            flags.append("component_without_centerline")
        if result["boundary_fallbacks"]:
            flags.append("boundary_axis_aligned_fallback")
        if len(result["boxes"]) > max(12, len(old) * 6):
            flags.append("many_segments_review")
        record = {
            "sample_id": row["sample_id"], "split": split, "source_group": row["source_group"],
            "image_path": str(image_path), "mask_path": str(mask_path),
            "old_boxes": len(old), "new_boxes": len(result["boxes"]),
            "crack_pixels": int(binary.sum()), "kept_pixels": int(result["kept_mask"].sum()),
            "ignored_pixels": result["ignored_pixels"], "junctions": result["junctions"],
            "old_coverage": before["coverage"], "new_coverage": after["coverage"],
            "old_background_pixels": before["background_pixels"], "new_background_pixels": after["background_pixels"],
            "background_reduction": 1 - after["background_pixels"] / before["background_pixels"] if before["background_pixels"] else 0.0,
            "old_box_area": before["box_area_sum"], "new_box_area": after["box_area_sum"],
            "boundary_fallbacks": result["boundary_fallbacks"], "no_path_components": result["no_path_components"],
            "flags": ";".join(flags), "preview": preview,
        }
        records.append(record)
        for boxes, lengths, widths in [(old, old_long, old_short), (result["boxes"], new_long, new_short)]:
            for box in boxes:
                a, b = np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[1])
                lengths.append(float(max(a, b) * 448 / max(binary.shape)))
                widths.append(float(min(a, b) * 448 / max(binary.shape)))
        record["merged_groups"] = result["merged_groups"]
        record["boundary_adjustments"] = result["boundary_adjustments"]
        record["padding_ltrb"] = json.dumps(padding)
        record["original_width"], record["original_height"] = original_shape[1], original_shape[0]
        record["output_width"], record["output_height"] = binary.shape[1], binary.shape[0]
        if not full or index % 50 == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] {row['sample_id']} boxes {len(old)} -> {len(result['boxes'])}, coverage={after['coverage']:.3%}", flush=True)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    total_kept = sum(r["kept_pixels"] for r in records)
    old_background = sum(r["old_background_pixels"] for r in records)
    summary = {
        "version": version, "comparison": before_name, "mode": "full" if full else "preview_subset",
        "samples": len(records), "splits": dict(Counter(r["split"] for r in records)),
        "old_boxes": sum(r["old_boxes"] for r in records), "new_boxes": sum(r["new_boxes"] for r in records),
        "kept_mask_pixels": total_kept, "ignored_small_component_pixels": sum(r["ignored_pixels"] for r in records),
        "new_coverage": sum(r["new_coverage"] * r["kept_pixels"] for r in records) / max(1, total_kept),
        "old_background_pixels": old_background,
        "new_background_pixels": sum(r["new_background_pixels"] for r in records),
        "background_reduction": 1 - sum(r["new_background_pixels"] for r in records) / max(1, old_background),
        "flag_counts": dict(Counter(flag for r in records for flag in r["flags"].split(";") if flag)),
        "settings": asdict(cfg), "upstream": {"repository": "https://github.com/Kronbii/fine-crack-tracing", "commit": PIN},
        "source_manifest": str(args.manifest.resolve()),
        "review_samples": len(review_ids),
        "padded_images": sum(json.loads(r["padding_ltrb"]) != [0, 0, 0, 0] for r in records),
        "max_padding_per_side_px": max(max(json.loads(r["padding_ltrb"])) for r in records),
        "geometry_at_448": {
            name: {"long_median_px": float(np.median(lengths)) if lengths else 0,
                   "long_under16_pct": float(np.mean(np.array(lengths) < 16) * 100) if lengths else 0,
                   "long_under32_pct": float(np.mean(np.array(lengths) < 32) * 100) if lengths else 0,
                   "short_median_px": float(np.median(widths)) if widths else 0,
                   "short_under4_pct": float(np.mean(np.array(widths) < 4) * 100) if widths else 0}
            for name, lengths, widths in [(before_name, old_long, old_short), (after_name, new_long, new_short)]
        },
        "note": "Geometric pseudo-labels for crack segments; lower box background does not prove improved model accuracy.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "data.yaml").write_text(
        f"# {'Full dataset' if full else 'PREVIEW SUBSET ONLY'}; automatic crack-segment OBB labels\n"
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: crack\n",
        encoding="utf-8",
    )
    with (out / "source_groups.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["source_group", "split"])
        writer.writerows(sorted({(r["source_group"], r["split"]) for r in rows}))
    write_reviews(out, records, before_name, after_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Review gallery: {out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
