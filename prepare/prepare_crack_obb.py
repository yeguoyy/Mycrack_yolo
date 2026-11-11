import csv
import filecmp
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SOURCE_MANIFEST = PROJECT_DIR / "dev" / "data" / "mnw_crack_det" / "manifest.csv"
DATASET_DIR = SCRIPT_DIR / "data" / "crack_obb_auto_v1"
REVIEW_DIR = DATASET_DIR / "review"
CRACK_BGR = np.array((0, 128, 0), dtype=np.uint8)
SPALLING_BGR = np.array((0, 0, 128), dtype=np.uint8)
MIN_COMPONENT_PIXELS = 12
MIN_BOX_SIDE = 2.0
PREVIEW_SEED = 20261017


def read_manifest() -> list[dict[str, str]]:
    with SOURCE_MANIFEST.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f"No samples found in {SOURCE_MANIFEST}")

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["source_group"]].add(row["split"])
    leaked = {group: splits for group, splits in group_splits.items() if len(splits) != 1}
    if leaked:
        raise ValueError(f"Source groups appear in multiple splits: {list(leaked.items())[:5]}")
    return rows


def obb_corners(component: np.ndarray) -> tuple[np.ndarray, float, float]:
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Connected component has no contour")
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    (cx, cy), (width, height), angle = rect
    width = max(float(width), MIN_BOX_SIDE)
    height = max(float(height), MIN_BOX_SIDE)
    corners = cv2.boxPoints(((cx, cy), (width, height), angle))
    return corners, width, height


def convert_mask(mask_path: Path, label_path: Path) -> dict:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    background = np.all(mask == 0, axis=2)
    crack = np.all(mask == CRACK_BGR, axis=2)
    spalling = np.all(mask == SPALLING_BGR, axis=2)
    unknown_pixels = int(mask.shape[0] * mask.shape[1] - np.count_nonzero(background | crack | spalling))
    if unknown_pixels:
        raise ValueError(f"{mask_path} contains {unknown_pixels} pixels with unknown colors")

    count, components, stats, _ = cv2.connectedComponentsWithStats(crack.astype(np.uint8), 8)
    height, width = crack.shape
    labels = []
    aspects = []
    ignored_components = 0
    ignored_pixels = 0
    for component_id in range(1, count):
        x, y, box_width, box_height, area = stats[component_id]
        if area < MIN_COMPONENT_PIXELS:
            ignored_components += 1
            ignored_pixels += int(area)
            continue

        component = (components[y : y + box_height, x : x + box_width] == component_id).astype(np.uint8)
        corners, side_a, side_b = obb_corners(component)
        aspect = max(side_a, side_b) / max(min(side_a, side_b), 1e-6)
        aspects.append(aspect)
        corners[:, 0] = np.clip(corners[:, 0] + x, 0, width - 1)
        corners[:, 1] = np.clip(corners[:, 1] + y, 0, height - 1)
        normalized = corners.astype(np.float64)
        normalized[:, 0] /= width
        normalized[:, 1] /= height
        labels.append("0 " + " ".join(f"{coordinate:.6f}" for coordinate in normalized.ravel()))

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    return {
        "image_size": [width, height],
        "crack_pixels": int(np.count_nonzero(crack)),
        "spalling_pixels": int(np.count_nonzero(spalling)),
        "components": int(count - 1),
        "ignored_small_components": ignored_components,
        "ignored_small_pixels": ignored_pixels,
        "boxes": len(labels),
        "max_aspect": max(aspects, default=0.0),
        "aspects": aspects,
        "corners": [np.fromstring(line.split(" ", 1)[1], sep=" ").reshape(4, 2).tolist() for line in labels],
    }


def link_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination) or filecmp.cmp(source, destination, shallow=False):
            return
        raise FileExistsError(f"Refusing to replace an existing non-linked image: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        # The destination is generated data; copying is the portable fallback when hard links are unavailable.
        import shutil

        shutil.copy2(source, destination)


def draw_preview(image_path: Path, row: dict[str, str], result: dict) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    for points in result["corners"]:
        polygon = np.rint(np.asarray(points) * np.array([image.shape[1], image.shape[0]])).astype(np.int32)
        cv2.polylines(image, [polygon], True, (0, 165, 255), 2, cv2.LINE_AA)
    cv2.putText(
        image,
        f"{row['sample_id']} | OBBs: {result['boxes']} | max AR: {result['max_aspect']:.1f}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"{row['sample_id']} | OBBs: {result['boxes']} | max AR: {result['max_aspect']:.1f}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return image


def make_contact_sheet(split: str, previews: list[np.ndarray], rows: list[dict]) -> None:
    tile_size = 320
    columns = 4
    tiles = []
    for image, row in zip(previews, rows):
        scale = min((tile_size - 16) / image.shape[1], (tile_size - 16) / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        tile = np.full((tile_size, tile_size, 3), 245, dtype=np.uint8)
        x = (tile_size - resized.shape[1]) // 2
        y = (tile_size - resized.shape[0]) // 2
        tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        tiles.append(tile)
    if not tiles:
        return
    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 245))
    sheet = np.vstack([np.hstack(tiles[index : index + columns]) for index in range(0, len(tiles), columns)])
    if not cv2.imwrite(str(REVIEW_DIR / f"{split}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not write preview for split {split}")


def select_preview_rows(rows: list[dict], results: dict[str, dict]) -> list[dict]:
    def box_count(row: dict) -> int:
        return results[row["sample_id"]]["boxes"]

    def max_aspect(row: dict) -> float:
        return results[row["sample_id"]]["max_aspect"]

    selected: dict[str, dict] = {}
    for row in sorted(rows, key=lambda item: (-box_count(item), item["sample_id"]))[:6]:
        selected[row["sample_id"]] = row
    for row in sorted(rows, key=lambda item: (-max_aspect(item), item["sample_id"]))[:5]:
        selected[row["sample_id"]] = row
    rng = random.Random(PREVIEW_SEED)
    for row in rng.sample(rows, min(5, len(rows))):
        selected[row["sample_id"]] = row
    return list(selected.values())


def write_outputs(rows: list[dict[str, str]], results: dict[str, dict], summaries: dict[str, dict]) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    group_split: dict[str, str] = {}
    for row in rows:
        group_split[row["source_group"]] = row["split"]
    with (DATASET_DIR / "source_groups.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["source_group", "split"])
        for group, split in sorted(group_split.items()):
            writer.writerow([group, split])

    summary = {
        "version": "crack_obb_auto_v1",
        "method": "one minimum-area rotated rectangle per 8-connected green-mask component",
        "class_id": 0,
        "class_name": "crack",
        "min_component_pixels": MIN_COMPONENT_PIXELS,
        "minimum_box_side_pixels": MIN_BOX_SIDE,
        "patches": len(rows),
        "source_groups": len(group_split),
        "split_groups": Counter(group_split.values()),
        "splits": summaries,
        "note": "Automatically derived OBB labels from CUBIT-Seg pixel masks; curved/branched crack boxes require visual review.",
    }
    (DATASET_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    data_yaml = (
        "# Automatically generated OBB labels; see README.md for conversion limits.\n"
        f"path: {DATASET_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: crack\n"
    )
    (DATASET_DIR / "data.yaml").write_text(data_yaml, encoding="utf-8")

    manifest_path = DATASET_DIR / "manifest.csv"
    fields = [
        "sample_id",
        "split",
        "source_group",
        "batch",
        "image_path",
        "mask_path",
        "crack_pixels",
        "spalling_pixels",
        "boxes",
        "ignored_small_components",
        "ignored_small_pixels",
        "max_aspect",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            result = results[row["sample_id"]]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "source_group": row["source_group"],
                    "batch": row["batch"],
                    "image_path": row["image"],
                    "mask_path": row["mask"],
                }
                | {
                    "crack_pixels": result["crack_pixels"],
                    "spalling_pixels": result["spalling_pixels"],
                    "boxes": result["boxes"],
                    "ignored_small_components": result["ignored_small_components"],
                    "ignored_small_pixels": result["ignored_small_pixels"],
                    "max_aspect": f"{result['max_aspect']:.4f}",
                }
            )

    (REVIEW_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    rows = read_manifest()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    summary_by_split: dict[str, dict] = {}
    preview_by_split: dict[str, list[tuple[dict, np.ndarray]]] = defaultdict(list)
    split_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        split_rows[row["split"]].append(row)

    total = len(rows)
    for index, row in enumerate(rows, start=1):
        source_image = Path(row["image"])
        source_mask = Path(row["mask"])
        image_name = source_image.name
        image_out = DATASET_DIR / "images" / row["split"] / image_name
        label_out = DATASET_DIR / "labels" / row["split"] / f"{source_image.stem}.txt"
        link_image(source_image, image_out)
        result = convert_mask(source_mask, label_out)
        results[row["sample_id"]] = result
        print(
            f"[{index}/{total}] {row['sample_id']} split={row['split']} "
            f"boxes={result['boxes']} ignored_small={result['ignored_small_components']}"
        )

    for split, candidates in split_rows.items():
        split_results = [results[row["sample_id"]] for row in candidates]
        box_counts = [result["boxes"] for result in split_results]
        aspects = [aspect for result in split_results for aspect in result["aspects"]]
        summary_by_split[split] = {
            "patches": len(candidates),
            "source_groups": len({row["source_group"] for row in candidates}),
            "images_with_boxes": sum(count > 0 for count in box_counts),
            "images_without_boxes": sum(count == 0 for count in box_counts),
            "boxes": sum(box_counts),
            "ignored_small_components": sum(result["ignored_small_components"] for result in split_results),
            "ignored_small_pixels": sum(result["ignored_small_pixels"] for result in split_results),
            "box_aspect_median": float(np.median(aspects)) if aspects else 0.0,
            "box_aspect_p90": float(np.percentile(aspects, 90)) if aspects else 0.0,
            "box_aspect_max": max(aspects, default=0.0),
        }

        selected = select_preview_rows(candidates, results)
        for row in selected:
            preview_path = DATASET_DIR / "images" / split / Path(row["image"]).name
            preview_by_split[split].append((row, draw_preview(preview_path, row, results[row["sample_id"]])))

    for split, entries in preview_by_split.items():
        make_contact_sheet(split, [image for _, image in entries], [row for row, _ in entries])
    write_outputs(rows, results, summary_by_split)

    print("\nOBB data ready:", DATASET_DIR.resolve())
    print("Ultralytics data config:", (DATASET_DIR / "data.yaml").resolve())
    print("Label review sheets:", REVIEW_DIR.resolve())
    for split, summary in summary_by_split.items():
        print(
            f"{split}: {summary['patches']} images, {summary['boxes']} boxes, "
            f"{summary['images_without_boxes']} no-crack images"
        )


if __name__ == "__main__":
    main()
