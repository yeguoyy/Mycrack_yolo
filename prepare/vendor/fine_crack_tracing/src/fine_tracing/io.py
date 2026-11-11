"""Reading masks and photos, and pairing them up."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".webp")


def read_mask(path: str) -> np.ndarray:
    p = str(path)
    if p.endswith(".npy"):
        return np.load(p)
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"could not read mask {p}")
    return m


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image {path}")
    return img


def _listing(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS or p.suffix == ".npy")


def pair_inputs(masks: str, frames: Optional[str] = None) -> List[Tuple[str, Path, Optional[Path]]]:
    """(name, mask path, photo path or None) for a mask file or folder.

    Photos are matched to masks by file name, whatever the extension.
    """
    mp = Path(masks)
    mask_files = [mp] if mp.is_file() else _listing(mp)
    frame_index = {}
    if frames:
        fp = Path(frames)
        for f in [fp] if fp.is_file() else _listing(fp):
            frame_index.setdefault(f.stem, f)
    out = []
    for m in mask_files:
        f = frame_index.get(m.stem)
        if f is None and frames and Path(frames).is_file():
            f = Path(frames)
        out.append((m.stem, m, f))
    return out


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


__all__ = ["IMAGE_EXTS", "read_mask", "read_image", "pair_inputs", "ensure_dir"]

