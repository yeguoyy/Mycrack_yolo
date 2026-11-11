"""The corner tracer: Shi-Tomasi corners pulled onto the centreline and
linked by a minimum spanning tree.

This is the toolkit's original approach, kept as the fallback for cracks
whose skeleton does not describe their mask. Two fixes over the first
version: corners found on the crack's edge are walked uphill on the distance
map until they sit on the centreline (so the trace follows the middle of a
wide crack, not its outline), and the spanning tree is kept whole rather than
cut down to its longest path, so branches survive.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree

from .graph import Graph, from_tree


def detect(
    comp: np.ndarray,
    gray: Optional[np.ndarray],
    quality: float,
    min_distance: int,
) -> np.ndarray:
    """Shi-Tomasi corners inside a crack component, as (x, y) floats."""
    m8 = comp.astype(np.uint8) * 255
    if gray is not None:
        src = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    else:
        src = cv2.GaussianBlur(m8, (3, 3), 0)
    support = cv2.dilate(m8, np.ones((3, 3), np.uint8))
    pts = cv2.goodFeaturesToTrack(
        src,
        maxCorners=0,
        qualityLevel=quality,
        minDistance=min_distance,
        mask=support,
        blockSize=3,
    )
    if pts is None:
        return np.zeros((0, 2))
    return pts.reshape(-1, 2).astype(float)


def snap_to_ridge(pts: np.ndarray, dt: np.ndarray, max_steps: int) -> np.ndarray:
    """Walk each point uphill on the distance map to the centreline."""
    H, W = dt.shape
    p = np.round(pts).astype(int)
    p[:, 0] = np.clip(p[:, 0], 0, W - 1)
    p[:, 1] = np.clip(p[:, 1], 0, H - 1)
    pad = np.pad(dt, 1, constant_values=-1)
    offs = np.array([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    for _ in range(max_steps):
        vals = np.stack([pad[p[:, 1] + 1 + dy, p[:, 0] + 1 + dx] for dx, dy in offs], axis=1)
        best = np.argmax(vals, axis=1)
        here = pad[p[:, 1] + 1, p[:, 0] + 1]
        move = vals[np.arange(len(p)), best] > here + 1e-6
        if not move.any():
            break
        p[move] += offs[best[move]]
    return p.astype(float)


def _thin_out(pts: np.ndarray, min_d: float) -> np.ndarray:
    if len(pts) < 2:
        return pts
    tree = cKDTree(pts)
    keep = np.ones(len(pts), bool)
    for i, j in sorted(tree.query_pairs(min_d)):
        if keep[i] and keep[j]:
            keep[j] = False
    return pts[keep]


def spanning_tree(pts: np.ndarray) -> np.ndarray:
    """Edges (i, j) of the Euclidean minimum spanning tree over the points."""
    n = len(pts)
    if n < 2:
        return np.zeros((0, 2), int)
    k = min(n - 1, 12)
    tree = cKDTree(pts)
    d, j = tree.query(pts, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = j[:, 1:].ravel()
    w = np.maximum(d[:, 1:].ravel(), 1e-6)
    G = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    G = G.maximum(G.T)
    ncomp, _ = connected_components(G, directed=False)
    if ncomp > 1 and n <= 4000:
        D = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        D[D == 0] = 1e-6
        np.fill_diagonal(D, 0)
        G = coo_matrix(D).tocsr()
    T = minimum_spanning_tree(G).tocoo()
    return np.column_stack([T.row, T.col])


def corner_graph(
    comp: np.ndarray,
    dt: np.ndarray,
    gray: Optional[np.ndarray],
    quality: float = 0.02,
    min_distance: int = 5,
) -> Graph:
    pts = detect(comp, gray, quality, min_distance)
    # a straight crack edge has no corners, so seed the ridge as well
    ridge = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6)
    ry, rx = np.nonzero(ridge)
    seeds = np.column_stack([rx, ry]).astype(float)
    pts = np.vstack([pts, seeds]) if len(pts) else seeds
    if len(pts) == 0:
        return Graph()
    steps = int(np.ceil(dt.max())) + 2
    pts = snap_to_ridge(pts, dt, steps)
    pts = np.unique(pts, axis=0)
    pts = _thin_out(pts, max(1.0, min_distance / 2))
    if len(pts) == 1:
        g = Graph()
        g.add_node(pts[0])
        return g
    pairs = spanning_tree(pts)
    return from_tree(pts, pairs, "corners")


__all__ = ["detect", "snap_to_ridge", "spanning_tree", "corner_graph"]

