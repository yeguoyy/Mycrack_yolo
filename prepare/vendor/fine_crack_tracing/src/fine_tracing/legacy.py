"""The v0.1 tracer, kept so the benchmark can compare against it.

This follows the first release step for step: for each outer contour of the
mask, find ORB keypoints or Shi-Tomasi corners in the CLAHE-equalised photo
inside the contour's box, order them (by y, by the longest path through a
minimum spanning tree, or greedily) and join them with straight lines. Its
only change is resizing the mask along with the image when ORB upsizes a
small crop, which used to raise an OpenCV error.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree

# v0.1 defaults
MAXCORNERS = 1000000
MINQUALITY = 0.1
MINDISTANCE = 10
BLOCKSIZE = 3
CORNER_DIST = 20
ORB_FEATURES = 600
INTERP_DEG = 3
GREEDY_DIST = 10


def _shitomasi(img, mask, x0, y0, offset):
    raw = cv2.goodFeaturesToTrack(
        img,
        maxCorners=MAXCORNERS,
        qualityLevel=MINQUALITY - offset,
        minDistance=MINDISTANCE,
        mask=mask,
        blockSize=BLOCKSIZE,
    )
    if raw is None or (len(raw) - 1) < INTERP_DEG:
        return None
    return [[int(x) + x0, int(y) + y0] for x, y in raw.reshape(-1, 2)]


def _orb(img, mask, x0, y0):
    if img.shape[0] < 32 or img.shape[1] < 32:
        img = cv2.resize(img, (32, 32))
        mask = cv2.resize(mask, (32, 32), interpolation=cv2.INTER_NEAREST)
    kps, _ = cv2.ORB_create(nfeatures=ORB_FEATURES).detectAndCompute(img, mask)
    if not kps:
        return None
    raw = np.array([kp.pt for kp in kps])
    if (len(raw) - 1) < INTERP_DEG:
        return None
    return [[int(x + x0), int(y + y0)] for x, y in raw]


def _mst_order(pts):
    P = np.asarray(pts, float)
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    D[D == 0] = 1e-9
    np.fill_diagonal(D, 0)
    T = minimum_spanning_tree(D)
    T = T + T.T
    deg = np.asarray((T > 0).sum(axis=1)).ravel()
    leaves = np.nonzero(deg == 1)[0]
    if len(leaves) == 0:
        return P.astype(int).tolist()
    d0 = dijkstra(T, directed=False, indices=int(leaves[0]))
    far = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    d1, pred = dijkstra(T, directed=False, indices=far, return_predecessors=True)
    opp = int(np.argmax(np.where(np.isfinite(d1), d1, -1)))
    chain = [opp]
    while chain[-1] != far:
        chain.append(int(pred[chain[-1]]))
    return P[chain[::-1]].astype(int).tolist()


def _greedy_order(pts):
    P = np.asarray(pts, float)
    start = int(np.argmin(P[:, 1]))
    visited = {start}
    path = [start]
    cur = start
    while len(visited) < len(P):
        d = np.linalg.norm(P[cur] - P, axis=1)
        d[list(visited)] = np.inf
        nxt = int(np.argmin(d))
        visited.add(nxt)
        path.append(nxt)
        cur = nxt
    out = P[path].astype(int).tolist()
    if len(out) >= 2:
        last, blast = out[-1], out[-2]
        if np.linalg.norm(np.subtract(last, blast)) > GREEDY_DIST and blast[0] > last[0]:
            out = out[:-1]
    return out


def trace(
    mask: np.ndarray,
    frame: np.ndarray,
    detection: str = "orb",
    sort: str = "classic",
) -> List[np.ndarray]:
    """One polyline per mask contour, as (N, 2) x, y arrays (empty when
    nothing was found), the way v0.1 drew them."""
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out: List[np.ndarray] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        crop, cm = gray[y : y + h, x : x + w], m[y : y + h, x : x + w]
        corners: Optional[list] = None
        if detection == "orb":
            corners = _orb(crop, cm, x, y)
        else:
            idx = 0.0
            while idx <= 0.06:
                corners = _shitomasi(crop, cm, x, y, idx)
                if corners is None:
                    idx += 0.01
                    continue
                dist = float(np.linalg.norm(np.subtract(corners[0], corners[-1])))
                if dist and dist / max(len(corners), 1) > CORNER_DIST:
                    idx += 0.01
                    continue
                break
        if corners is None:
            out.append(np.zeros((0, 2)))
            continue
        if sort == "MST":
            ordered = _mst_order(corners)
        elif sort == "greedy":
            ordered = _greedy_order(corners)
        else:
            ordered = sorted(corners, key=lambda p: p[1])
        out.append(np.asarray(ordered, float))
    return out


__all__ = ["trace"]

