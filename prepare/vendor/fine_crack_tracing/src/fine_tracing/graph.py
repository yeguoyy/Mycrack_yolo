"""A crack as a graph: free ends and junctions joined by pixel paths.

Coordinates are ``(x, y)`` floats in image pixels throughout. Every edge keeps
its full point list from one node to the other, plus a per-segment flag that
marks the stretches added to bridge a gap in the mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

Radius = Callable[[np.ndarray], float]


@dataclass
class Edge:
    u: int
    v: int
    pts: np.ndarray  # (N, 2) x, y from node u to node v
    bridged: np.ndarray  # (N-1,) bool, one per segment
    methods: FrozenSet[str] = field(default_factory=frozenset)

    @property
    def length(self) -> float:
        if len(self.pts) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.pts, axis=0), axis=1).sum())

    def oriented(self, start: int) -> Tuple[np.ndarray, np.ndarray]:
        """Points and flags read from ``start``."""
        if start == self.u:
            return self.pts, self.bridged
        return self.pts[::-1], self.bridged[::-1]


class Graph:
    def __init__(self) -> None:
        self.nodes: Dict[int, np.ndarray] = {}
        self.edges: Dict[int, Edge] = {}
        self.adj: Dict[int, Set[int]] = {}
        self._nid = 0
        self._eid = 0

    # ---- building -----------------------------------------------------
    def add_node(self, xy: Iterable[float]) -> int:
        nid = self._nid
        self._nid += 1
        self.nodes[nid] = np.asarray(xy, dtype=float).copy()
        self.adj[nid] = set()
        return nid

    def add_edge(
        self,
        u: int,
        v: int,
        pts: np.ndarray,
        bridged: Optional[np.ndarray] = None,
        methods: Iterable[str] = (),
    ) -> int:
        pts = np.asarray(pts, dtype=float)
        if bridged is None:
            bridged = np.zeros(max(len(pts) - 1, 0), bool)
        eid = self._eid
        self._eid += 1
        self.edges[eid] = Edge(u, v, pts, np.asarray(bridged, bool), frozenset(methods))
        self.adj[u].add(eid)
        self.adj[v].add(eid)
        return eid

    def remove_edge(self, eid: int) -> Edge:
        e = self.edges.pop(eid)
        self.adj[e.u].discard(eid)
        self.adj[e.v].discard(eid)
        return e

    def remove_node(self, nid: int) -> None:
        for eid in list(self.adj[nid]):
            self.remove_edge(eid)
        del self.adj[nid]
        del self.nodes[nid]

    def absorb(self, other: "Graph", offset: Tuple[float, float] = (0.0, 0.0)) -> None:
        """Copy another graph in, shifted by ``offset``."""
        off = np.asarray(offset, float)
        remap = {nid: self.add_node(xy + off) for nid, xy in other.nodes.items()}
        for e in other.edges.values():
            self.add_edge(remap[e.u], remap[e.v], e.pts + off, e.bridged.copy(), e.methods)

    # ---- queries ------------------------------------------------------
    def degree(self, nid: int) -> int:
        return sum(2 if self.edges[e].u == self.edges[e].v else 1 for e in self.adj[nid])

    def other(self, eid: int, nid: int) -> int:
        e = self.edges[eid]
        return e.v if e.u == nid else e.u

    def total_length(self) -> float:
        return float(sum(e.length for e in self.edges.values()))

    def components(self) -> List[Tuple[Set[int], Set[int]]]:
        seen: Set[int] = set()
        out = []
        for start in self.nodes:
            if start in seen:
                continue
            nodes, edges, stack = {start}, set(), [start]
            seen.add(start)
            while stack:
                n = stack.pop()
                for eid in self.adj[n]:
                    edges.add(eid)
                    m = self.other(eid, n)
                    if m not in seen:
                        seen.add(m)
                        nodes.add(m)
                        stack.append(m)
            out.append((nodes, edges))
        return out

    def end_tangent(self, nid: int, reach: float, skip: float = 0.0) -> Optional[np.ndarray]:
        """Unit direction pointing out of a free end.

        Fitted to the stretch from ``skip`` to ``skip + reach`` px in from
        the end: at a square-cut end the last half-width of skeleton bends off
        towards a corner, so it is left out when ``skip`` is set.
        """
        if self.degree(nid) != 1:
            return None
        (eid,) = self.adj[nid]
        pts, _ = self.edges[eid].oriented(nid)
        if len(pts) < 2:
            return None
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < skip + 4.0:
            skip = max(0.0, s[-1] - reach) if s[-1] > reach else 0.0
        a = int(np.searchsorted(s, skip))
        b = max(a + 2, int(np.searchsorted(s, skip + reach)) + 1)
        tail = pts[a:b]
        if len(tail) < 2:
            tail = pts[: max(2, b)]
        c = tail - tail.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        d = vt[0]
        # point it outwards
        if np.dot(d, tail[0] - tail[-1]) < 0:
            d = -d
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else None

    # ---- simplification ----------------------------------------------
    def merge_degree2(self) -> None:
        """Fuse every pass-through node into one longer edge."""
        changed = True
        while changed:
            changed = False
            for nid in list(self.nodes):
                if nid not in self.nodes or len(self.adj[nid]) != 2:
                    continue
                e1, e2 = list(self.adj[nid])
                a, b = self.edges[e1], self.edges[e2]
                if a.u == a.v or b.u == b.v:
                    continue
                p1, f1 = a.oriented(self.other(e1, nid))  # ends at nid
                p2, f2 = b.oriented(nid)  # starts at nid
                u, v = self.other(e1, nid), self.other(e2, nid)
                pts = np.vstack([p1, p2[1:]])
                flags = np.concatenate([f1, f2])
                self.remove_edge(e1)
                self.remove_edge(e2)
                del self.adj[nid]
                del self.nodes[nid]
                self.add_edge(u, v, pts, flags, a.methods | b.methods)
                changed = True


# ---------------------------------------------------------------------------
# skeleton -> graph
# ---------------------------------------------------------------------------


def from_skeleton(skel: np.ndarray, method: str = "skeleton") -> Graph:
    """Build a graph from a one-pixel-wide skeleton.

    Diagonal steps that duplicate a two-step path through a 4-neighbour are
    dropped first, so staircase corners do not read as junctions. Clusters of
    junction pixels become a single node at their centroid.
    """
    g = Graph()
    S = np.pad(np.asarray(skel, bool), 1)
    rr, cc = np.nonzero(S)
    n = len(rr)
    if n == 0:
        return g
    idx = -np.ones(S.shape, np.int64)
    idx[rr, cc] = np.arange(n)

    ii, jj = [], []
    for dr, dc in ((0, 1), (1, 0)):
        m = S[rr + dr, cc + dc]
        ii.append(np.nonzero(m)[0])
        jj.append(idx[rr[m] + dr, cc[m] + dc])
    for dr, dc in ((1, 1), (1, -1)):
        m = S[rr + dr, cc + dc] & ~S[rr, cc + dc] & ~S[rr + dr, cc]
        ii.append(np.nonzero(m)[0])
        jj.append(idx[rr[m] + dr, cc[m] + dc])
    i = np.concatenate(ii)
    j = np.concatenate(jj)
    A = sparse.coo_matrix((np.ones(len(i), np.int8), (i, j)), shape=(n, n)).tocsr()
    A = (A + A.T).tocsr()
    indptr, indices = A.indptr, A.indices
    deg = np.diff(indptr)
    xy = np.stack([cc - 1, rr - 1], axis=1).astype(float)

    node_of = -np.ones(n, np.int64)
    junction = np.nonzero(deg >= 3)[0]
    if len(junction):
        sub = A[junction][:, junction]
        ncl, lab = csgraph.connected_components(sub, directed=False)
        for c in range(ncl):
            members = junction[lab == c]
            nid = g.add_node(xy[members].mean(axis=0))
            node_of[members] = nid
    for p in np.nonzero(deg <= 1)[0]:
        node_of[p] = g.add_node(xy[p])

    visited = np.zeros(n, bool)
    seen_pairs: Set[Tuple[int, int]] = set()
    meth = (method,)

    def walk(p: int, q: int) -> Tuple[List[int], int]:
        path = [p, q]
        prev, cur = p, q
        while node_of[cur] < 0:
            visited[cur] = True
            nb = indices[indptr[cur] : indptr[cur + 1]]
            nxt = nb[0] if nb[0] != prev else nb[1]
            path.append(nxt)
            prev, cur = cur, nxt
            if cur == p:  # came back round a loop through a node pixel
                break
        return path, cur

    for p in np.nonzero(node_of >= 0)[0]:
        u = node_of[p]
        for q in indices[indptr[p] : indptr[p + 1]]:
            if node_of[q] >= 0:
                if node_of[q] == u:
                    continue
                key = (min(p, q), max(p, q))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                v = node_of[q]
                g.add_edge(u, v, _with_nodes(g, u, v, xy[[p, q]]), methods=meth)
                continue
            if visited[q]:
                continue
            path, end = walk(p, q)
            v = node_of[end]
            g.add_edge(u, v, _with_nodes(g, u, v, xy[path]), methods=meth)

    # closed loops with no node on them
    for p in np.nonzero((deg == 2) & ~visited & (node_of < 0))[0]:
        if visited[p]:
            continue
        nid = g.add_node(xy[p])
        node_of[p] = nid
        visited[p] = True
        nb = indices[indptr[p] : indptr[p + 1]]
        path, _ = walk(p, nb[0])
        g.add_edge(nid, nid, xy[path], methods=meth)

    # tiny loops that thinning leaves inside a junction cluster
    for eid in [k for k, e in g.edges.items() if e.u == e.v and e.length < 10]:
        nid = g.edges[eid].u
        g.remove_edge(eid)
        if not g.adj[nid]:
            g.remove_node(nid)
    g.merge_degree2()
    return g


def _with_nodes(g: Graph, u: int, v: int, pts: np.ndarray) -> np.ndarray:
    """Start and finish a pixel path exactly on its node positions."""
    parts = []
    if np.linalg.norm(g.nodes[u] - pts[0]) > 1e-9:
        parts.append(g.nodes[u][None])
    parts.append(pts)
    if np.linalg.norm(g.nodes[v] - pts[-1]) > 1e-9:
        parts.append(g.nodes[v][None])
    return np.vstack(parts)


def from_tree(points: np.ndarray, pairs: np.ndarray, method: str) -> Graph:
    """Graph from points and the edges of a spanning tree over them."""
    g = Graph()
    ids = [g.add_node(p) for p in points]
    for a, b in pairs:
        g.add_edge(ids[a], ids[b], np.vstack([points[a], points[b]]), methods=(method,))
    g.merge_degree2()
    return g


# ---------------------------------------------------------------------------
# pruning
# ---------------------------------------------------------------------------


def prune(g: Graph, radius: Radius, ratio: float, min_px: float) -> int:
    """Cut short side branches (spurs) from boundary bumps. Returns the count.

    A spur is an edge from a free end to a junction. It goes when it is
    shorter than ``ratio`` half-widths of the crack at the junction (a bump
    on a crack of half-width r leaves a spur of about r), or than
    ``min_px``. The shortest spurs go first and each junction is re-checked,
    so the two prongs at a square crack end lose only the shorter one.
    """
    removed = 0
    if ratio <= 0 and min_px <= 0:
        return 0
    while True:
        spurs = []
        for eid, e in g.edges.items():
            if e.u == e.v or e.bridged.any():
                continue
            du, dv = g.degree(e.u), g.degree(e.v)
            if du == 1 and dv >= 3:
                tip, j = e.u, e.v
            elif dv == 1 and du >= 3:
                tip, j = e.v, e.u
            else:
                continue
            limit = max(min_px, ratio * radius(g.nodes[j]))
            if e.length < limit:
                spurs.append((e.length, eid, tip, j))
        if not spurs:
            break
        spurs.sort()
        cut = 0
        for _, eid, tip, j in spurs:
            if eid not in g.edges or g.degree(j) < 3:
                continue
            g.remove_edge(eid)
            g.remove_node(tip)
            cut += 1
        g.merge_degree2()
        removed += cut
        if cut == 0:
            break
    return removed


# ---------------------------------------------------------------------------
# bridging gaps
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, a: int) -> int:
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    nb = np.linalg.norm(b)
    if nb < 1e-9:
        return 0.0
    c = float(np.clip(np.dot(a, b / nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def bridge(
    g: Graph,
    radius: Radius,
    max_gap: float,
    max_angle: float,
    min_piece: float = 15.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Join crack pieces across short breaks in the mask.

    Two free ends are joined when they are within ``max_gap`` and both point
    at each other to within ``max_angle``; the cheapest joins go first and no
    join may close a loop. A free end that points into the side of another
    crack within ``0.75 * max_gap`` is then joined to it, making a junction.
    Every join must touch at least one piece of ``min_piece`` px or more, so
    specks are never strung together into a crack of their own.
    Returns the bridges as (start, end) pairs.
    """
    made: List[Tuple[np.ndarray, np.ndarray]] = []
    if max_gap == 0 or len(g.nodes) < 2:
        return made
    uf = _UnionFind()
    for eid, e in g.edges.items():
        uf.union(e.u, e.v)
    ends = [n for n in g.nodes if g.degree(n) <= 1]
    if not ends:
        return made
    piece: Dict[int, float] = {}
    for e in g.edges.values():
        root = uf.find(e.u)
        piece[root] = piece.get(root, 0.0) + e.length

    def big_enough(a: int, b: int) -> bool:
        return max(piece.get(uf.find(a), 0.0), piece.get(uf.find(b), 0.0)) >= min_piece

    def join(a: int, b: int) -> bool:
        ra, rb = uf.find(a), uf.find(b)
        if not uf.union(a, b):
            return False
        piece[uf.find(a)] = piece.pop(ra, 0.0) + piece.pop(rb, 0.0) if ra != rb else piece.get(ra, 0.0)
        return True

    rad = {n: end_radius(g, n, radius) for n in ends}

    def reach(n: int) -> float:
        return float(np.clip(4.0 * rad[n], 8.0, 40.0))

    # two readings of each end's direction: right at the tip (a tapered end
    # that hooks) and from a little way in (a cut end whose last half-width
    # bends off to a corner); a join may use whichever fits
    tang = {n: g.end_tangent(n, reach(n), skip=1.5 * rad[n]) for n in ends}
    tang_tip = {n: g.end_tangent(n, float(np.clip(2.0 * rad[n], 6.0, 25.0))) for n in ends}

    def turn(n: int, v: np.ndarray) -> float:
        angs = [_angle(t, v) for t in (tang[n], tang_tip[n]) if t is not None]
        return min(angs) if angs else 0.0

    pos = np.array([g.nodes[n] for n in ends])
    tree = cKDTree(pos)
    r_max = max(rad.values())

    def allowed(na: int, nb: int) -> float:
        # auto: wide cracks may jump further than hairlines
        return max_gap if max_gap >= 0 else max(30.0, 6.0 * (rad[na] + rad[nb]))

    reach_all = (max_gap if max_gap >= 0 else max(30.0, 12.0 * r_max)) + 2 * r_max
    cands = []
    for a, b in tree.query_pairs(reach_all):
        na, nb = ends[a], ends[b]
        if uf.find(na) == uf.find(nb):
            continue
        v = pos[b] - pos[a]
        # a skeleton stops about a half-width short of each cut end
        d = float(np.linalg.norm(v)) - rad[na] - rad[nb]
        lim = allowed(na, nb)
        if d > lim:
            continue
        d = max(d, 0.0)
        ta, tb = tang[na], tang[nb]
        if ta is None and tb is None:
            if d > lim / 2:
                continue
            ang_a = ang_b = 0.0
        else:
            if (ta is None or tb is None) and d > lim / 2:
                continue
            ang_a = turn(na, v)
            ang_b = turn(nb, -v)
        if ang_a > max_angle or ang_b > max_angle or ang_a + ang_b > 1.5 * max_angle:
            continue
        cost = d * (1.0 + (ang_a + ang_b) / (2.0 * max(max_angle, 1e-6)))
        cands.append((cost, na, nb))
    cands.sort()
    used: Set[int] = set()
    for _, na, nb in cands:
        if na in used or nb in used or not big_enough(na, nb) or not join(na, nb):
            continue
        used.update((na, nb))
        _add_bridge(g, na, nb)
        made.append((g.nodes[na].copy(), g.nodes[nb].copy()))

    # free end -> side of another crack
    free = [n for n in ends if n not in used and n in g.nodes and g.degree(n) == 1]

    def side_gap(n: int) -> float:
        return 0.75 * (max_gap if max_gap >= 0 else max(30.0, 8.0 * rad[n]))

    t_cands = []
    for n in free:
        hit = _side_target(g, n, tang[n], uf, side_gap(n), max_angle * 0.75)
        if hit is not None:
            t_cands.append((hit[0], n))
    t_cands.sort()
    for _, n in t_cands:
        hit = _side_target(g, n, tang[n], uf, side_gap(n), max_angle * 0.75)
        if hit is None:
            continue
        _, eid, k = hit
        e = g.edges[eid]
        root = uf.find(e.u)
        # landing next to a free end: join the end rather than leave a stub
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(e.pts, axis=0), axis=1))])
        near_end = max(10.0, 2.5 * radius(e.pts[k]))
        if g.degree(e.u) == 1 and s[k] <= near_end:
            target = e.u
        elif g.degree(e.v) == 1 and s[-1] - s[k] <= near_end:
            target = e.v
        else:
            target = _split_edge(g, eid, k)
            uf.union(target, root)
        if not big_enough(n, target) or not join(n, target):
            continue
        _add_bridge(g, n, target)
        made.append((g.nodes[n].copy(), g.nodes[target].copy()))
    return made


def end_radius(g: Graph, n: int, radius: Radius, reach: float = 25.0) -> float:
    """Half-width of the crack just inside a free end (the very tip of a
    skeleton often sits on a thin horn of the mask)."""
    r0 = radius(g.nodes[n])
    if g.degree(n) != 1:
        return r0
    (eid,) = g.adj[n]
    pts, _ = g.edges[eid].oriented(n)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    k = max(1, int(np.searchsorted(s, reach)))
    return float(max([r0] + [radius(p) for p in pts[:k:3]]))


def _side_target(g, n, t, uf, max_d, max_angle):
    if t is None:
        return None
    p = g.nodes[n]
    root = uf.find(n)
    best = None
    for eid, e in g.edges.items():
        if uf.find(e.u) == root or len(e.pts) < 3:
            continue
        lo, hi = e.pts.min(axis=0) - max_d, e.pts.max(axis=0) + max_d
        if np.any(p < lo) or np.any(p > hi):
            continue
        inner = e.pts[1:-1]
        v = inner - p
        d = np.linalg.norm(v, axis=1)
        ok = d <= max_d
        if not ok.any():
            continue
        cosang = (v @ t) / np.maximum(d, 1e-9)
        ok &= cosang >= np.cos(np.radians(max_angle))
        if not ok.any():
            continue
        k = int(np.argmin(np.where(ok, d, np.inf)))
        if best is None or d[k] < best[0]:
            best = (float(d[k]), eid, k + 1)
    return best


def _split_edge(g: Graph, eid: int, k: int) -> int:
    e = g.remove_edge(eid)
    nid = g.add_node(e.pts[k])
    g.add_edge(e.u, nid, e.pts[: k + 1], e.bridged[:k], e.methods)
    g.add_edge(nid, e.v, e.pts[k:], e.bridged[k:], e.methods)
    return nid


def _add_bridge(g: Graph, a: int, b: int) -> None:
    pa, pb = g.nodes[a], g.nodes[b]
    d = float(np.linalg.norm(pb - pa))
    k = max(2, int(np.ceil(d)) + 1)
    pts = np.linspace(pa, pb, k)
    g.add_edge(a, b, pts, np.ones(k - 1, bool), ("bridge",))


# ---------------------------------------------------------------------------
# free ends
# ---------------------------------------------------------------------------


def extend_ends(g: Graph, mask: np.ndarray, radius: Radius) -> None:
    """Carry each free end along its direction to where the mask stops.

    A skeleton stops one half-width short of a blunt crack end (and of the
    image border), which shortens every crack by about one crack width.
    """
    H, W = mask.shape
    for n in list(g.nodes):
        if g.degree(n) != 1:
            continue
        r = end_radius(g, n, radius)
        t = g.end_tangent(n, float(np.clip(3.0 * r, 6.0, 30.0)), skip=r)
        if t is None:
            continue
        p = g.nodes[n]
        limit = 1.5 * r + 3.0
        s = 0.0
        last = p
        while s < limit:
            q = p + (s + 0.5) * t
            x, y = int(round(q[0])), int(round(q[1]))
            if x < 0 or y < 0 or x >= W or y >= H:
                last = np.clip(q, [-0.5, -0.5], [W - 0.5, H - 0.5])
                break
            if not mask[y, x]:
                break
            s += 0.5
            last = q
        if np.linalg.norm(last - p) < 0.75:
            continue
        (eid,) = g.adj[n]
        e = g.edges[eid]
        k = max(2, int(np.ceil(np.linalg.norm(last - p))) + 1)
        ext = np.linspace(p, last, k)[1:]
        if e.u == n:
            e.pts = np.vstack([ext[::-1], e.pts])
            e.bridged = np.concatenate([np.zeros(len(ext), bool), e.bridged])
        else:
            e.pts = np.vstack([e.pts, ext])
            e.bridged = np.concatenate([e.bridged, np.zeros(len(ext), bool)])
        g.nodes[n] = last.copy()


# ---------------------------------------------------------------------------
# splitting a network into a trunk and branches
# ---------------------------------------------------------------------------


@dataclass
class PathSpec:
    steps: List[Tuple[int, int]]  # (edge id, node it is read from)
    parent: Optional[int]
    order: int


def decompose(g: Graph, nodes: Set[int], edges: Set[int]) -> List[PathSpec]:
    """Split one connected network into paths: the longest run end to end
    (the trunk), then the longest remaining run off what is already covered,
    and so on until every edge belongs to exactly one path."""
    nodes_l = sorted(nodes)
    loc = {n: i for i, n in enumerate(nodes_l)}
    unused = set(edges)
    paths: List[PathSpec] = []
    node_path: Dict[int, int] = {}

    def best_edges(es: Set[int]):
        best: Dict[Tuple[int, int], Tuple[float, int]] = {}
        for eid in es:
            e = g.edges[eid]
            if e.u == e.v:
                continue
            a, b = loc[e.u], loc[e.v]
            key = (min(a, b), max(a, b))
            w = max(e.length, 1e-6)
            if key not in best or w < best[key][0]:
                best[key] = (w, eid)
        return best

    def matrix(best):
        if not best:
            return sparse.csr_matrix((len(nodes_l), len(nodes_l)))
        keys = np.array(list(best.keys()))
        w = np.array([v[0] for v in best.values()])
        M = sparse.coo_matrix((w, (keys[:, 0], keys[:, 1])), shape=(len(nodes_l),) * 2)
        return M.tocsr()

    def steps_from(pred_row, src, dst, best) -> List[Tuple[int, int]]:
        chain = [dst]
        while chain[-1] != src:
            chain.append(pred_row[chain[-1]])
        chain.reverse()
        out = []
        for a, b in zip(chain[:-1], chain[1:]):
            eid = best[(min(a, b), max(a, b))][1]
            out.append((eid, nodes_l[a]))
        return out

    def record(spec: PathSpec) -> None:
        pid = len(paths)
        paths.append(spec)
        for eid, start in spec.steps:
            unused.discard(eid)
            for n in (g.edges[eid].u, g.edges[eid].v):
                node_path.setdefault(n, pid)

    # trunk
    best = best_edges(unused)
    ends = [loc[n] for n in nodes_l if g.degree(n) == 1]
    if len(ends) >= 2 and best:
        M = matrix(best)
        dist, pred = csgraph.dijkstra(M, directed=False, indices=ends, return_predecessors=True)
        sub = dist[:, ends]
        sub[~np.isfinite(sub)] = -1
        a, b = np.unravel_index(int(np.argmax(sub)), sub.shape)
        if sub[a, b] > 0:
            record(PathSpec(steps_from(pred[a], ends[a], ends[b], best), None, 1))
    if not paths:
        eid = max(unused, key=lambda k: g.edges[k].length)
        record(PathSpec([(eid, g.edges[eid].u)], None, 1))

    # branches
    while unused:
        best = best_edges(unused)
        covered = sorted({loc[n] for n in node_path})
        found = False
        if best:
            M = matrix(best)
            dist, pred, src = csgraph.dijkstra(
                M, directed=False, indices=covered, return_predecessors=True, min_only=True
            )
            dist[~np.isfinite(dist)] = -1
            f = int(np.argmax(dist))
            if dist[f] > 0:
                s = int(src[f])
                steps = steps_from(pred, s, f, best)
                parent = node_path.get(nodes_l[s])
                order = paths[parent].order + 1 if parent is not None else 1
                record(PathSpec(steps, parent, order))
                found = True
        if not found:
            # what is left closes loops between covered nodes
            for eid in sorted(unused, key=lambda k: -g.edges[k].length):
                e = g.edges[eid]
                parent = node_path.get(e.u, node_path.get(e.v))
                order = paths[parent].order + 1 if parent is not None else 1
                record(PathSpec([(eid, e.u)], parent, order))
    return paths


def assemble(g: Graph, spec: PathSpec) -> Tuple[np.ndarray, np.ndarray]:
    pts_parts, flag_parts = [], []
    for k, (eid, start) in enumerate(spec.steps):
        p, f = g.edges[eid].oriented(start)
        if k:
            p = p[1:]
        pts_parts.append(p)
        flag_parts.append(f)
    return np.vstack(pts_parts), np.concatenate(flag_parts)


__all__ = [
    "Edge",
    "Graph",
    "PathSpec",
    "from_skeleton",
    "from_tree",
    "prune",
    "bridge",
    "extend_ends",
    "decompose",
    "assemble",
]

