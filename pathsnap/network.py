"""Road network: geometry in metres, spatial index for candidates, shortest paths."""
import heapq

import numpy as np
from scipy.spatial import cKDTree

from .geo import point_segment


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj):
        self.proj = proj
        self.arcs = arcs.astype(np.int64)
        self.nxy = proj.fwd(nodes_lonlat)
        self.A = self.nxy[self.arcs[:, 0]]
        self.B = self.nxy[self.arcs[:, 1]]
        d = self.B - self.A
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = np.arctan2(d[:, 0], d[:, 1])  # compass bearing, radians
        n = len(nodes_lonlat)
        self.out_deg = np.bincount(self.arcs[:, 0], minlength=n)
        self.in_deg = np.bincount(self.arcs[:, 1], minlength=n)
        # CSR adjacency: node -> outgoing arcs
        order = np.argsort(self.arcs[:, 0], kind="stable")
        self._out_arcs = order
        self._out_ptr = np.r_[0, np.cumsum(self.out_deg)]
        self._tree = None
        self._sp_cache: dict[int, tuple[float, dict]] = {}

    # ---------------------------------------------------------------- index
    def build_index(self, pts_xy: np.ndarray, radius: float, step: float = 10.0):
        """Densify only arcs near the track and index the sample points."""
        lo = pts_xy.min(0) - radius - 50
        hi = pts_xy.max(0) + radius + 50
        amin = np.minimum(self.A, self.B)
        amax = np.maximum(self.A, self.B)
        near = np.where((amax[:, 0] >= lo[0]) & (amin[:, 0] <= hi[0]) &
                        (amax[:, 1] >= lo[1]) & (amin[:, 1] <= hi[1]))[0]
        nseg = np.maximum(1, np.ceil(self.length[near] / step).astype(int))
        arc_rep = np.repeat(near, nseg + 1)
        offs = np.concatenate([np.linspace(0, 1, k + 1) for k in nseg]) if len(near) else np.zeros(0)
        samples = self.A[arc_rep] + (self.B[arc_rep] - self.A[arc_rep]) * offs[:, None]
        self._sample_arc = arc_rep
        self._tree = cKDTree(samples) if len(samples) else None
        self._step = step

    def candidates(self, pts_xy: np.ndarray, radius: float, max_k: int):
        """All arcs within `radius` metres of each point (nearest `max_k` kept).

        Returns flat arrays (obs_idx, arc_id, dist, frac) sorted by obs then dist."""
        if self._tree is None:
            self.build_index(pts_xy, radius)
        if self._tree is None:
            return (np.zeros(0, int),) * 2 + (np.zeros(0),) * 2
        lists = self._tree.query_ball_point(pts_xy, r=radius + self._step * 0.75)
        counts = np.fromiter((len(l) for l in lists), int, len(lists))
        if counts.sum() == 0:
            return (np.zeros(0, int),) * 2 + (np.zeros(0),) * 2
        pi = np.repeat(np.arange(len(pts_xy)), counts)
        ai = self._sample_arc[np.concatenate([l for l in lists if l])]
        nA = len(self.arcs)
        key = np.unique(pi.astype(np.int64) * nA + ai)
        pi, ai = key // nA, key % nA
        d, frac = point_segment(pts_xy[pi], self.A[ai], self.B[ai])
        m = d <= radius
        pi, ai, d, frac = pi[m], ai[m], d[m], frac[m]
        order = np.lexsort((d, pi))
        pi, ai, d, frac = pi[order], ai[order], d[order], frac[order]
        start = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]
        rank = np.arange(len(pi)) - np.repeat(start, np.diff(np.r_[start, len(pi)]))
        k = rank < max_k
        return pi[k], ai[k], d[k], frac[k]

    def dist_to_arcs(self, pts_xy: np.ndarray, arc_ids: np.ndarray):
        return point_segment(pts_xy, self.A[arc_ids], self.B[arc_ids])

    # -------------------------------------------------------- shortest path
    def dijkstra(self, src: int, cutoff: float) -> dict:
        """Distances (m) from node `src` to every node within `cutoff` (cached)."""
        c = self._sp_cache.get(src)
        if c is not None and c[0] >= cutoff:
            return c[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        done = set()
        while heap:
            du, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for e in self._out_arcs[self._out_ptr[u]:self._out_ptr[u + 1]]:
                v = int(self.arcs[e, 1])
                nd = du + self.length[e]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        if len(self._sp_cache) > 20000:
            self._sp_cache.clear()
        self._sp_cache[src] = (cutoff, dist)
        return dist

    def path_arcs(self, a: int, b: int, cutoff: float = 3000.0) -> list[int]:
        """Arc sequence leading from the end of arc a to the start of arc b (exclusive)."""
        src, dst = int(self.arcs[a, 1]), int(self.arcs[b, 0])
        if src == dst:
            return []
        prev = {src: -1}
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            du, u = heapq.heappop(heap)
            if u == dst:
                break
            if du > dist.get(u, np.inf):
                continue
            for e in self._out_arcs[self._out_ptr[u]:self._out_ptr[u + 1]]:
                v = int(self.arcs[e, 1])
                nd = du + self.length[e]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    prev[v] = int(e)
                    heapq.heappush(heap, (nd, v))
        if dst not in prev:
            return []
        out = []
        v = dst
        while prev[v] != -1:
            e = prev[v]
            out.append(e)
            v = int(self.arcs[e, 0])
        return out[::-1]
