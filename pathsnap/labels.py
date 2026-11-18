"""Derive an observation-level reference arc from the supplied .route.

The .route file is only the ordered list of arcs the vehicle traversed; it does
not say which arc each GPS observation belongs to. We align observations to the
route with a monotone dynamic programme: observation i is assigned route
position j_i with j_0 <= j_1 <= ... (the vehicle never goes back along the
route), minimising the total point-to-segment distance. Arcs may be skipped
(short arcs passed between two 1 Hz samples).

This label is used only for training/evaluation; it never enters the prediction
pipeline.
"""
import numpy as np

from .geo import point_segment


def align_route(pts_xy: np.ndarray, net, route: np.ndarray, clip: float = 200.0):
    N, M = len(pts_xy), len(route)
    A, B = net.A[route], net.B[route]
    back = np.empty((N, M), dtype=np.int32 if M > 32000 else np.int16)
    idx = np.arange(M)
    dp = None
    for i in range(N):
        c, _ = point_segment(np.repeat(pts_xy[i:i + 1], M, 0), A, B)
        c = np.minimum(c, clip)
        if dp is None:
            dp = c
            back[i] = idx
            continue
        pm = np.minimum.accumulate(dp)
        arg = np.maximum.accumulate(np.where(dp == pm, idx, 0))
        dp = c + pm
        back[i] = arg
    j = int(np.argmin(dp))
    pos = np.empty(N, dtype=int)
    for i in range(N - 1, -1, -1):
        pos[i] = j
        j = int(back[i, j])
    ref = route[pos]
    d, _ = point_segment(pts_xy, net.A[ref], net.B[ref])
    return ref, pos, d
