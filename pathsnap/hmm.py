"""HMM + Viterbi route reconstruction.

Hidden state at time i : one candidate arc of observation i (with its projection
                         point on that arc), so the candidate -> arc link is kept.
Emission  log P(z_i | s) : w * log( classifier probability, renormalised over the
                         candidates of observation i ).
Transition log P(s' | s) : -|d_route(s, s') - d_gc(z_i-1, z_i)| / beta
                         (Newson & Krumm 2009): a move is likely when the distance
                         driven along the network between the two projection
                         points matches the straight-line distance between the
                         two GPS fixes. d_route uses Dijkstra on the trip's
                         directed road graph.
Viterbi recurrence     : V_i(s') = max_s [V_{i-1}(s) + log T(s, s')] + log E_i(s')
Breaks                 : a time gap > gap_s seconds, or no reachable transition
                         at all, ends the current HMM chain and starts a new one.
Missing candidates     : observations with no candidate inside the search radius
                         are skipped by the chain and get the arc of the nearest
                         decoded neighbour afterwards.
"""
import numpy as np
import pandas as pd

NEG = -1e18


def viterbi_decode(cands: pd.DataFrame, prob: np.ndarray, net, xy: np.ndarray, t: np.ndarray,
                   beta: float = 20.0, emis_w: float = 1.0, gap_s: float = 60.0,
                   back_tol: float = 25.0, cutoff_extra: float = 200.0):
    """cands: rows for ONE trip sorted by (obs, dist) with columns obs, arc, proj_frac.

    Returns (decoded arc per observation (N,), number of chain breaks)."""
    N = len(xy)
    obs = cands["obs"].to_numpy()
    arcs = cands["arc"].to_numpy()
    pos = cands["proj_frac"].to_numpy() * net.length[arcs]
    p = np.clip(prob, 1e-6, 1.0)
    bounds = np.r_[0, np.flatnonzero(np.diff(obs)) + 1, len(obs)]
    steps = [(int(obs[s]), arcs[s:e], pos[s:e], p[s:e] / p[s:e].sum()) for s, e in zip(bounds[:-1], bounds[1:])]

    decoded = np.full(N, -1, dtype=np.int64)
    n_breaks = 0
    chain = []           # list of (obs index, arcs, backpointers)
    score = None
    prev = None

    def flush():
        if not chain:
            return
        k = int(np.argmax(score))
        for (o, a, bp) in reversed(chain):
            decoded[o] = a[k]
            k = bp[k] if bp is not None else k

    for (o, a, ps, pe) in steps:
        le = emis_w * np.log(pe)
        if prev is None:
            score, chain = le.copy(), [(o, a, None)]
            prev = (o, a, ps)
            continue
        po, pa, pps = prev
        dt = t[o] - t[po]
        d_gc = float(np.hypot(*(xy[o] - xy[po])))
        trans = np.full((len(pa), len(a)), NEG)
        if dt <= gap_s:
            cutoff = 2.0 * d_gc + cutoff_extra
            for i, (ai, pi) in enumerate(zip(pa, pps)):
                rem = net.length[ai] - pi
                dd = net.dijkstra(int(net.arcs[ai, 1]), cutoff)
                for j, (bj, pj) in enumerate(zip(a, ps)):
                    if bj == ai and pj >= pi - back_tol:
                        dr = max(pj - pi, 0.0)
                    else:
                        dn = dd.get(int(net.arcs[bj, 0]))
                        if dn is None:
                            continue
                        dr = rem + dn + pj
                    trans[i, j] = -abs(dr - d_gc) / beta
        tot = score[:, None] + trans
        best = tot.max(axis=0)
        if best.max() <= NEG / 2:          # gap or nothing reachable -> new chain
            flush()
            n_breaks += 1
            score, chain = le.copy(), [(o, a, None)]
        else:
            bp = tot.argmax(axis=0)
            score = best + le
            score -= score.max()
            chain.append((o, a, bp))
        prev = (o, a, ps)
    flush()
    return fill_missing(decoded), n_breaks


def fill_missing(decoded: np.ndarray) -> np.ndarray:
    s = pd.Series(np.where(decoded >= 0, decoded, np.nan))
    s = s.ffill().bfill()
    return s.fillna(-1).to_numpy().astype(np.int64)


def classifier_only(cands: pd.DataFrame, prob: np.ndarray, N: int) -> np.ndarray:
    d = cands[["obs", "arc"]].assign(p=prob)
    best = d.loc[d.groupby("obs")["p"].idxmax()]
    out = np.full(N, -1, dtype=np.int64)
    out[best["obs"].to_numpy()] = best["arc"].to_numpy()
    return fill_missing(out)


def reconstruct_route(obs_arcs: np.ndarray, net, max_bridge: float = 3000.0) -> list[int]:
    """Collapse the per-observation arcs into an ordered arc list, bridging
    non-adjacent consecutive arcs with the shortest network path."""
    seq = [int(a) for a in obs_arcs if a >= 0]
    seq = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    route = []
    for i, a in enumerate(seq):
        if i > 0:
            b = seq[i - 1]
            if net.arcs[b, 1] != net.arcs[a, 0]:
                route.extend(net.path_arcs(b, a, cutoff=max_bridge))
        route.append(a)
    return route
