"""train.py: packs n circles into the unit square so that the sum of their radii is as large as possible.

The harness calls solve(n, seed, time_limit) and checks the returned packing exactly.
Returns (centers, radii): centers is an (n, 2) array, radii an (n,) array. Every circle must lie inside
[0, 1] x [0, 1] and no two circles may overlap.
"""
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

# Settings
TIME_FRACTION = 0.8
INIT_LOW = 0.1
MAX_ITER = 200
INIT_RADIUS = 0.05
SHRINK = 1e-9


def _holes(n, rng):
    """Greedy hole-filling: add circumcentres of the largest Delaunay holes, choosing among the top 3 at random."""
    c = rng.uniform(INIT_LOW, 1 - INIT_LOW, (3, 2))
    r = np.full(3, INIT_RADIUS)
    sent = np.array([[-1, -1], [2, -1], [-1, 2], [2, 2], [0.5, -2], [0.5, 3], [-2, 0.5], [3, 0.5]])
    while len(c) < n:
        tri = Delaunay(np.vstack([c, sent]))
        a, b, e = tri.points[tri.simplices].transpose(1, 0, 2)
        with np.errstate(all="ignore"):
            D = 2 * (a[:, 0] * (b[:, 1] - e[:, 1]) + b[:, 0] * (e[:, 1] - a[:, 1]) + e[:, 0] * (a[:, 1] - b[:, 1]))
            sa, sb, se = (a ** 2).sum(1), (b ** 2).sum(1), (e ** 2).sum(1)
            h = np.column_stack([(sa * (b[:, 1] - e[:, 1]) + sb * (e[:, 1] - a[:, 1]) + se * (a[:, 1] - b[:, 1])) / D,
                                 (sa * (e[:, 0] - b[:, 0]) + sb * (a[:, 0] - e[:, 0]) + se * (b[:, 0] - a[:, 0])) / D])
        h = h[np.isfinite(h).all(1) & (h >= 0).all(1) & (h <= 1).all(1)]
        if len(h) == 0:
            h = rng.uniform(0.05, 0.95, (1, 2))
        dc = np.sqrt(((h[:, None] - c[None]) ** 2).sum(-1)).min(1)
        s = np.minimum(dc, np.minimum(h, 1 - h).min(1))
        j = np.argsort(-s)[:3][rng.integers(min(3, len(s)))]
        c = np.vstack([c, h[j]])
        r = np.append(r, s[j])
    return c, r


def _solve_once(n, rng):
    c0, r0 = _holes(n, rng)
    x0 = np.concatenate([c0.ravel(), r0])
    iu = np.triu_indices(n, 1)

    def cons(v, cap):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r, cap - r])

    x = x0
    for cap in list(np.arange(0.05, 0.4 + 1e-9, 0.02)) + [1.0]:
        res = minimize(lambda v: -v[2 * n:].sum(), x, method="SLSQP",
                       constraints={"type": "ineq", "fun": lambda v, cap=cap: cons(v, cap)},
                       options={"maxiter": MAX_ITER})
        x = res.x
    return x[:2 * n].reshape(n, 2), x[2 * n:]


def _repair(c, r):
    """Shrink radii until the packing is valid, so that the harness never sees an overlap."""
    n = len(r)
    r = np.minimum(r, np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1 - c[:, 0], 1 - c[:, 1])))
    r = np.maximum(r, 0)
    for _ in range(50):
        d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
        over = r[:, None] + r[None] - d
        np.fill_diagonal(over, -1)
        i, j = np.unravel_index(np.argmax(over), over.shape)
        if over[i, j] <= 0:
            break
        s = d[i, j] / (r[i] + r[j])
        r[i], r[j] = r[i] * s, r[j] * s
    return np.maximum(r - SHRINK, 0)


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
