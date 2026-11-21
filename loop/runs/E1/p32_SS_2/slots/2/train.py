"""train.py: packs n circles into the unit square so that the sum of their radii is as large as possible.

The harness calls solve(n, seed, time_limit) and checks the returned packing exactly.
Returns (centers, radii): centers is an (n, 2) array, radii an (n,) array. Every circle must lie inside
[0, 1] x [0, 1] and no two circles may overlap.
"""
import time

import numpy as np
from scipy.optimize import minimize

# Settings
TIME_FRACTION = 0.8
INIT_LOW = 0.1
MAX_ITER = 200
INIT_RADIUS = 0.05
SHRINK = 1e-9


def _reinsert_start(c, r, rng):
    """Move the smallest circle to the largest sampled hole and shrink the others slightly."""
    k = int(np.argmin(r))
    keep = np.arange(len(r)) != k
    P = rng.random((2000, 2))
    d = np.sqrt(((P[:, None] - c[keep][None]) ** 2).sum(-1)) - r[keep][None]
    h = np.minimum(d.min(1), np.minimum(P, 1 - P).min(1))
    c, r = c.copy(), r * 0.95
    c[k] = P[np.argmax(h)]
    r[k] = max(h.max(), 0.01) * 0.95
    return np.concatenate([c.ravel(), r])


def _solve_once(n, rng, x0=None):
    if x0 is None:
        R = int(rng.integers(5, 8))
        m = int(np.ceil(n / R))
        pts = np.array([((j + 0.5 + 0.5 * (i % 2)) / (m + 0.5), (i + 0.5) / R) for i in range(R) for j in range(m)])
        pts = pts[rng.permutation(len(pts))[:n]] + rng.normal(0, 0.02, (n, 2))
        pts = np.clip(pts, 0.03, 0.97)
        x0 = np.concatenate([pts.ravel(), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": MAX_ITER})
    return res.x[:2 * n].reshape(n, 2), res.x[2 * n:]


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
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        if time.process_time() - t0 < 0.5 * time_limit:   # random restarts
            c, r = _solve_once(n, rng)
        else:                                             # remove-and-reinsert local search
            c, r = _solve_once(n, rng, _reinsert_start(best[0], best[1], rng))
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
