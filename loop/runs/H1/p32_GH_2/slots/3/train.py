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
PENALTY = 100.0


def _solve_once(n, rng, start=None):
    if start is None:
        centers = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2))
    else:
        centers = np.clip(start + rng.normal(0, 0.02, (n, 2)), 0, 1)
    x0 = np.concatenate([centers.ravel(), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def obj(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1) + 1e-18)
        pair = np.maximum(0, r[iu[0]] + r[iu[1]] - d)
        wall = np.maximum(0, np.stack([r - c[:, 0], r - c[:, 1], c[:, 0] + r - 1, c[:, 1] + r - 1], 1))
        return -r.sum() + PENALTY * (pair.sum() + wall.sum())

    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, options={"maxiter": MAX_ITER})
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
    rng, t0, best, start = np.random.default_rng(seed), time.process_time(), None, None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # restarts until the time is used
        c, r = _solve_once(n, rng, start)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
        start = best[0] if rng.random() < 0.5 else None   # reuse the best centres half of the time
    return best
