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
MC_SWEEPS = 100
MC_STEP = 0.05


def _grow(c):
    """Largest radii that keep the packing valid for fixed centres (conservative: r_i <= d_ij / 2)."""
    d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    wall = np.minimum(c, 1 - c).min(1)
    return np.maximum(np.minimum(wall, 0.5 * d.min(1)), 0)


def _mc_start(n, rng):
    """Hard-disk Monte Carlo compression: accept random displacements that keep the packing valid."""
    c = rng.uniform(0.05, 0.95, (n, 2))
    r = _grow(c)
    for _ in range(MC_SWEEPS):
        for i in range(n):
            p = c[i] + rng.uniform(-MC_STEP, MC_STEP, 2)
            if p.min() < r[i] or p.max() > 1 - r[i]:
                continue
            d = np.sqrt(((c - p) ** 2).sum(1))
            d[i] = np.inf
            if np.all(d >= r + r[i]):
                c[i] = p
        r = _grow(c)
    return c, r


def _solve_once(n, rng):
    c0, r0 = _mc_start(n, rng)
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
