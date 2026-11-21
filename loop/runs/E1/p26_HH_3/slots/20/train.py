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
RESTART_FRACTION = 0.3


def _optimise(n, x0):
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": MAX_ITER})
    return res.x[:2 * n].reshape(n, 2), res.x[2 * n:]


def _solve_once(n, rng):
    k = min(int(rng.integers(3, 6)), n)
    cuts = np.sort(rng.choice(np.arange(1, n), k - 1, replace=False))
    counts = np.diff(np.concatenate([[0], cuts, [n]]))
    pts = []
    for j, m in enumerate(counts):
        R = 0.42 * (k - j) / k
        ang = rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.arange(m) / m
        pts.append(np.stack([0.5 + R * np.cos(ang), 0.5 + R * np.sin(ang)], 1))
    c = np.clip(np.vstack(pts) + rng.normal(0, 0.02, (n, 2)), 0, 1)
    x0 = np.concatenate([c.ravel(), np.full(n, INIT_RADIUS)])
    return _optimise(n, x0)


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
    budget = TIME_FRACTION * time_limit
    while time.process_time() - t0 < RESTART_FRACTION * budget or best is None:   # short random-restart phase
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    c, r = best
    while time.process_time() - t0 < budget:   # remove-and-reinsert the smallest circle
        c2, r2 = c.copy(), r.copy()
        i = np.argmin(r2)
        c2[i] = rng.uniform(0, 1, 2)
        r2[i] = 0
        c2, r2 = _optimise(n, np.concatenate([c2.ravel(), r2]))
        c2 = np.clip(c2, 0, 1)
        r2 = _repair(c2, r2)
        if r2.sum() > r.sum():
            c, r = c2, r2
    return c, r
