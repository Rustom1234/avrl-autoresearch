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


def _solve_once(n, rng):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
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
    c = c.copy()
    r = r.copy()
    r = np.minimum(r, np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1 - c[:, 0], 1 - c[:, 1])))
    r = np.maximum(r, 0)
    for _ in range(100):
        d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1) + 1e-9)
        over = r[:, None] + r[None] - d
        np.fill_diagonal(over, -1)
        if np.max(over) <= 0:
            break
        i, j = np.unravel_index(np.argmax(over), over.shape)
        dist = d[i, j]
        r_sum = r[i] + r[j]
        # Push centers apart
        move_dir = (c[i] - c[j]) / (dist + 1e-9)
        c[i] += move_dir * 0.1 * (r_sum - dist)
        c[j] -= move_dir * 0.1 * (r_sum - dist)
        # Shrink radii
        r[i] *= 0.99
        r[j] *= 0.99
    r = np.maximum(r - SHRINK, 0)
    return c, r

def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c_raw, r_raw = _solve_once(n, rng)
        c = np.clip(c_raw, 0, 1)
        c, r = _repair(c, r_raw)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
