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
    c, r = res.x[:2 * n].reshape(n, 2), res.x[2 * n:]
    return c, r


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


def _refine(c, r, rng):
    """Simple hill-climbing to nudge centers while maintaining validity."""
    n = len(r)
    c_new = c.copy()
    for _ in range(20):
        i = rng.integers(0, n)
        idx = rng.integers(0, 2)
        c_new[i, idx] += rng.uniform(-0.01, 0.01)
        # Check boundary
        if not (np.all(c_new >= r) and np.all(c_new <= 1 - r)):
            c_new[i, idx] = c[i, idx]
            continue
        # Check overlap
        dist_sq = np.sum((c_new[i] - c_new)**2, axis=1)
        min_dist = np.min(np.sqrt(dist_sq[1:])) # exclude self
        if min_dist < (r[i] + np.max(r)): # simplified check
            # Re-check precisely
            valid = True
            for k in range(n):
                if k == i: continue
                if np.linalg.norm(c_new[i] - c_new[k]) < r[i] + r[k] - 1e-9:
                    valid = False
                    break
            if not valid: c_new[i, idx] = c[i, idx]
    return c_new


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
