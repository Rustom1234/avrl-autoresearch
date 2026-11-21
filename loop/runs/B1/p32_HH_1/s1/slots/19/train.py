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


def _overlap(v, n, R):
    """Sum of squared pair and wall overlaps for equal radii R, as a function of the centres."""
    c = v.reshape(n, 2)
    iu = np.triu_indices(n, 1)
    d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1))
    p = np.maximum(2 * R - d, 0)
    w = np.maximum(R - c, 0) + np.maximum(c - (1 - R), 0)
    return (p ** 2).sum() + (w ** 2).sum()


def _continuation(n, base):
    """Bisect a common radius R in [0.05, 0.12]; each step relaxes overlaps at fixed R, warm-started from the previous step."""
    lo, hi, v = 0.05, 0.12, base.ravel()
    best_R, best_v = lo, v
    for _ in range(6):
        R = 0.5 * (lo + hi)
        v = minimize(_overlap, v, args=(n, R), method="SLSQP", options={"maxiter": 60}).x
        if _overlap(v, n, R) < 1e-8:
            lo, best_R, best_v = R, R, v
        else:
            hi = R
    return best_R, best_v


def _solve_once(n, rng, init=None):
    base = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2)) if init is None else np.clip(init + rng.normal(0, 0.03, (n, 2)), 0, 1)
    radii = np.full(n, INIT_RADIUS)
    if init is None and rng.random() < 0.5:   # radius-continuation restart
        R, v = _continuation(n, base)
        base, radii = v.reshape(n, 2), np.full(n, R)
    x0 = np.concatenate([base.ravel(), radii])
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
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        guided = best is not None and time.process_time() - t0 > 0.5 * TIME_FRACTION * time_limit and rng.random() < 0.5
        c, r = _solve_once(n, rng, best[0] if guided else None)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
