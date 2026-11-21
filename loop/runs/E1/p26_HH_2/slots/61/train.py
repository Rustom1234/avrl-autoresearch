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


def _polish(c, r, n):
    """Final SLSQP stage from a warm start, with radius at most 1."""
    iu = np.triu_indices(n, 1)

    def cons(v):
        c_, r_ = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c_[iu[0]] - c_[iu[1]]) ** 2).sum(1)) - r_[iu[0]] - r_[iu[1]]
        return np.concatenate([d, c_[:, 0] - r_, c_[:, 1] - r_, 1 - c_[:, 0] - r_, 1 - c_[:, 1] - r_, r_, 1 - r_])

    x = np.concatenate([np.clip(c, 0, 1).ravel(), np.maximum(r, 0)])
    res = minimize(lambda v: -v[2 * n:].sum(), x, method="SLSQP",
                   constraints={"type": "ineq", "fun": cons}, options={"maxiter": MAX_ITER})
    return res.x[:2 * n].reshape(n, 2), res.x[2 * n:]


def _insert_two(c, r):
    """Insert two circles at the largest holes found on a 200 x 200 grid."""
    g = np.linspace(0, 1, 200)
    gx, gy = np.meshgrid(g, g)
    p = np.stack([gx.ravel(), gy.ravel()], 1)
    for _ in range(2):
        d = np.sqrt(((p[:, None] - c[None]) ** 2).sum(-1)) - r[None]
        wall = np.minimum(np.minimum(p[:, 0], p[:, 1]), np.minimum(1 - p[:, 0], 1 - p[:, 1]))
        clear = np.minimum(d.min(1), wall)
        k = np.argmax(clear)
        c = np.vstack([c, p[k]])
        r = np.append(r, max(clear[k], 0))
    return c, r


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        cands = [(np.clip(c, 0, 1), r)]
        c24, r24 = _solve_once(n - 2, rng)
        c24, r24 = _insert_two(np.clip(c24, 0, 1), r24)
        cands.append(_polish(c24, r24, n))
        c28, r28 = _solve_once(n + 2, rng)
        c28, keep = np.clip(c28, 0, 1), np.argsort(r28)[2:]
        cands.append(_polish(c28[keep], r28[keep], n))
        for c, r in cands:
            c = np.clip(c, 0, 1)
            r = _repair(c, r)
            if best is None or r.sum() > best[1].sum():
                best = (c, r)
    return best
