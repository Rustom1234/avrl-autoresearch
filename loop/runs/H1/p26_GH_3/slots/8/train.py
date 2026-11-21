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


def _optimize(n, x0, bounds=None):
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", bounds=bounds,
                   constraints={"type": "ineq", "fun": cons}, options={"maxiter": MAX_ITER})
    return res.x[:2 * n].reshape(n, 2), res.x[2 * n:]


def _solve_once(n, rng):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    return _optimize(n, x0)


def _nested(n, c, r, rng, k=10, rounds=4):
    """Pin the centres of the k largest circles, re-solve the rest, rotate the pinned set, keep the best."""
    best = (c, r)
    for rd in range(rounds):
        order = np.argsort(-best[1])
        fixed = np.roll(order, -rd * k)[:k]
        cur_c, cur_r = best
        x0 = np.concatenate([cur_c.ravel(), cur_r])
        free = np.ones(n, bool)
        free[fixed] = False
        cc = x0[:2 * n].reshape(n, 2)
        cc[free] = np.clip(cc[free] + rng.normal(0, 0.02, (free.sum(), 2)), 0, 1)
        bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
        for i in fixed:
            bounds[2 * i] = (cur_c[i, 0], cur_c[i, 0])
            bounds[2 * i + 1] = (cur_c[i, 1], cur_c[i, 1])
        c2, r2 = _optimize(n, x0, bounds)
        c2 = np.clip(c2, 0, 1)
        r2 = _repair(c2, r2)
        if r2.sum() > best[1].sum():
            best = (c2, r2)
    return best


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
        c, r = _nested(n, c, r, rng)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
