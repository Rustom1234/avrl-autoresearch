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


def _solve_once(n, rng, init=None):
    base = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2)) if init is None else np.clip(init + rng.normal(0, 0.03, (n, 2)), 0, 1)
    x0 = np.concatenate([base.ravel(), np.full(n, INIT_RADIUS)])
    k = min(8, n - 1)

    def cons(v, pi, pj):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[pi] - c[pj]) ** 2).sum(1)) - r[pi] - r[pj]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    v, prev = x0, np.inf
    for _ in range(4):
        c = v[:2 * n].reshape(n, 2)
        d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
        near = d < 0.45
        near[np.arange(n)[:, None], np.argsort(d, axis=1)[:, 1:k + 1]] = True
        pi, pj = np.nonzero(np.triu(near | near.T, 1))
        res = minimize(lambda w: -w[2 * n:].sum(), v, method="SLSQP",
                       constraints={"type": "ineq", "fun": lambda w: cons(w, pi, pj)}, options={"maxiter": 60})
        v = res.x
        obj = v[2 * n:].sum()
        if abs(obj - prev) < 1e-9:
            break
        prev = obj
    return v[:2 * n].reshape(n, 2), v[2 * n:]


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
