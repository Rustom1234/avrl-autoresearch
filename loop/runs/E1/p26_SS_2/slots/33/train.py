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


def _solve_once(n, rng, start=None, sd=0.0):
    if start is None:
        x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    else:
        x0 = np.concatenate([(start[0] + rng.normal(0, sd, start[0].shape)).ravel(), start[1] * 0.5])
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


def _crossover(n, rng, a, b):
    t = rng.uniform(0.3, 0.7)
    ma, mb = a[0][:, 0] < t, b[0][:, 0] >= t
    c = np.concatenate([a[0][ma], b[0][mb]])
    r = np.concatenate([a[1][ma], b[1][mb]])
    if len(r) > n:
        keep = np.argsort(-r)[:n]
        c, r = c[keep], r[keep]
    elif len(r) < n:
        k = n - len(r)
        c = np.concatenate([c, rng.uniform(0.05, 0.95, (k, 2))])
        r = np.concatenate([r, np.full(k, 0.02)])
    return c, r


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    pool = []
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        if len(pool) >= 2 and time.process_time() - t0 > 0.3 * time_limit:
            i, j = rng.choice(len(pool), 2, replace=False)
            child = _crossover(n, rng, pool[i], pool[j])
            c, r = _solve_once(n, rng, child, 0.0)
        else:
            c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if all(abs(r.sum() - p[1].sum()) > 1e-6 for p in pool):
            pool = sorted(pool + [(c, r)], key=lambda p: -p[1].sum())[:4]
        best = pool[0]
    return best
