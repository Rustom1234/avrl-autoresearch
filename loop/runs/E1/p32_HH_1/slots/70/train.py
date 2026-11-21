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


def _solve_once(n, rng, init=None, jitter=0.03, maxiter=None):
    base = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2)) if init is None else np.clip(init + rng.normal(0, jitter, (n, 2)), 0, 1)
    x0 = np.concatenate([base.ravel(), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": MAX_ITER if maxiter is None else maxiter})
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
    budget = TIME_FRACTION * time_limit
    pool, guided_count = [], 0
    while True:
        frac = (time.process_time() - t0) / budget
        if frac >= 0.85:
            break
        if frac < 0.6 or best is None:   # phase 1: random restarts
            c, r = _solve_once(n, rng)
        else:                            # phase 2: guided restarts with halving jitter
            jitter = 0.03 * 0.5 ** (guided_count // 10)
            guided_count += 1
            c, r = _solve_once(n, rng, best[0], jitter)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        pool.append((c, r))
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    # phase 3: one long zero-jitter solve from each of the top 3 distinct packings
    tops = []
    for c, r in sorted(pool, key=lambda p: -p[1].sum()):
        if all(abs(r.sum() - t[1].sum()) > 1e-9 for t in tops):
            tops.append((c, r))
        if len(tops) == 3:
            break
    for c, _ in tops:
        c2, r2 = _solve_once(n, rng, c, 0.0, 1000)
        c2 = np.clip(c2, 0, 1)
        r2 = _repair(c2, r2)
        if r2.sum() > best[1].sum():
            best = (c2, r2)
    return best
