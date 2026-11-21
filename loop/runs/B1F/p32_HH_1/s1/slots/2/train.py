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


def _solve_once(n, rng, init=None, maxiter=MAX_ITER, x0=None, ftol=1e-6):
    if x0 is None:
        base = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2)) if init is None else np.clip(init + rng.normal(0, 0.03, (n, 2)), 0, 1)
        x0 = np.concatenate([base.ravel(), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": maxiter, "ftol": ftol})
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
    rng, t0 = np.random.default_rng(seed), time.process_time()
    pool, longest = [], 0.0   # top 5 phase-1 packings as (sum, centres, radii), best first
    while time.process_time() - t0 < 0.55 * time_limit:   # phase 1: cheap screening
        s = time.process_time()
        c, r = _solve_once(n, rng, maxiter=60)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        longest = max(longest, time.process_time() - s)
        if all(abs(r.sum() - p[0]) > 1e-6 for p in pool):
            pool = sorted(pool + [(r.sum(), c, r)], key=lambda p: -p[0])[:5]
    best = pool[0][1:]
    for _, c0, r0 in pool:   # phase 2: polish each kept packing from its exact centres and radii
        if 0.85 * time_limit - (time.process_time() - t0) < 2 * longest:
            break
        s = time.process_time()
        c, r = _solve_once(n, rng, maxiter=1000, x0=np.concatenate([c0.ravel(), r0]), ftol=1e-12)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        longest = max(longest, time.process_time() - s)
        if r.sum() > best[1].sum():
            best = (c, r)
    return best
