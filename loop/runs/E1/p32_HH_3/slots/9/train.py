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
SUBSET = 12


def _solve_once(n, rng, x0=None):
    if x0 is None:
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


def _solve_subset(c, r, idx):
    """SLSQP over the circles in idx only; all other circles are fixed obstacles."""
    k = len(idx)
    fixed = np.setdiff1d(np.arange(len(r)), idx)
    cf, rf = c[fixed], r[fixed]
    iu = np.triu_indices(k, 1)

    def cons(v):
        cs, rs = v[:2 * k].reshape(k, 2), v[2 * k:]
        d = np.sqrt(((cs[iu[0]] - cs[iu[1]]) ** 2).sum(1)) - rs[iu[0]] - rs[iu[1]]
        df = np.sqrt(((cs[:, None] - cf[None]) ** 2).sum(-1)) - rs[:, None] - rf[None]
        return np.concatenate([d, df.ravel(), cs[:, 0] - rs, cs[:, 1] - rs, 1 - cs[:, 0] - rs, 1 - cs[:, 1] - rs, rs])

    x0 = np.concatenate([c[idx].ravel(), r[idx]])
    res = minimize(lambda v: -v[2 * k:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": MAX_ITER})
    c2, r2 = c.copy(), r.copy()
    c2[idx] = res.x[:2 * k].reshape(k, 2)
    r2[idx] = res.x[2 * k:]
    return c2, r2


def solve(n, seed, time_limit):
    rng, t0 = np.random.default_rng(seed), time.process_time()
    c, r = _solve_once(n, rng)
    c = np.clip(c, 0, 1)
    best = (c, _repair(c, r))
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random subset re-optimisation
        bc, br = best
        p = rng.uniform(0, 1, 2)
        idx = np.argsort(((bc - p) ** 2).sum(1))[:min(SUBSET, n)]
        c, r = _solve_subset(bc, br, idx)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if r.sum() > br.sum():
            best = (c, r)
    return best
