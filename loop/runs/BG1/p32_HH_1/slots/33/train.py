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


def _lloyd(c, rng, iters=4):
    """Weighted Lloyd relaxation of the centres on a 60 x 60 grid, with a small random step per iteration."""
    g = (np.arange(60) + 0.5) / 60
    gx, gy = np.meshgrid(g, g)
    p = np.stack([gx.ravel(), gy.ravel()], 1)
    for _ in range(iters):
        lab = np.argmin(((p[:, None] - c[None]) ** 2).sum(-1), 1)
        cnt = np.bincount(lab, minlength=len(c))
        s = np.stack([np.bincount(lab, p[:, k], minlength=len(c)) for k in range(2)], 1)
        m = cnt > 0
        c = c.copy()
        c[m] = s[m] / cnt[m, None]
        c = np.clip(c + rng.normal(0, 0.01, c.shape), 0, 1)
    return c


def solve(n, seed, time_limit):
    rng, t0 = np.random.default_rng(seed), time.process_time()
    budget = TIME_FRACTION * time_limit
    cur = best = None
    while time.process_time() - t0 < budget:
        if cur is None:
            c, r = _solve_once(n, rng)
        else:
            frac = (time.process_time() - t0) / budget
            T = 0.01 * (1e-4 / 0.01) ** frac
            c0 = _lloyd(cur[0], rng)
            c, r = _solve_once(n, rng, np.concatenate([c0.ravel(), cur[1]]))
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if cur is None or r.sum() >= cur[1].sum() or np.exp((r.sum() - cur[1].sum()) / T) > rng.uniform():
            cur = (c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
