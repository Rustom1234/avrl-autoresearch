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


def _prerelax(c, n):
    """Push centres apart by minimising a squared-overlap penalty with growing fixed radii."""
    iu = np.triu_indices(n, 1)
    for R in (0.06, 0.08, 0.10):
        def f(v):
            p = v.reshape(n, 2)
            dd = p[iu[0]] - p[iu[1]]
            d = np.sqrt((dd ** 2).sum(1)) + 1e-12
            o = np.maximum(2 * R - d, 0)
            lo, hi = np.maximum(R - p, 0), np.maximum(p + R - 1, 0)
            g = np.zeros_like(p)
            gp = (-2 * o / d)[:, None] * dd
            np.add.at(g, iu[0], gp)
            np.add.at(g, iu[1], -gp)
            g += -2 * lo + 2 * hi
            return (o ** 2).sum() + (lo ** 2).sum() + (hi ** 2).sum(), g.ravel()
        c = minimize(f, np.asarray(c).ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 10}).x
    return np.clip(c.reshape(n, 2), 0.02, 0.98)


def _solve_once(n, rng, start=None):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    if start is None:
        x0[:2 * n] = _prerelax(x0[:2 * n], n).ravel()
    if start is not None:
        c0, r0 = start[0].copy(), start[1].copy()
        if rng.random() < 0.5:
            idx = rng.choice(n, rng.integers(2, 5), replace=False)
            c0[idx] = rng.uniform(INIT_LOW, 1 - INIT_LOW, (len(idx), 2))
            r0[idx] = INIT_RADIUS * 0.5
        else:
            c0 = c0 + rng.normal(0, 0.02, c0.shape)
        x0 = np.concatenate([np.clip(c0, 0, 1).ravel(), r0 * 0.9])
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
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        explore = time.process_time() - t0 < 0.3 * TIME_FRACTION * time_limit or best is None
        c, r = _solve_once(n, rng, None if explore else best)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
