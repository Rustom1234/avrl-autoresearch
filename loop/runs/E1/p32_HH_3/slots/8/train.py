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


def _penalty_start(n, v, iu):
    """Penalty continuation: minimise squared overlap/wall violations minus lambda * radius sum, lambda increasing."""
    def fg(v, lam):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        i, j = iu
        diff = c[i] - c[j]
        d = np.sqrt((diff ** 2).sum(1)) + 1e-12
        ov = np.maximum(0, r[i] + r[j] - d)
        f = (ov ** 2).sum() - lam * r.sum()
        coef = 2 * ov
        g_r = np.full(n, -lam) + np.bincount(i, coef, n) + np.bincount(j, coef, n)
        u = (-coef / d)[:, None] * diff
        g_c = np.stack([np.bincount(i, u[:, k], n) - np.bincount(j, u[:, k], n) for k in range(2)], 1)
        a = np.maximum(0, r[:, None] - c)
        b = np.maximum(0, c + r[:, None] - 1)
        f += (a ** 2).sum() + (b ** 2).sum()
        g_c = g_c - 2 * a + 2 * b
        g_r = g_r + 2 * a.sum(1) + 2 * b.sum(1)
        return f, np.concatenate([g_c.ravel(), g_r])

    bounds = [(None, None)] * (2 * n) + [(0, None)] * n
    for lam in np.geomspace(0.1, 100, 5):
        v = minimize(fg, v, args=(lam,), jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200}).x
    return v


def _solve_once(n, rng, x0=None):
    if x0 is None:
        x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)
    x0 = _penalty_start(n, x0, iu)

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
    rng, t0 = np.random.default_rng(seed), time.process_time()
    c, r = _solve_once(n, rng)
    c = np.clip(c, 0, 1)
    best = (c, _repair(c, r))
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # basin hopping from the incumbent
        bc, br = best
        c0 = np.clip(bc + rng.normal(0, 0.02, bc.shape), 0, 1)
        r0 = np.maximum(br + rng.normal(0, 0.02, br.shape), 0)
        c, r = _solve_once(n, rng, np.concatenate([c0.ravel(), r0]))
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if r.sum() > best[1].sum():
            best = (c, r)
    return best
