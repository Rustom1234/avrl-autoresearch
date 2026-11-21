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


def _compress(n, rng):
    """Lubachevsky-Stillinger-style growth: grow radii, push overlaps apart, halve the growth rate when jammed."""
    c_ok, r_ok = rng.uniform(0.05, 0.95, (n, 2)), np.full(n, 1e-3)
    g = 0.1
    for _ in range(2000):
        if g < 1e-6:
            break
        c, r = c_ok.copy(), r_ok * (1 + g)
        ok = False
        for _ in range(150):
            d = c[:, None] - c[None]
            dist = np.sqrt((d ** 2).sum(-1)) + np.eye(n)
            over = np.maximum(r[:, None] + r[None] - dist, 0)
            np.fill_diagonal(over, 0)
            wl = np.maximum(r[:, None] - c, 0)
            wr = np.maximum(c + r[:, None] - 1, 0)
            if (over ** 2).sum() + (wl ** 2).sum() + (wr ** 2).sum() < 1e-12:
                ok = True
                break
            grad = -2 * ((over / dist)[:, :, None] * d).sum(1) + 2 * (wr - wl)
            c = c - 0.1 * grad
        if ok:
            c_ok, r_ok = c, r
        else:
            g /= 2
            c_ok = np.clip(c_ok + rng.normal(0, 1e-3, c_ok.shape), 0, 1)
    return c_ok, r_ok


def solve(n, seed, time_limit):
    rng, t0 = np.random.default_rng(seed), time.process_time()
    c, r = _compress(n, rng)
    c, r = _solve_once(n, rng, np.concatenate([c.ravel(), r]))
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
