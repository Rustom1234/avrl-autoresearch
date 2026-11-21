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
    iu = np.triu_indices(n, 1)
    ia, ib = iu

    def pen(v, mu):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        g = np.zeros(3 * n)
        gc = g[:2 * n].reshape(n, 2)
        gr = g[2 * n:]
        gr -= 1.0
        f = -r.sum()
        diff = c[ia] - c[ib]
        d = np.sqrt((diff ** 2).sum(1)) + 1e-12
        o = np.maximum(r[ia] + r[ib] - d, 0)
        f += mu * (o ** 2).sum()
        w = 2 * mu * o
        np.add.at(gr, ia, w)
        np.add.at(gr, ib, w)
        u = (w / d)[:, None] * diff
        np.add.at(gc, ia, -u)
        np.add.at(gc, ib, u)
        for k in range(2):
            e1 = np.maximum(r - c[:, k], 0)
            e2 = np.maximum(c[:, k] + r - 1, 0)
            f += mu * ((e1 ** 2).sum() + (e2 ** 2).sum())
            gr += 2 * mu * (e1 + e2)
            gc[:, k] += 2 * mu * (e2 - e1)
        return f, g

    bnds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    for mu in (10, 100, 1000, 1e4):
        x0 = minimize(pen, x0, args=(mu,), jac=True, method="L-BFGS-B", bounds=bnds,
                      options={"maxiter": 200}).x

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
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        guided = best is not None and time.process_time() - t0 > 0.5 * TIME_FRACTION * time_limit and rng.random() < 0.5
        c, r = _solve_once(n, rng, best[0] if guided else None)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
