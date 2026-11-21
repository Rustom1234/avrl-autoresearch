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

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    def jac(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        m, k = len(iu[0]), np.arange(n)
        diff = c[iu[0]] - c[iu[1]]
        dist = np.maximum(np.sqrt((diff ** 2).sum(1)), 1e-12)
        u = diff / dist[:, None]
        J = np.zeros((m + 5 * n, 3 * n))
        rows = np.arange(m)
        J[rows, 2 * iu[0]] = u[:, 0]
        J[rows, 2 * iu[0] + 1] = u[:, 1]
        J[rows, 2 * iu[1]] = -u[:, 0]
        J[rows, 2 * iu[1] + 1] = -u[:, 1]
        J[rows, 2 * n + iu[0]] = -1
        J[rows, 2 * n + iu[1]] = -1
        o = m
        J[o + k, 2 * k] = 1
        J[o + k, 2 * n + k] = -1
        o += n
        J[o + k, 2 * k + 1] = 1
        J[o + k, 2 * n + k] = -1
        o += n
        J[o + k, 2 * k] = -1
        J[o + k, 2 * n + k] = -1
        o += n
        J[o + k, 2 * k + 1] = -1
        J[o + k, 2 * n + k] = -1
        o += n
        J[o + k, 2 * n + k] = 1
        return J

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP",
                   jac=lambda v: np.concatenate([np.zeros(2 * n), -np.ones(n)]),
                   constraints={"type": "ineq", "fun": cons, "jac": jac},
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
