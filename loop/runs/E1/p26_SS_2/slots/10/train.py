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

    P = len(iu[0])
    kk, aa = np.arange(P), np.arange(n)

    def jac(v):
        c = v[:2 * n].reshape(n, 2)
        diff = c[iu[0]] - c[iu[1]]
        u = diff / np.maximum(np.hypot(diff[:, 0], diff[:, 1]), 1e-12)[:, None]
        J = np.zeros((P + 5 * n, 3 * n))
        J[kk, 2 * iu[0]] = u[:, 0]
        J[kk, 2 * iu[0] + 1] = u[:, 1]
        J[kk, 2 * iu[1]] = -u[:, 0]
        J[kk, 2 * iu[1] + 1] = -u[:, 1]
        J[kk, 2 * n + iu[0]] = -1
        J[kk, 2 * n + iu[1]] = -1
        J[P + aa, 2 * aa] = 1
        J[P + n + aa, 2 * aa + 1] = 1
        J[P + 2 * n + aa, 2 * aa] = -1
        J[P + 3 * n + aa, 2 * aa + 1] = -1
        J[P + aa, 2 * n + aa] = -1
        J[P + n + aa, 2 * n + aa] = -1
        J[P + 2 * n + aa, 2 * n + aa] = -1
        J[P + 3 * n + aa, 2 * n + aa] = -1
        J[P + 4 * n + aa, 2 * n + aa] = 1
        return J

    g = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", jac=lambda v: g,
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
        if best is not None and time.process_time() - t0 > 0.3 * time_limit:
            c, r = _solve_once(n, rng, best, rng.choice([0.005, 0.01, 0.02, 0.05]))
        else:
            c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
