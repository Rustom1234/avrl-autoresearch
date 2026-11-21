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


def _solve_once(n, rng):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    I, J = iu
    m = len(I)
    k = np.arange(m)
    a = np.arange(n)
    W = np.zeros((4 * n, 3 * n))
    W[a, 2 * a] = 1
    W[a, 2 * n + a] = -1
    W[n + a, 2 * a + 1] = 1
    W[n + a, 2 * n + a] = -1
    W[2 * n + a, 2 * a] = -1
    W[2 * n + a, 2 * n + a] = -1
    W[3 * n + a, 2 * a + 1] = -1
    W[3 * n + a, 2 * n + a] = -1
    b = np.concatenate([np.zeros(2 * n), np.ones(2 * n)])

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = c[I] - c[J]
        g = (d ** 2).sum(1) - (r[I] + r[J]) ** 2
        return np.concatenate([g, W @ v + b])

    def jac(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = c[I] - c[J]
        s = r[I] + r[J]
        P = np.zeros((m, 3 * n))
        P[k, 2 * I] = 2 * d[:, 0]
        P[k, 2 * I + 1] = 2 * d[:, 1]
        P[k, 2 * J] = -2 * d[:, 0]
        P[k, 2 * J + 1] = -2 * d[:, 1]
        P[k, 2 * n + I] = -2 * s
        P[k, 2 * n + J] = -2 * s
        return np.vstack([P, W])

    grad = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    res = minimize(lambda v: -v[2 * n:].sum(), x0, jac=lambda v: grad, method="SLSQP", bounds=bounds,
                   constraints={"type": "ineq", "fun": cons, "jac": jac}, options={"maxiter": MAX_ITER})
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
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
