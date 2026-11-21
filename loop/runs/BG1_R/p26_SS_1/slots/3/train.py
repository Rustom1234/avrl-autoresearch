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

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    I, J = iu
    m = len(I)
    k, a = np.arange(m), np.arange(n)
    g = np.zeros(3 * n)
    g[2 * n:] = -1

    def cjac(v):
        c = v[:2 * n].reshape(n, 2)
        diff = c[I] - c[J]
        u = diff / np.maximum(np.sqrt((diff ** 2).sum(1)), 1e-12)[:, None]
        M = np.zeros((m + 5 * n, 3 * n))
        M[k, 2 * I], M[k, 2 * I + 1], M[k, 2 * J], M[k, 2 * J + 1] = u[:, 0], u[:, 1], -u[:, 0], -u[:, 1]
        M[k, 2 * n + I] = -1
        M[k, 2 * n + J] = -1
        for q, (col, sgn) in enumerate([(0, 1), (1, 1), (0, -1), (1, -1)]):
            M[m + q * n + a, 2 * a + col] = sgn
            M[m + q * n + a, 2 * n + a] = -1
        M[m + 4 * n + a, 2 * n + a] = 1
        return M

    res = minimize(lambda v: -v[2 * n:].sum(), x0, jac=lambda v: g, method="SLSQP",
                   constraints={"type": "ineq", "fun": cons, "jac": cjac}, options={"maxiter": MAX_ITER})
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
