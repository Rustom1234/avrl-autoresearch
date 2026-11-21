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

    P, I, J = len(iu[0]), iu[0], iu[1]
    ar, rows = np.arange(n), np.arange(P)
    J0 = np.zeros((P + 5 * n, 3 * n))
    for b, (col, sg) in enumerate([(0, 1), (1, 1), (0, -1), (1, -1)]):
        J0[P + b * n + ar, 2 * ar + col] = sg
        J0[P + b * n + ar, 2 * n + ar] = -1
    J0[P + 4 * n + ar, 2 * n + ar] = 1
    J0[rows, 2 * n + I] = -1
    J0[rows, 2 * n + J] = -1

    def cons_jac(v):
        c = v[:2 * n].reshape(n, 2)
        diff = c[I] - c[J]
        g = diff / np.maximum(np.sqrt((diff ** 2).sum(1)), 1e-12)[:, None]
        M = J0.copy()
        M[rows, 2 * I], M[rows, 2 * I + 1] = g[:, 0], g[:, 1]
        M[rows, 2 * J], M[rows, 2 * J + 1] = -g[:, 0], -g[:, 1]
        return M

    gobj = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", jac=lambda v: gobj,
                   constraints={"type": "ineq", "fun": cons, "jac": cons_jac},
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
