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
RESTART_FRACTION = 0.7
INIT_LOW = 0.1
MAX_ITER = 100
POLISH_ITER = 600
INIT_RADIUS = 0.05
SHRINK = 1e-9


def _solve_once(n, rng, start=None, maxiter=MAX_ITER):
    if start is None:
        R = int(rng.integers(5, 8))
        m = int(np.ceil(n / R))
        pts = np.array([((j + 0.5 + 0.5 * (i % 2)) / (m + 0.5), (i + 0.5) / R) for i in range(R) for j in range(m)])
        pts = pts[rng.permutation(len(pts))[:n]] + rng.normal(0, 0.02, (n, 2))
        pts = np.clip(pts, 0.03, 0.97)
        x0 = np.concatenate([pts.ravel(), np.full(n, INIT_RADIUS)])
    else:
        x0 = np.concatenate([start[0].ravel(), start[1]])
    iu = np.triu_indices(n, 1)
    P = len(iu[0])
    k = np.arange(P)
    a = np.arange(n)

    def jac(v):
        c = v[:2 * n].reshape(n, 2)
        diff = c[iu[0]] - c[iu[1]]
        u = diff / (np.sqrt((diff ** 2).sum(1)) + 1e-12)[:, None]
        J = np.zeros((P + 5 * n, 3 * n))
        J[k, 2 * iu[0]] = u[:, 0]
        J[k, 2 * iu[0] + 1] = u[:, 1]
        J[k, 2 * iu[1]] = -u[:, 0]
        J[k, 2 * iu[1] + 1] = -u[:, 1]
        J[k, 2 * n + iu[0]] = -1
        J[k, 2 * n + iu[1]] = -1
        for q, (col, sgn) in enumerate([(0, 1), (1, 1), (0, -1), (1, -1)]):
            J[P + q * n + a, 2 * a + col] = sgn
            J[P + q * n + a, 2 * n + a] = -1
        J[P + 4 * n + a, 2 * n + a] = 1
        return J

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP",
                   constraints={"type": "ineq", "fun": cons, "jac": jac},
                   jac=None, options={"maxiter": maxiter})
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
    while best is None or time.process_time() - t0 < RESTART_FRACTION * time_limit:
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    c, r = _solve_once(n, rng, start=best, maxiter=POLISH_ITER)
    c = np.clip(c, 0, 1)
    r = _repair(c, r)
    if r.sum() > best[1].sum():
        best = (c, r)
    return best
