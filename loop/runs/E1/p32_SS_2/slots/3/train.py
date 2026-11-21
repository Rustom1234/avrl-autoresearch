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


def _solve_once(n, rng, x0=None, maxiter=MAX_ITER):
    if x0 is None:
        R = int(rng.integers(5, 8))
        m = int(np.ceil(n / R))
        pts = np.array([((j + 0.5 + 0.5 * (i % 2)) / (m + 0.5), (i + 0.5) / R) for i in range(R) for j in range(m)])
        pts = pts[rng.permutation(len(pts))[:n]] + rng.normal(0, 0.02, (n, 2))
        pts = np.clip(pts, 0.03, 0.97)
        x0 = np.concatenate([pts.ravel(), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)
    P = len(iu[0])
    rows = np.arange(P)
    ar = np.arange(n)

    def jac(v):
        c = v[:2 * n].reshape(n, 2)
        diff = c[iu[0]] - c[iu[1]]
        dist = np.maximum(np.sqrt((diff ** 2).sum(1)), 1e-12)
        g = diff / dist[:, None]
        J = np.zeros((P + 5 * n, 3 * n * 2 // 3 * 1 + n))
        for k in (0, 1):
            J[rows, 2 * iu[0] + k] = g[:, k]
            J[rows, 2 * iu[1] + k] = -g[:, k]
        J[rows, 2 * n + iu[0]] = -1
        J[rows, 2 * n + iu[1]] = -1
        for b, (k, s) in enumerate([(0, 1), (1, 1), (0, -1), (1, -1)]):
            J[P + b * n + ar, 2 * ar + k] = s
            J[P + b * n + ar, 2 * n + ar] = -1
        J[P + 4 * n + ar, 2 * n + ar] = 1
        return J

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    gobj = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    res = minimize(lambda v: -v[2 * n:].sum(), x0, jac=lambda v: gobj, method="SLSQP",
                   constraints={"type": "ineq", "fun": cons, "jac": jac}, options={"maxiter": maxiter})
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
    T = TIME_FRACTION * time_limit
    pool = []

    def run(x0=None, maxiter=MAX_ITER):
        c, r = _solve_once(n, rng, x0, maxiter)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        pool.append((r.sum(), c, r))

    while time.process_time() - t0 < 0.5 * T or not pool:    # phase 1: random restarts
        run()
    while time.process_time() - t0 < 0.8 * T:                # phase 2: perturb the top 3
        pool.sort(key=lambda p: -p[0])
        _, c, r = pool[int(rng.integers(min(3, len(pool))))]
        run(np.concatenate([np.clip(c + rng.normal(0, 0.01, c.shape), 0.01, 0.99).ravel(), r * 0.9]))
    pool.sort(key=lambda p: -p[0])                           # phase 3: long polish of the best
    _, c, r = pool[0]
    run(np.concatenate([c.ravel(), r]), 1000)
    _, c, r = max(pool, key=lambda p: p[0])
    return c, r
