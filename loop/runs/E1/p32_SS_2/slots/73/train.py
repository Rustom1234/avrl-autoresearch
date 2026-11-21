"""train.py: packs n circles into the unit square so that the sum of their radii is as large as possible.

The harness calls solve(n, seed, time_limit) and checks the returned packing exactly.
Returns (centers, radii): centers is an (n, 2) array, radii an (n,) array. Every circle must lie inside
[0, 1] x [0, 1] and no two circles may overlap.
"""
import time

import numpy as np
from scipy.optimize import minimize

# Settings
TIME_FRACTION = 0.9
INIT_LOW = 0.1
MAX_ITER = 150
INIT_RADIUS = 0.05
SHRINK = 1e-9


def _solve_once(n, rng):
    R = int(rng.integers(5, 8))
    m = int(np.ceil(n / R))
    pts = np.array([((j + 0.5 + 0.5 * (i % 2)) / (m + 0.5), (i + 0.5) / R) for i in range(R) for j in range(m)])
    pts = pts[rng.permutation(len(pts))[:n]] + rng.normal(0, 0.02, (n, 2))
    pts = np.clip(pts, 0.03, 0.97)
    x0 = np.concatenate([pts.ravel(), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    a, b = iu
    P = len(a)
    k = np.arange(P)
    idx = np.arange(n)
    grad = np.zeros(3 * n)
    grad[2 * n:] = -1.0

    def jac(v):
        c = v[:2 * n].reshape(n, 2)
        diff = c[a] - c[b]
        dist = np.maximum(np.sqrt((diff ** 2).sum(1)), 1e-12)
        u = diff / dist[:, None]
        J = np.zeros((P + 5 * n, 3 * n))
        J[k, 2 * a] = u[:, 0]
        J[k, 2 * a + 1] = u[:, 1]
        J[k, 2 * b] = -u[:, 0]
        J[k, 2 * b + 1] = -u[:, 1]
        J[k, 2 * n + a] = -1
        J[k, 2 * n + b] = -1
        for q, (col, sgn) in enumerate([(0, 1), (1, 1), (0, -1), (1, -1)]):
            rows = P + q * n + idx
            J[rows, 2 * idx + col] = sgn
            J[rows, 2 * n + idx] = -1
        J[P + 4 * n + idx, 2 * n + idx] = 1
        return J

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", jac=lambda v: grad,
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
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
