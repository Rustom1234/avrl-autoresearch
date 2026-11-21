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
    R = int(rng.integers(5, 8))
    m = int(np.ceil(n / R))
    pts = np.array([((j + 0.5 + 0.5 * (i % 2)) / (m + 0.5), (i + 0.5) / R) for i in range(R) for j in range(m)])
    pts = pts[rng.permutation(len(pts))[:n]] + rng.normal(0, 0.02, (n, 2))
    pts = np.clip(pts, 0.03, 0.97)
    iu = np.triu_indices(n, 1)
    i0, j0 = iu
    P = len(i0)
    rr = INIT_RADIUS

    def pen(p):
        c = p.reshape(n, 2)
        diff = c[i0] - c[j0]
        dist = np.sqrt((diff ** 2).sum(1)) + 1e-12
        ov = np.maximum(0, 2 * rr - dist)
        lo = np.maximum(0, rr - c)
        hi = np.maximum(0, c - (1 - rr))
        val = (ov ** 2).sum() + (lo ** 2).sum() + (hi ** 2).sum()
        g = -2 * lo + 2 * hi
        gp = (-2 * ov / dist)[:, None] * diff
        np.add.at(g, i0, gp)
        np.add.at(g, j0, -gp)
        return val, g.ravel()

    pts = minimize(pen, pts.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 30}).x.reshape(n, 2)
    pts = np.clip(pts, 0.03, 0.97)
    x0 = np.concatenate([pts.ravel(), np.full(n, INIT_RADIUS)])

    rows = np.arange(P)
    a = np.arange(n)
    Jt = np.zeros((P + 5 * n, 3 * n))
    Jt[rows, 2 * n + i0] = -1
    Jt[rows, 2 * n + j0] = -1
    Jt[P + a, 2 * a] = 1
    Jt[P + a, 2 * n + a] = -1
    Jt[P + n + a, 2 * a + 1] = 1
    Jt[P + n + a, 2 * n + a] = -1
    Jt[P + 2 * n + a, 2 * a] = -1
    Jt[P + 2 * n + a, 2 * n + a] = -1
    Jt[P + 3 * n + a, 2 * a + 1] = -1
    Jt[P + 3 * n + a, 2 * n + a] = -1
    Jt[P + 4 * n + a, 2 * n + a] = 1

    def jac(v):
        c = v[:2 * n].reshape(n, 2)
        diff = c[i0] - c[j0]
        u = diff / (np.sqrt((diff ** 2).sum(1)) + 1e-12)[:, None]
        J = Jt.copy()
        J[rows, 2 * i0] = u[:, 0]
        J[rows, 2 * i0 + 1] = u[:, 1]
        J[rows, 2 * j0] = -u[:, 0]
        J[rows, 2 * j0 + 1] = -u[:, 1]
        return J

    def grad(v):
        g = np.zeros(3 * n)
        g[2 * n:] = -1
        return g

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, jac=grad, method="SLSQP",
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
