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
    I, J = iu
    P = len(I)
    D0 = 2 * INIT_RADIUS * 1.6

    def energy(x):
        c = x.reshape(n, 2)
        diff = c[I] - c[J]
        d = np.sqrt((diff ** 2).sum(1)) + 1e-12
        ov = np.maximum(0, D0 - d)
        g = np.zeros((n, 2))
        coef = (-2 * ov / d)[:, None] * diff
        np.add.at(g, I, coef)
        np.add.at(g, J, -coef)
        return (ov ** 2).sum(), g.ravel()

    pts = minimize(energy, pts.ravel(), jac=True, method="L-BFGS-B", bounds=[(0.03, 0.97)] * (2 * n),
                   options={"maxiter": 30}).x.reshape(n, 2)
    x0 = np.concatenate([pts.ravel(), np.full(n, INIT_RADIUS)])

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    def cons_jac(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        Jm = np.zeros((P + 5 * n, 3 * n))
        diff = c[I] - c[J]
        d = np.sqrt((diff ** 2).sum(1)) + 1e-12
        u = diff / d[:, None]
        rows = np.arange(P)
        Jm[rows, 2 * I] = u[:, 0]
        Jm[rows, 2 * I + 1] = u[:, 1]
        Jm[rows, 2 * J] = -u[:, 0]
        Jm[rows, 2 * J + 1] = -u[:, 1]
        Jm[rows, 2 * n + I] = -1
        Jm[rows, 2 * n + J] = -1
        k = np.arange(n)
        b = P
        Jm[b + k, 2 * k] = 1
        Jm[b + k, 2 * n + k] = -1
        Jm[b + n + k, 2 * k + 1] = 1
        Jm[b + n + k, 2 * n + k] = -1
        Jm[b + 2 * n + k, 2 * k] = -1
        Jm[b + 2 * n + k, 2 * n + k] = -1
        Jm[b + 3 * n + k, 2 * k + 1] = -1
        Jm[b + 3 * n + k, 2 * n + k] = -1
        Jm[b + 4 * n + k, 2 * n + k] = 1
        return Jm

    gobj = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    res = minimize(lambda v: -v[2 * n:].sum(), x0, jac=lambda v: gobj, method="SLSQP",
                   constraints={"type": "ineq", "fun": cons, "jac": cons_jac}, options={"maxiter": MAX_ITER})
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
