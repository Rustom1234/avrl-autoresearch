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
AL_ROUNDS = 4
AL_ITER = 150
INIT_RADIUS = 0.05
SHRINK = 1e-9


def _cons_jac(v, cap, n, iu):
    """Constraints g(v) >= 0 (pairs, walls, r >= 0, r <= cap) and their Jacobian."""
    c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
    i, j = iu
    diff = c[i] - c[j]
    d = np.sqrt((diff ** 2).sum(1))
    p, ar, pr = len(i), np.arange(n), np.arange(len(i))
    g = np.concatenate([d - r[i] - r[j], c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r, cap - r])
    J = np.zeros((p + 6 * n, 3 * n))
    u = diff / np.maximum(d, 1e-12)[:, None]
    J[pr, 2 * i], J[pr, 2 * i + 1] = u[:, 0], u[:, 1]
    J[pr, 2 * j], J[pr, 2 * j + 1] = -u[:, 0], -u[:, 1]
    J[pr, 2 * n + i] = -1
    J[pr, 2 * n + j] = -1
    J[p + ar, 2 * ar] = 1
    J[p + ar, 2 * n + ar] = -1
    J[p + n + ar, 2 * ar + 1] = 1
    J[p + n + ar, 2 * n + ar] = -1
    J[p + 2 * n + ar, 2 * ar] = -1
    J[p + 2 * n + ar, 2 * n + ar] = -1
    J[p + 3 * n + ar, 2 * ar + 1] = -1
    J[p + 3 * n + ar, 2 * n + ar] = -1
    J[p + 4 * n + ar, 2 * n + ar] = 1
    J[p + 5 * n + ar, 2 * n + ar] = -1
    return g, J


def _solve_once(n, rng):
    x = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)
    bounds = [(0, 1)] * (3 * n)
    for cap in list(np.arange(0.05, 0.4 + 1e-9, 0.02)) + [1.0]:
        lam, rho = np.zeros(len(iu[0]) + 6 * n), 10.0
        for _ in range(AL_ROUNDS):
            def fun(v):
                g, J = _cons_jac(v, cap, n, iu)
                t = np.maximum(0, lam / rho - g)
                grad = -rho * (t @ J)
                grad[2 * n:] -= 1
                return -v[2 * n:].sum() + 0.5 * rho * (t @ t) - (lam @ lam) / (2 * rho), grad
            x = minimize(fun, x, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": AL_ITER}).x
            g, _ = _cons_jac(x, cap, n, iu)
            lam = np.maximum(0, lam - rho * g)
            rho *= 3
    return x[:2 * n].reshape(n, 2), x[2 * n:]


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
