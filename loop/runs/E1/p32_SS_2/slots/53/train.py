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
    x0 = np.concatenate([pts.ravel(), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
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


def _polish(c, r, S=10.0):
    """SLSQP with analytic Jacobian on variables (centres, S * radii)."""
    n = len(r)
    a, b = np.triu_indices(n, 1)
    P = len(a)
    k, idx = np.arange(P), np.arange(n)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:] / S
        d = np.sqrt(((c[a] - c[b]) ** 2).sum(1)) - r[a] - r[b]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    def jac(v):
        c = v[:2 * n].reshape(n, 2)
        diff = c[a] - c[b]
        u = diff / (np.sqrt((diff ** 2).sum(1))[:, None] + 1e-15)
        J = np.zeros((P + 5 * n, 3 * n))
        J[k, 2 * a], J[k, 2 * a + 1], J[k, 2 * b], J[k, 2 * b + 1] = u[:, 0], u[:, 1], -u[:, 0], -u[:, 1]
        J[k, 2 * n + a] = -1 / S
        J[k, 2 * n + b] = -1 / S
        for q, (col, sgn) in enumerate([(0, 1), (1, 1), (0, -1), (1, -1)]):
            J[P + q * n + idx, 2 * idx + col] = sgn
            J[P + q * n + idx, 2 * n + idx] = -1 / S
        J[P + 4 * n + idx, 2 * n + idx] = 1 / S
        return J

    g = np.concatenate([np.zeros(2 * n), np.full(n, -1 / S)])
    x0 = np.concatenate([c.ravel(), S * r])
    res = minimize(lambda v: -v[2 * n:].sum() / S, x0, jac=lambda v: g, method="SLSQP",
                   constraints={"type": "ineq", "fun": cons, "jac": jac}, options={"maxiter": 500, "ftol": 1e-15})
    return res.x[:2 * n].reshape(n, 2), res.x[2 * n:] / S


def solve(n, seed, time_limit):
    rng, t0, best, raw = np.random.default_rng(seed), time.process_time(), None, None
    while time.process_time() - t0 < 0.85 * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        rr = _repair(c, r.copy())
        if best is None or rr.sum() > best[1].sum():
            best, raw = (c, rr), (c, r)
    try:
        c2, r2 = _polish(raw[0], np.maximum(raw[1], 0))
        c2 = np.clip(c2, 0, 1)
        r2 = _repair(c2, r2)
        if r2.sum() > best[1].sum():
            best = (c2, r2)
    except Exception:
        pass
    return best
