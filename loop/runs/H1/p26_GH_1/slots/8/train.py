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


def _anneal(n, x, rng, steps=2000):
    """Short simulated annealing on (centres, radii); returns the lowest-energy state found."""
    c, r = x[:2 * n].reshape(n, 2).copy(), x[2 * n:].copy()

    def energy(c, r):
        d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
        ov = np.maximum(r[:, None] + r[None] - d, 0)
        np.fill_diagonal(ov, 0)
        wall = np.maximum(r[:, None] - c, 0) + np.maximum(r[:, None] + c - 1, 0)
        return -r.sum() + 50 * (ov.sum() + wall.sum())

    E = energy(c, r)
    best = (E, c.copy(), r.copy())
    T0 = 0.05
    for k in range(steps):
        f = 1 - k / steps
        T = T0 * f + 1e-6
        i = rng.integers(n)
        c2, r2 = c.copy(), r.copy()
        c2[i] += rng.normal(0, 0.05 * f + 0.005, 2)
        r2[i] = max(0.0, r2[i] + rng.normal(0, 0.01))
        E2 = energy(c2, r2)
        if E2 < E or rng.random() < np.exp(-(E2 - E) / T):
            c, r, E = c2, r2, E2
            if E < best[0]:
                best = (E, c.copy(), r.copy())
    return np.concatenate([best[1].ravel(), best[2]])


def _solve_once(n, rng):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    x0 = _anneal(n, x0, rng)
    iu = np.triu_indices(n, 1)
    s0 = rng.uniform(0.7, 1.4)   # elliptical stretch used only in the initial search

    def make_cons(s):
        def cons(v):
            c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
            d = np.sqrt((c[iu[0], 0] - c[iu[1], 0]) ** 2 + (s * (c[iu[0], 1] - c[iu[1], 1])) ** 2) - r[iu[0]] - r[iu[1]]
            return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])
        return cons

    def obj(v):
        return -v[2 * n:].sum()

    res = minimize(obj, x0, method="SLSQP", constraints={"type": "ineq", "fun": make_cons(s0)},
                   options={"maxiter": MAX_ITER})
    # round to true circles: re-solve with the real circle constraints, starting from the stretched solution
    res = minimize(obj, res.x, method="SLSQP", constraints={"type": "ineq", "fun": make_cons(1.0)},
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
