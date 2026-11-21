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
RESTART_FRACTION = 0.3


def _optimise(n, x0, best_sum=None):
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    def run(x, maxiter, ftol):
        return minimize(lambda v: -v[2 * n:].sum(), x, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                        options={"maxiter": maxiter, "ftol": ftol}).x

    x = run(x0, 60, 1e-6)   # loose first stage
    if best_sum is None or np.maximum(x[2 * n:], 0).sum() >= best_sum - 0.01:
        x = run(x, MAX_ITER, 1e-9)   # tight continuation only for promising starts
    return x[:2 * n].reshape(n, 2), x[2 * n:]


def _solve_once(n, rng, best_sum=None):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    return _optimise(n, x0, best_sum)


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
    budget = TIME_FRACTION * time_limit
    while time.process_time() - t0 < RESTART_FRACTION * budget or best is None:   # short random-restart phase
        c, r = _solve_once(n, rng, None if best is None else best[1].sum())
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    c, r = best
    while time.process_time() - t0 < budget:   # remove-and-reinsert the smallest circle
        c2, r2 = c.copy(), r.copy()
        i = np.argmin(r2)
        c2[i] = rng.uniform(0, 1, 2)
        r2[i] = 0
        c2, r2 = _optimise(n, np.concatenate([c2.ravel(), r2]), r.sum())
        c2 = np.clip(c2, 0, 1)
        r2 = _repair(c2, r2)
        if r2.sum() > r.sum():
            c, r = c2, r2
    return c, r
