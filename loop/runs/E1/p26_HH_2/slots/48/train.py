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


def _solve_once(n, rng, start=None):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)]) if start is None else start
    iu = np.triu_indices(n, 1)

    def cons(v, cap):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r, cap - r])

    x = x0
    for cap in list(np.arange(0.05, 0.4 + 1e-9, 0.02)) + [1.0]:
        res = minimize(lambda v: -v[2 * n:].sum(), x, method="SLSQP",
                       constraints={"type": "ineq", "fun": lambda v, cap=cap: cons(v, cap)},
                       options={"maxiter": MAX_ITER})
        x = res.x
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


def _dihedral_start(c, r, rng):
    """Random one of the 8 symmetries of the square applied to the centres, plus small noise, as a start vector."""
    n = len(r)
    x, y = c[:, 0], c[:, 1]
    maps = [(x, y), (y, x), (1 - x, y), (x, 1 - y), (1 - x, 1 - y), (1 - y, x), (y, 1 - x), (1 - y, 1 - x)]
    u, v = maps[rng.integers(8)]
    c2 = np.clip(np.column_stack([u, v]) + rng.normal(0, 0.01, (n, 2)), 0, 1)
    return np.concatenate([c2.ravel(), r])


def solve(n, seed, time_limit):
    rng, t0, best, it = np.random.default_rng(seed), time.process_time(), None, 0
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        start = _dihedral_start(best[0], best[1], rng) if best is not None and it % 4 == 3 else None
        it += 1
        c, r = _solve_once(n, rng, start)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
