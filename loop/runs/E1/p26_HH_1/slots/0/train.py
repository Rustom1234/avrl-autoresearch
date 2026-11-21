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


def _solve_once(n, rng, x0=None):
    if x0 is None:
        x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
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


def _from_larger(n, rng):
    """Optimise n+1 circles, drop the one with the smallest radius, and return the start vector for n."""
    c, r = _solve_once(n + 1, rng)
    k = np.argmin(r)
    return np.concatenate([np.delete(c, k, 0).ravel(), np.delete(r, k)])


def _from_smaller(n, rng):
    """Optimise n-1 circles, then add a circle at the point with the largest gap."""
    c, r = _solve_once(n - 1, rng)
    p = rng.uniform(0, 1, (2000, 2))
    d = np.sqrt(((p[:, None] - c[None]) ** 2).sum(-1)).min(1)
    d = np.minimum(d, np.minimum(p, 1 - p).min(1))
    g = p[np.argmax(d)]
    return np.concatenate([np.vstack([c, g]).ravel(), np.append(r, 0.01)])


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    k = 0
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # restarts cycle: random, n+1 minus one, n-1 plus one
        mode = k % 3
        k += 1
        if mode == 0:
            c, r = _solve_once(n, rng)
        elif mode == 1:
            c, r = _solve_once(n, rng, _from_larger(n, rng))
        else:
            c, r = _solve_once(n, rng, _from_smaller(n, rng))
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
