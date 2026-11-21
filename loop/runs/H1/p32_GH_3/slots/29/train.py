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
    x0 = x0.copy()
    x0[2 * n:] = np.log(np.maximum(x0[2 * n:], 1e-9))   # optimise s = log(r)
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    k = np.arange(m)
    a = np.arange(n)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), np.exp(v[2 * n:])
        d = np.sqrt(((c[ii] - c[jj]) ** 2).sum(1)) - r[ii] - r[jj]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r])

    def cons_jac(v):
        c, r = v[:2 * n].reshape(n, 2), np.exp(v[2 * n:])
        diff = c[ii] - c[jj]
        d = np.maximum(np.sqrt((diff ** 2).sum(1)), 1e-12)
        u = diff / d[:, None]
        J = np.zeros((m + 4 * n, 3 * n))
        J[k, 2 * ii] = u[:, 0]
        J[k, 2 * ii + 1] = u[:, 1]
        J[k, 2 * jj] = -u[:, 0]
        J[k, 2 * jj + 1] = -u[:, 1]
        J[k, 2 * n + ii] = -r[ii]
        J[k, 2 * n + jj] = -r[jj]
        o = m
        J[o + a, 2 * a] = 1
        J[o + a, 2 * n + a] = -r
        J[o + n + a, 2 * a + 1] = 1
        J[o + n + a, 2 * n + a] = -r
        J[o + 2 * n + a, 2 * a] = -1
        J[o + 2 * n + a, 2 * n + a] = -r
        J[o + 3 * n + a, 2 * a + 1] = -1
        J[o + 3 * n + a, 2 * n + a] = -r
        return J

    res = minimize(lambda v: -np.exp(v[2 * n:]).sum(), x0, method="SLSQP",
                   jac=lambda v: np.concatenate([np.zeros(2 * n), -np.exp(v[2 * n:])]),
                   constraints={"type": "ineq", "fun": cons, "jac": cons_jac},
                   options={"maxiter": MAX_ITER})
    return res.x[:2 * n].reshape(n, 2), np.exp(res.x[2 * n:])


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


def _jump(c, r, rng):
    """Move one random circle to a random spot and re-optimise from that layout."""
    c, r = c.copy(), r.copy()
    i = rng.integers(len(r))
    c[i] = rng.uniform(INIT_LOW, 1 - INIT_LOW, 2)
    r[i] = INIT_RADIUS
    return _solve_once(len(r), rng, np.concatenate([c.ravel(), r]))


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        if best is not None and time.process_time() - t0 > 0.5 * TIME_FRACTION * time_limit:
            c, r = _jump(best[0], best[1], rng)   # basin-hopping jump from the best packing
        else:
            c, r = _solve_once(n, rng)            # random restart
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
