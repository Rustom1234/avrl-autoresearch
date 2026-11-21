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
    # Variables: centres (2n) and log-radii s (n), so r = exp(s) stays positive.
    if x0 is None:
        x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, np.log(INIT_RADIUS))])
    iu, ju = np.triu_indices(n, 1)
    m, k = len(iu), np.arange(n)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), np.exp(v[2 * n:])
        d = np.sqrt(((c[iu] - c[ju]) ** 2).sum(1)) - r[iu] - r[ju]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r])

    def cons_jac(v):
        c, r = v[:2 * n].reshape(n, 2), np.exp(v[2 * n:])
        diff = c[iu] - c[ju]
        u = diff / np.maximum(np.sqrt((diff ** 2).sum(1)), 1e-12)[:, None]
        jac = np.zeros((m + 4 * n, 3 * n))
        p = np.arange(m)
        jac[p, 2 * iu] = u[:, 0]
        jac[p, 2 * iu + 1] = u[:, 1]
        jac[p, 2 * ju] = -u[:, 0]
        jac[p, 2 * ju + 1] = -u[:, 1]
        jac[p, 2 * n + iu] = -r[iu]
        jac[p, 2 * n + ju] = -r[ju]
        q = m + k
        jac[q, 2 * k] = 1
        jac[q, 2 * n + k] = -r
        q = m + n + k
        jac[q, 2 * k + 1] = 1
        jac[q, 2 * n + k] = -r
        q = m + 2 * n + k
        jac[q, 2 * k] = -1
        jac[q, 2 * n + k] = -r
        q = m + 3 * n + k
        jac[q, 2 * k + 1] = -1
        jac[q, 2 * n + k] = -r
        return jac

    def obj(v):
        return -np.exp(v[2 * n:]).sum()

    def obj_jac(v):
        g = np.zeros(3 * n)
        g[2 * n:] = -np.exp(v[2 * n:])
        return g

    res = minimize(obj, x0, jac=obj_jac, method="SLSQP",
                   constraints={"type": "ineq", "fun": cons, "jac": cons_jac}, options={"maxiter": MAX_ITER})
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
    return _solve_once(len(r), rng, np.concatenate([c.ravel(), np.log(np.maximum(r, 1e-9))]))


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
