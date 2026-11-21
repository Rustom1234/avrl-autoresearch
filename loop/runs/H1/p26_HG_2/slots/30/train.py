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
    centers = np.zeros((n, 2))
    for i in range(n):
        success = False
        for _ in range(200):
            cand = rng.uniform(0, 1, 2)
            if i == 0:
                centers[i] = cand
                success = True
                break
            dist = np.linalg.norm(centers[:i] - cand, axis=1)
            if np.all(dist > 0.12):
                centers[i] = cand
                success = True
                break
        if not success:
            centers[i] = rng.uniform(0, 1, 2)

    iu = np.triu_indices(n, 1)

    # Stage 1: Optimize for equal radii 's'
    x0_eq = np.concatenate([centers.flatten(), np.array([INIT_RADIUS])])
    def cons_eq(v):
        c = v[:2 * n].reshape(n, 2)
        s = v[2 * n]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - 2 * s
        return np.concatenate([d, c[:, 0] - s, c[:, 1] - s, 1 - c[:, 0] - s, 1 - c[:, 1] - s, s])

    res_eq = minimize(lambda v: -v[2 * n], x0_eq, method="SLSQP", constraints={"type": "ineq", "fun": cons_eq},
                       options={"maxiter": MAX_ITER})
    
    # Stage 2: Use result as warm start for mixed radii
    c_warm = res_eq.x[:2 * n].reshape(n, 2)
    r_warm = np.full(n, res_eq.x[2 * n])
    x0 = np.concatenate([c_warm.flatten(), r_warm])

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


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
