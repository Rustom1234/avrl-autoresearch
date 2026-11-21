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
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": MAX_ITER})
    c_pre, r_pre = res.x[:2 * n].reshape(n, 2), res.x[2 * n:]

    # Coordinate descent on radii with fixed centers
    # Maximize sum(r) s.t. r_i + r_j <= dist_ij and r_i <= boundary_dist_i
    r = r_pre.copy()
    for _ in range(40):
        changed = False
        for i in range(n):
            # Constraint: r_i <= min(c_i1, c_i2, 1-c_i1, 1-c_i2)
            bound_r = np.minimum(np.minimum(c_pre[i, 0], c_pre[i, 1]), 
                                   np.minimum(1 - c_pre[i, 0], 1 - c_pre[i, 1]))
            # Constraint: r_i + r_j <= d_ij  =>  r_i <= d_ij - r_j
            dists = np.linalg.norm(c_pre - c_pre[i], axis=1)
            other_rs = r - r[i]
            max_ri = np.minimum(bound_r, np.minimum(dists - other_rs, dists[i] + 0.0000001))
            
            # We want to maximize r_i, but only if it doesn't violate other constraints
            # The coordinate descent step tries to push r_i to its local upper bound
            new_ri = np.min(max_ri)
            if new_ri > r[i] + 1e-9:
                r[i] = new_ri
                changed = True
        if not changed: break

    return c_pre, r


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
