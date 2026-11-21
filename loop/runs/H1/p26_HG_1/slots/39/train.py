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
    budget_spent = 0
    
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        elapsed = time.process_time() - t0
        # Use 70% of time for random restarts, 30% for incumbent swaps
        if elapsed < 0.7 * (TIME_FRACTION * time_limit) or best is None:
            c, r = _solve_once(n, rng)
            c = np.clip(c, 0, 1)
            r = _repair(c, r)
        else:
            # Radius-permutation swap on the incumbent
            c_best, r_best = best
            indices = np.arange(n)
            idx = rng.choice(indices, 2, replace=False)
            i, j = idx[0], idx[1]
            if abs(r_best[i] - r_best[j]) > 0.02:
                r_swap = r_best.copy()
                r_swap[i], r_swap[j] = r_swap[j], r_swap[i]
                # Re-optimize starting from swapped configuration
                x0 = np.concatenate([c_best.flatten(), r_swap])
                iu = np.triu_indices(n, 1)
                def cons(v):
                    c_v, r_v = v[:2 * n].reshape(n, 2), v[2 * n:]
                    d = np.sqrt(((c_v[iu[0]] - c_v[iu[1]]) ** 2).sum(1)) - r_v[iu[0]] - r_v[iu[1]]
                    return np.concatenate([d, c_v[:, 0] - r_v, c_v[:, 1] - r_v, 1 - c_v[:, 0] - r_v, 1 - c_v[:, 1] - r_v, r_v])
                
                res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons}, 
                               options={"maxiter": MAX_ITER})
                c, r = res.x[:2 * n].reshape(n, 2), res.x[2 * n:]
                c = np.clip(c, 0, 1)
                r = _repair(c, r)
            else:
                c, r = _solve_once(n, rng)
                c = np.clip(c, 0, 1)
                r = _repair(c, r)
                
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
