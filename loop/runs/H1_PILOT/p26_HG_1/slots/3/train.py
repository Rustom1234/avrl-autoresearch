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
    budget = TIME_FRACTION * time_limit
    
    # Initial exploration
    while time.process_time() - t0 < budget * 0.4:
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
            
    # Exploitation with annealed perturbation
    while time.process_time() - t0 < budget:
        if best is None: break
        
        elapsed = (time.process_time() - t0) / budget
        sd = 0.08 * (0.002 / 0.08) ** elapsed
        
        # Create a perturbed starting point
        c_best, r_best = best
        # Use a custom x0 for a local refinement call
        noise = rng.standard_normal(size=(n, 2)) * sd
        x0 = np.concatenate([c_best + noise, r_best.flatten()])
        
        def local_cons(v):
            c_loc, r_loc = v[:2*n].reshape(n, 2), v[2*n:]
            iu = np.triu_indices(n, 1)
            d = np.sqrt(((c_loc[iu[0]] - c_loc[iu[1]]) ** 2).sum(1)) - r_loc[iu[0]] - r_loc[iu[1]]
            return np.concatenate([d, c_loc[:, 0] - r_loc, c_loc[:, 1] - r_loc, 
                                   1 - c_loc[:, 0] - r_loc, 1 - c_loc[:, 1] - r_loc, r_loc])
        
        res = minimize(lambda v: -v[2*n:].sum(), x0, method="SLSQP", 
                        constraints={"type": "ineq", "fun": local_cons}, 
                        options={"maxiter": MAX_ITER})
        
        if res.success:
            c_new, r_new = res.x[:2*n].reshape(n, 2), res.x[2*n:]
            c_new = np.clip(c_new, 0, 1)
            r_new = _repair(c_new, r_new)
            if r_new.sum() > best[1].sum():
                best = (c_new, r_new)
                
    return best
