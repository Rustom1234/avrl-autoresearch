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
    """Repair the packing by pushing centers apart and shrinking radii to resolve overlaps."""
    n = len(r)
    c = c.astype(float)
    r = r.astype(float)
    
    # Initial boundary clamping
    r = np.minimum(r, np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1 - c[:, 0], 1 - c[:, 1])))
    r = np.maximum(r, 0)

    for _ in range(100):
        # Calculate distance matrix
        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2) + 1e-9)
        
        # Overlap and boundary violations
        overlap = (r[:, None] + r[None, :]) - d
        np.fill_diagonal(overlap, -1.0)
        
        # Simple gradient descent step
        grad_c = np.zeros_like(c)
        grad_r = np.zeros_like(r)
        
        for i in range(n):
            # Boundary repulsion (push towards center)
            for k in [0, 1]: # x and y
                dist_to_bound = np.minimum(c[i, k], 1 - c[i, k])
                if dist_to_bound < r[i]:
                    grad_c[i, k] += (0.5 - c[i, k]) * (r[i] - dist_to_bound)
            
            # Inter-circle repulsion
            for j in range(n):
                if i == j: continue
                dist = d[i, j]
                if (r[i] + r[j]) > dist:
                    direction = (c[i] - c[j]) / (dist + 1e-9)
                    grad_c[i] += direction * (r[i] + r[j] - dist)
                    grad_r[i] -= (r[i] + r[j] - dist) * 0.5

        c += grad_c * 0.1
        r += grad_r * 0.1
        
        # Project back to valid bounds
        c = np.clip(c, r[None, :] + 1e-7, 1 - r[None, :] - 1e-7)
        r = np.clip(r, 0, 0.5)

        if np.max(overlap) <= 0:
            break

    return c, np.maximum(r - SHRINK, 0)


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
