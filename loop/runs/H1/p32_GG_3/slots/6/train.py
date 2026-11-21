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
    # Force-directed initialization
    c = rng.uniform(0.1, 0.9, (n, 2))
    r = np.full(n, INIT_RADIUS)
    
    # Simple force-directed layout to spread centers
    for _ in range(50):
        forces = np.zeros((n, 2))
        # Repulsion between circles
        for i in range(n):
            diff = c - c[i]
            dist_sq = np.sum(diff**2, axis=1)
            mask = dist_sq > 1e-9
            dist = np.sqrt(dist_sq[mask])
            # Repulsion force magnitude inversely proportional to distance
            f = (diff[mask] / dist[:, np.newaxis]) * (r[mask, np.newaxis] + r[i])
            forces[i] += np.sum(f, axis=0)
            # Repulsion from other circle's radius influence
            forces[mask] -= (diff[mask] / dist[:, np.newaxis]) * (r[mask, np.newaxis] + r[i])
            
        # Wall repulsion
        forces += (c < 0.1).astype(float) * 0.5 + (c > 0.9).astype(float) * 0.5
        c += forces * 0.01
        c = np.clip(c, 0.05, 0.95)
        
    x0 = np.concatenate([c.flatten(), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c_v, r_v = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c_v[iu[0]] - c_v[iu[1]]) ** 2).sum(1)) - r_v[iu[0]] - r_v[iu[1]]
        return np.concatenate([d, c_v[:, 0] - r_v, c_v[:, 1] - r_v, 1 - c_v[:, 0] - r_v, 1 - c_v[:, 1] - r_v, r_v])

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
