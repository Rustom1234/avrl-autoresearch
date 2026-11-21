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
    # Use a jittered grid for better coverage than pure random
    grid_size = int(np.ceil(np.sqrt(n)))
    x = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, x)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1).reshape(-1, 2)
    pts = pts[:n] # Take first n points
    
    # Jitter the points and keep them within bounds
    jitter = rng.uniform(-0.05, 0.05, pts.shape)
    pts = np.clip(pts + jitter, INIT_LOW, 1 - INIT_LOW)
    
    x0 = np.concatenate([pts.flatten(), np.full(n, INIT_RADIUS)])
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
    for _ in range(100):
        d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
        over = r[:, None] + r[None] - d
        np.fill_diagonal(over, -1)
        i, j = np.unravel_index(np.argmax(over), over.shape)
        if over[i, j] <= 0:
            break
        s = d[i, j] / (r[i] + r[j])
        r[i], r[j] = r[i] * s, r[j] * s
    
    # Final check for validity
    d_final = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
    over_final = r[:, None] + r[None] - d_final
    np.fill_diagonal(over_final, -1)
    if np.any(over_final > 0):
        return None
    return np.maximum(r - SHRINK, 0)


def solve(n, seed, time_limit):
    rng = np.random.default_rng(seed)
    t0 = time.process_time()
    best = None
    
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r_repaired = _repair(c, r)
        
        if r_repaired is not None:
            if best is None or r_repaired.sum() > best[1].sum():
                best = (c, r_repaired)
        else:
            # Restart with a new seed if repair fails
            seed = (seed + 1) % (2**31 - 1)
            rng = np.random.default_rng(seed)
            
    return best
