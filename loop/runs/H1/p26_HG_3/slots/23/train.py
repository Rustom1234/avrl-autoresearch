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
    # Seed some circles in a ring along the boundary
    k = rng.integers(6, 9)
    x0 = np.zeros((n, 2))
    r = np.full(n, INIT_RADIUS)
    
    # Boundary ring initialization
    for i in range(k):
        angle = (2 * np.pi * i / k) + rng.uniform(0, 2 * np.pi / k)
        # Distance from center to make the ring just inside the [0, 1] square
        # and inset by a small margin (0.05)
        dist = 0.45
        x0[i, 0] = 0.5 + dist * np.cos(angle)
        x0[i, 1] = 0.5 + dist * np.sin(angle)
        # Adjust radius to ensure they stay within bounds
        r[i] = min(0.05, 1 - max(abs(x0[i, 0] - 0.5), abs(x0[i, 1] - 0.5))) * 0.8
    
    # Fill the rest randomly
    remaining = n - k
    if remaining > 0:
        x0[k:] = rng.uniform(INIT_LOW, 1 - INIT_LOW, (remaining, 2))
        r[k:] = INIT_RADIUS
        
    # Ensure x0 is a (n, 2) array
    x0 = x0.reshape(n, 2)
    
    # Create the flat array for the solver
    c_flat = np.concatenate([x0[:, 0], x0[:, 1]])
    x0_flat = np.concatenate([c_flat, r])
    
    iu = np.triu_indices(n, 1)

    def cons(v):
        c_vec = v[:2 * n].reshape(n, 2)
        r_vec = v[2 * n:]
        d = np.sqrt(((c_vec[iu[0]] - c_vec[iu[1]]) ** 2).sum(1)) - r_vec[iu[0]] - r_vec[iu[1]]
        return np.concatenate([d, c_vec[:, 0] - r_vec, c_vec[:, 1] - r_vec, 1 - c_vec[:, 0] - r_vec, 1 - c_vec[:, 1] - r_vec, r_vec])

    res = minimize(lambda v: -v[2 * n:].sum(), x0_flat, method="SLSQP", constraints={"type": "ineq", "fun": cons},
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
