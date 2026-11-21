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
    c = np.zeros((n, 2))
    r = np.zeros(n)
    # Greedy initialization
    for i in range(n):
        # Try to place a circle with a decent radius
        # Start with a reasonable radius and shrink if it overlaps
        r[i] = 0.1
        # Sample a few positions
        found = False
        for _ in range(20):
            pos = rng.uniform(0.1, 0.9, 2)
            # Check overlap with previous circles
            dist_sq = np.sum((c[:i] - pos)**2, axis=1)
            min_dist = np.min(np.sqrt(dist_sq))
            current_r_sum = np.sum(r[:i]) # heuristic
            # Simplify: check if it fits
            if min_dist > r[i] + (max(r[:i]) if i > 0 else 0):
                c[i] = pos
                found = True
                break
            r[i] *= 0.8
        if not found:
            c[i] = rng.uniform(0, 1, 2)
            r[i] = 0.01
    
    x0 = np.concatenate([c.flatten(), r])
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
