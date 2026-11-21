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
    radii = np.zeros(n)
    # Greedy largest-hole construction
    first_pos = rng.uniform(0, 1, 2)
    centers[0] = first_pos
    # Clearance to boundary
    c0 = np.minimum(first_pos, 1 - first_pos)
    radii[0] = np.min(c0)

    for i in range(1, n):
        candidates = rng.uniform(0, 1, (2000, 2))
        # Distances to existing centers
        d_centers = np.linalg.norm(candidates - centers[:i], axis=1)
        min_d_centers = np.min(d_centers, axis=1)
        # Distance to boundaries
        dist_to_bound = np.minimum(np.minimum(candidates, 1 - candidates), axis=1)
        min_dist_to_bound = np.min(dist_to_bound, axis=1)
        
        # Clearances available at candidates
        clearances = np.minimum(min_d_centers - radii[:i], min_dist_to_bound)
        
        # Add small noise to keep diversity
        scores = clearances + rng.uniform(0, 0.005, size=2000)
        best_idx = np.argmax(scores)
        
        centers[i] = candidates[best_idx]
        radii[i] = max(0, scores[best_idx])

    x0 = np.concatenate([centers.flatten(), radii])
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
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
