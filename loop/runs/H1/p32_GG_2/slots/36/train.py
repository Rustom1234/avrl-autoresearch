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
    # Greedy "Maximal Area" initialization
    centers = np.zeros((n, 2))
    # Use a grid-based search for the 'most spacious' point to find initial centers
    grid_res = 15
    gx = np.linspace(0, 1, grid_res)
    gy = np.linspace(0, 1, grid_res)
    grid_pts = np.array([np.broadcast_to([x, y], (1, 2)).ravel() for x in gx for y in gy]).reshape(-1, 2)
    
    placed_indices = []
    for i in range(n):
        # Calculate 'spaciousness' for each grid point
        # (distance to nearest placed circle boundary - radius)
        # For simplicity, we use distance to nearest circle center
        if i == 0:
            best_pt = grid_pts[rng.choice(len(grid_pts))]
        else:
            # Distance to already placed circles
            dist_to_placed = np.min(np.linalg.norm(grid_pts - centers[placed_indices], axis=1), axis=1)
            # Distance to boundary
            dist_to_bound = np.minimum(grid_pts[:, 0], 1 - grid_pts[:, 0])
            dist_to_bound = np.minimum(dist_to_bound, np.minimum(grid_pts[:, 1], 1 - grid_pts[:, 1]))
            
            # Score: distance to nearest obstacle (combined)
            score = np.minimum(dist_to_placed, dist_to_bound)
            best_pt = grid_pts[np.argmax(score)]
        
        centers[i] = best_pt
        placed_indices.append(i)

    x0 = np.concatenate([centers.flatten(), np.full(n, INIT_RADIUS)])
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
