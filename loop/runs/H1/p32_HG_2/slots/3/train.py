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
INIT_LOW = 0.05
MAX_ITER = 200
INIT_RADIUS = 0.01
SHRINK = 1e-9


def _ls_growth(n, rng):
    c = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2))
    r = np.full(n, INIT_RADIUS)
    for _ in range(200):
        # Growth step
        dist_to_walls = np.minimum(c, 1 - c).max(axis=1)
        # Calculate neighbor gaps
        diff = c[:, None, :] - c[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # r_i + r_j <= dist_ij  =>  growth is limited by min(wall_dist, min_neighbor_gap)
        # gap = dist_ij - (r_i + r_j)
        gaps = dists - (r[:, None] + r[None, :])
        np.fill_diagonal(gaps, np.inf)
        # Growth factor: how much can we increase r based on nearest neighbor gap?
        max_growth = np.minimum(dist_to_walls, np.min(gaps, axis=1) / 2.0)
        dr = np.minimum(max_growth - r, 0.05)
        dr[dr < 0] = 0
        r += dr
        
        # Relaxation step (pushing overlaps apart)
        over = (r[:, None] + r[None, :]) - dists
        np.fill_diagonal(over, -1)
        if np.any(over > 0):
            i, j = np.where(over > 0)[0, 1]
            # Only handle first pair for simplicity and stability
            i, j = i[0], j[0]
            move_vec = (c[j] - c[i]) / (np.linalg.norm(c[j] - c[i]) + 1e-9)
            move_dist = over[i, j] / 2.0
            c[i] -= move_vec * move_dist
            c[j] += move_vec * move_dist
            c = np.clip(c, 0, 1)
            
        if np.all(dr < 1e-6):
            break
    return c, r


def _solve_once(n, rng):
    # Use LS growth as a starting point for SLSQP
    c, r = _ls_growth(n, rng)
    x0 = np.concatenate([c.flatten(), r])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c_flat, r_flat = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c_flat[iu[0]] - c_flat[iu[1]]) ** 2).sum(1)) - r_flat[iu[0]] - r_flat[iu[1]]
        return np.concatenate([d, c_flat[:, 0] - r_flat, c_flat[:, 1] - r_flat, 
                                1 - c_flat[:, 0] - r_flat, 1 - c_flat[:, 1] - r_flat, r_flat])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", 
                     constraints={"type": "ineq", "fun": cons},
                     options={"maxiter": MAX_ITER, "ftol": 1e-9})
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
        idx = np.argmax(over)
        i, j = np.unravel_index(idx, over.shape)
        if over[i, j] <= 0:
            break
        s = d[i, j] / (r[i] + r[j] + 1e-12)
        r[i], r[j] = r[i] * s, r[j] * s
    return np.maximum(r - SHRINK, 0)


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
