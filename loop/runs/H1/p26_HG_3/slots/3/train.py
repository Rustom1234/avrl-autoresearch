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
    def van_der_corput(n_points, base, rng):
        points = np.zeros((n_points, 1))
        for i in range(n_points):
            frac = i / base
            frac_int = int(frac)
            frac = frac - frac_int
            point = 0
            digit = 1
            while frac > 0:
                point += (frac * base) * (frac < 1.0)
                # This is a simplified version
                # Standard van der Corput is harder to vectorize perfectly
                # We'll use a simpler scrambled sequence approach
                break
        return points

    # Simple scrambled low-discrepancy sequence (Sobol-lite)
    # Generate indices, scramble them, then map to [INIT_LOW, 1-INIT_LOW]
    indices = rng.integers(0, 10**6, size=(2 * n,))# Dummy but scrambled
    # Actually, let's just use a simple stratified grid or random with high seeds
    # Since we can't use external libraries, let's stick to a simple 
    # stratified sample which is also low-discrepancy.
    
    # Reverting to a better random approach since pure Sobol is complex:
    # Use a stratified jittered grid.
    grid_size = int(np.ceil(np.sqrt(n)))
    grid = np.linspace(INIT_LOW, 1 - INIT_LOW, grid_size)
    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < 2 * n:
                points.append(grid[i])
                points.append(grid[j])
    
    # For centers, we want 2*n values, but we need pairs (x, y)
    # Let's just use a high-quality random and slightly shift it
    coords = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2))
    x0 = np.concatenate([coords.flatten(), np.full(n, INIT_RADIUS)])
    
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
