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
    # Area-based initialization: Partition into a grid and assign radii proportional to cell area
    grid_dim = int(np.ceil(np.sqrt(n)))
    cell_w = 1.0 / grid_dim
    cell_h = 1.0 / grid_dim
    
    # Create grid coordinates
    grid_coords = np.indices((grid_dim, grid_dim)).reshape(2, -1).T
    # Adjust to be the centers of the cells
    pts = (grid_coords + 0.5 * cell_w) * (grid_coords + 0.5 * cell_h) # This is slightly wrong, let's use cell_w
    # Corrected: centers of cells
    pts = (grid_coords / grid_dim + 0.5) * (1.0 / grid_dim)
    # Use a large enough grid to accommodate n points
    grid_size = int(np.ceil(np.sqrt(n)))
    indices = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    pts = indices[:n] 
    pts = (pts / grid_size).reshape(n, 2)
    # Shift to centers
    pts = (pts + 0.5) / grid_size
    
    # Clamp to ensure they are inside [0.1, 0.9]
    pts = np.clip(pts, INIT_LOW, 1 - INIT_LOW)
    
    # Initialize radii: area of cell is (1/grid_size)^2.
    # Sum of area pi*r^2 = Area of square = 1. So r = 
    # sqrt(1 / (n * pi)) is a rough average.
    # For individual cells, r = sqrt(cell_area / pi) = (1/grid_size) / sqrt(pi)
    r_init = (1.0 / grid_size) / np.sqrt(np.pi)
    # Scale slightly to allow for overlap in the initial configuration
    r_init *= 0.8
    
    x0 = np.concatenate([pts.flatten(), np.full(n, r_init)])
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
