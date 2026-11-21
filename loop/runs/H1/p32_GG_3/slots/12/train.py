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
MAX_ITER = 1000
INIT_RADIUS = 0.02


def _solve_once(n, rng):
    c = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2))
    r = rng.uniform(INIT_RADIUS, 0.1, n)
    
    lr = 0.01
    for i in range(MAX_ITER):
        # Gradient-like updates
        # 1. Overlap repulsion
        diff = c[:, None, :] - c[None, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(np.maximum(dist_sq, 1e-9))
        
        overlap = (r[:, None] + r[None, :]) - dist
        # Filter only positive overlaps
        mask = overlap > 0
        
        # Gradient of overlap
        grad_c = np.zeros_like(c)
        if np.any(mask):
            # For every pair (i, j) that overlaps, push them apart
            # This is a simplified force-directed approach
            for i_idx in range(n):
                for j_idx in range(n):
                    if i_idx == j_idx: continue
                    if mask[i_idx, j_idx]:
                        direction = diff[i_idx, j_idx]
                        grad_c[i_idx] += direction * 0.01
                        grad_c[j_idx] -= direction * 0.01
        
        # 2. Boundary attraction (keep inside [0, 1]
        # 3. Radius growth
        r += 0.001
        c -= lr * grad_c
        c = np.clip(c, r[:, None] + INIT_LOW, 1 - r[:, None] - INIT_LOW)
        
        if i % 100 == 0:
            lr *= 0.99

    return c, r

def _repair(c, r):
    """Shrink radii until the packing is valid, so that the harness never sees an overlap."""
    n = len(r)
    r = np.minimum(r, np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1 - c[:, 0], 1 - c[:, 1])))
    r = np.maximum(r, 0)
    for _ in range(100):
        d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
        over = r[:, None] + r[None] - d
        np.fill_diagonal(over, -1)
        max_over = np.max(over)
        if max_over <= 0:
            break
        i, j = np.unravel_index(np.argmax(over), over.shape)
        s = d[i, j] / (r[i] + r[j] + 1e-9)
        r[i], r[j] = r[i] * s, r[j] * s
    return r


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
