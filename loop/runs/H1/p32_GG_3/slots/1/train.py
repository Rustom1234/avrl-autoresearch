"""train.py: packs n circles into the unit square so that the sum of their radii is as large as possible.

The harness calls solve(n, seed, time_limit) and checks the returned packing exactly.
Returns (centers, radii): centers is an (n, 2) array, radii an (n,) array. Every circle must lie inside
[0, 1] x [0, 1] and no two circles may overlap.
"""
import time

import numpy as np
from scipy.optimize import minimize

# Settings
TIME_FRACTION = 0.9
INIT_RADIUS = 0.05
SHRINK = 1e-9


def _solve_once(n, rng):
    # Simple Simulated Annealing
    # Initialize centers and radii
    c = rng.uniform(0.05, 0.95, (n, 2))
    r = rng.uniform(0.01, INIT_RADIUS, n)
    
    current_r = r.copy()
    current_c = c.copy()
    current_sum = current_r.sum()
    
    temp = 0.5
    cooling = 0.9995
    
    for i in range(4000):
        # Perturb
        idx = rng.integers(0, n)
        if rng.random() < 0.5:
            # Move center
            current_c[idx] += rng.normal(0, 0.02, 2)
            current_c[idx] = np.clip(current_c[idx], 0.001, 0.999)
        else:
            # Change radius
            current_r[idx] += rng.normal(0, 0.01)
            current_r[idx] = np.clip(current_r[idx], 0, 0.4)
        
        # Fast check for validity
        valid = True
        for k in range(n):
            # Boundary check
            if (current_c[k, 0] - current_r[k] < 0 or 
                current_c[k, 0] + current_r[k] > 1 or 
                current_c[k, 1] - current_r[k] < 0 or 
                current_c[k, 1] + current_r[k] > 1):
                valid = False
                break
            for m in range(k + 1, n):
                dist = np.linalg.norm(current_c[k] - current_c[m])
                if dist < current_r[k] + current_r[m]:
                    valid = False
                    break
            if not valid: break
        
        if valid:
            new_sum = current_r.sum()
            if new_sum > current_sum or rng.random() < np.exp((new_sum - current_sum) / temp):
                current_sum = new_sum
        
        temp *= cooling
        if i % 500 == 0 and i > 0: temp *= 0.8

    return current_c, current_r


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
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
