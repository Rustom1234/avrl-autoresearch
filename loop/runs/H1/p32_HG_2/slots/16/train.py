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


def _solve_once(n, rng, x0=None):
    if x0 is None:
        if rng.random() < 0.5:
            # Symmetric start
            num_half = (n + 1) // 2
            c_half = rng.uniform(INIT_LOW, 1 - INIT_LOW, (num_half, 2))
            # Mirror across anti-diagonal: (x,y) -> (y,x)
            c_mirrored = c_half[:, [1, 0]]
            if n % 2 == 1:
                # If n is odd, the middle circle is on the diagonal (x=y)
                # Let's just join them and handle the count
                c_full = np.vstack([c_half, c_mirrored])
                if c_full.shape[0] > n:
                    c_full = c_full[:n]
            else:
                c_full = np.vstack([c_half, c_mirrored])
            # Adjust if we have one extra due to odd/even logic
            if c_full.shape[0] > n:
                c_full = c_full[:n]
            c_full = c_full[:n]
            # Ensure the middle one is on diagonal if odd
            if n % 2 == 1:
                idx = n // 2
                c_full[idx] = np.array([rng.uniform(INIT_LOW, 1-INIT_LOW), rng.uniform(INIT_LOW, 1-INIT_LOW)])
                # To keep it simple and valid, let's just do a clean construction:
                c_half = rng.uniform(INIT_LOW, 1 - INIT_LOW, (num_half, 2))
                c_mirrored = c_half[:, [1, 0]]
                c_full = np.vstack([c_half, c_mirrored])
                if c_full.shape[0] > n:
                    c_full = c_full[:n]
                c_full = c_full[:n]
                if n % 2 == 1:
                    # middle element is at n // 2
                    mid = n // 2
                    c_full[mid] = np.array([rng.uniform(INIT_LOW, 1-INIT_LOW), rng.uniform(INIT_LOW, 1-INIT_LOW)])
            c = c_full
            x0 = np.concatenate([c.ravel(), np.full(n, INIT_RADIUS)])
        else:
            # Unrestricted start
            c = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2))
            x0 = np.concatenate([c.ravel(), np.full(n, INIT_RADIUS)])
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
    hops_since_improvement = 0
    
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        if best is None or hops_since_improvement > 15:
            c, r = _solve_once(n, rng)
            c = np.clip(c, 0, 1)
            r = _repair(c, r)
            hops_since_improvement = 0
        else:
            # Perturb the best packing
            c_curr, r_curr = best
            c_new = c_curr.copy()
            num_to_move = rng.integers(1, 4)
            indices = rng.choice(n, num_to_move, replace=False)
            
            if rng.random() < 0.5:
                c_new[indices] = rng.uniform(INIT_LOW, 1 - INIT_LOW, (num_to_move, 2))
            else:
                c_new += rng.normal(0, 0.02, (n, 2))
                c_new = np.clip(c_new, 0, 1)
            
            x0 = np.concatenate([c_new.ravel(), r_curr])
            c, r = _solve_once(n, rng, x0=x0)
            c = np.clip(c, 0, 1)
            r = _repair(c, r)
            
            if r.sum() > best[1].sum():
                best = (c, r)
                hops_since_improvement = 0
            else:
                hops_since_improvement += 1
        
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
            
    return best
