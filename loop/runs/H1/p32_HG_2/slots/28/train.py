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
            # Jittered square grid
            side = int(np.ceil(np.sqrt(n)))
            grid_x = np.linspace(0, 1, side)
            grid_y = np.linspace(0, 1, side)
            X, Y = np.meshgrid(grid_x, grid_y)
            c = np.stack([X.ravel(), Y.ravel()], axis=-1).reshape(-1, 2)
            c = c[:n]
            c += rng.normal(0, 0.02, (n, 2))
            c = np.clip(c, 0, 1)
        else:
            # Perturbed hexagonal lattice
            rows = rng.integers(int(np.sqrt(n * 0.8)), int(np.sqrt(n * 1.2)) + 1)
            dx = 1.0 / rows
            dy = dx * (2 / np.sqrt(3))
            x_coords = np.linspace(0, 1, rows)
            y_coords = np.linspace(0, 1, int(np.ceil(n / rows)))
            c = np.array([[x + (j % 2) * (dx / 2), y] for y in y_coords for x in x_coords]).reshape(-1, 2)
            c = c[:n]
            c = (c - c.min(axis=0)) / (c.max(axis=0) - c.min(axis=0) + 1e-9)
            c = np.clip(c, 0, 1)
            c += rng.normal(0, 0.02, (n, 2))
            c = np.clip(c, 0, 1)
        
        # Estimate radii based on nearest neighbor distance
        if n > 1:
            # Sample a few random pairs to estimate d_min quickly
            idx = rng.choice(n, min(n, 10), replace=False)
            d_min = np.min(np.linalg.norm(c[idx[1:]] - c[idx[:-1]], axis=1))
        else:
            d_min = 0.1
        r = np.full(n, d_min * 0.4 * rng.random() * (1.0 - 0.8 + 1.0))
        x0 = np.concatenate([c.ravel(), r])
    else:
        c = x0[:2 * n].reshape(n, 2)
        r = x0[2 * n:]
        x0 = x0
    
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
