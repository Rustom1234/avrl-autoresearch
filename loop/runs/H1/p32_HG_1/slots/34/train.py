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
    if rng.random() < 0.33:
        # Hexagonal lattice start
        # Approximate grid to cover unit square with ~32-40 points
        # Rows with alternating offsets
        rows = 6
        cols = 7
        pts = []
        for r in range(rows):
            for c in range(cols):
                x = c * (1.0 / (cols - 1))
                y = r * (1.0 / (rows - 1))
                if r % 2 == 1:
                    x += 0.5 / (cols - 1)
                pts.append([x, y])
        pts = np.array(pts)
        # Jitter and filter
        pts += rng.normal(0, 0.02, pts.shape)
        # Keep only n points that are most central
        dist_sq = np.sum((pts - 0.5)**2, axis=1)
        indices = np.argsort(dist_sq)[:n]
        pts = pts[indices]
        x0 = np.concatenate([pts.flatten(), np.full(n, INIT_RADIUS)])
    else:
        # Uniform random start
        x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])

    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    from scipy.optimize import LBFGSB
    
    # c = x[:2*n], r = x[2*n:]
    # Boundary constraints: x_i in [0,1], r_i in [0, 0.5]
    # Let's use LBFGSB and handle boundaries by clipping or just bound the variables
    # x_i in [0, 1], r_i in [0, 0.5]
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    
    # Initial values for Lagrange multipliers (lambdas)
    lambdas = np.zeros(n * (n - 1) // 2)
    mu = 10.0
    
    curr_x = x0
    for _ in range(6):
        def obj_func(v):
            c_v = v[:2 * n].reshape(n, 2)
            r_v = v[2 * n:]
            
            # Non-overlap g_ij = r_i + r_j - d_ij <= 0
            # In inequality terms, we want g_ij <= 0, so we penalize max(0, g_ij)^2
            # However, SLSQP handles g_ij <= 0 directly. 
            # For Augmented Lagrangian, we use: min -sum(r) + (mu/2) * sum(max(0, g_ij + lambda_ij/mu)^2)
            
            dist = np.sqrt(((c_v[:, None] - c_v[None]) ** 2).sum(-1))
            # Get upper triangle indices
            iu = np.triu_indices(n, 1)
            # g_ij = r_i + r_j - dist_ij
            g = r_v[iu[0]] + r_v[iu[1]] - dist[iu]
            
            penalty = 0
            for k, (i, j) in enumerate(zip(*iu)):
                val = g[k] + lambdas[k] / mu
                if val > 0: penalty += 0.5 * mu * (val ** 2)
            
            return -r_v.sum() + penalty
        
        res = minimize(obj_func, curr_x, method='L-BFGS-B', bounds=bounds, options={'maxiter': MAX_ITER // 7})
        curr_x = res.x
        
        # Update lambdas
        c_v = curr_x[:2 * n].reshape(n, 2)
        r_v = curr_x[2 * n:]
        dist = np.sqrt(((c_v[:, None] - c_v[None]) ** 2).sum(-1))
        iu = np.triu_indices(n, 1)
        g = r_v[iu[0]] + r_v[iu[1]] - dist[iu]
        lambdas = np.maximum(0, lambdas + mu * g)
        mu *= 3

    # Final SLSQP polish
    def cons(v):
        c_v, r_v = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c_v[iu[0]] - c_v[iu[1]]) ** 2).sum(1)) - r_v[iu[0]] - r_v[iu[1]]
        return np.concatenate([d, c_v[:, 0] - r_v, c_v[:, 1] - r_v, 1 - c_v[:, 0] - r_v, 1 - c_v[:, 1] - r_v, r_v])
    
    res_final = minimize(lambda v: -v[2 * n:].sum(), curr_x, method="SLSQP", 
                           constraints={"type": "ineq", "fun": cons}, 
                           options={"maxiter": 100})
    return res_final.x[:2 * n].reshape(n, 2), res_final.x[2 * n:]


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
