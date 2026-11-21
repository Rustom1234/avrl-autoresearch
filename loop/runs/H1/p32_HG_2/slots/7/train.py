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
        x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
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
            # Perform a remove-and-reinsert move
            c_curr, r_curr = best
            c_new = c_curr.copy()
            r_new = r_curr.copy()
            
            # Pick one of the 3 smallest circles
            idx = rng.choice(np.argsort(r_curr)[:3])
            
            # Candidate points for reinsertion
            candidates = rng.uniform(INIT_LOW, 1 - INIT_LOW, (200, 2))
            # Distance to other circles and boundaries
            # To keep it fast, we use a subset of points or a simple heuristic
            # Here we just sample 200 points and pick the best
            dist_scores = np.zeros((200, 1))
            for i in range(200):
                # Distance to walls
                d_wall = min(candidates[i, 0], 1 - candidates[i, 0], candidates[i, 1], 1 - candidates[i, 1])
                # Distance to other centers
                other_centers = np.delete(c_curr, idx, axis=0)
                d_others = np.min(np.linalg.norm(candidates[i, :1] - other_centers, axis=1), initial=10.0)
                dist_scores[i] = d_wall + d_others
            
            best_cand_idx = np.argmax(dist_scores)
            c_new[idx] = candidates[best_cand_idx]
            
            x0 = np.concatenate([c_new.ravel(), r_new])
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
