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
    
    def get_active_constraints(v):
        c = v[:2 * n].reshape(n, 2)
        r = v[2 * n:]
        # Boundary constraints
        c0 = c[:, 0] - r
        c1 = c[:, 1] - r
        b0 = 1 - c[:, 0] - r
        b1 = 1 - c[:, 1] - r
        # Pairwise distance constraints
        diff = c[iu[0]] - c[iu[1]]
        d = np.sqrt((diff**2).sum(1)) - (r[iu[0]] + r[iu[1]])
        # Active set: only consider circles close to each other
        dist_sq = (c[:, None, :].sum(2) - (c[None, :, :]).sum(2))
        # For performance, we simplify: just use all pairs but the vectorization 
        # makes the cost of all pairs manageable for n=32.
        return np.concatenate([d, c0, c1, b0, b1, r]), c, r

    def cons_fun(v):
        vals, _, _ = get_active_constraints(v)
        return vals

    def jac_fun(v):
        vals, c, r = get_active_constraints(v)
        # The objective is -sum(r), so grad is -1 for r indices
        # Gradient of constraints:
        # d/dv (c0) = [1, 0, -1, 0, ...]
        # d/dv (d) = d/dc (sqrt(sum(dx^2))) - d/dr (r1+r2)
        n = len(r)
        jac = np.zeros((len(vals), 2 * n + n))
        
        # Indices for c0, c1, b0, b1, r
        # c0 = c[:,0]-r   -> grad: [1,0, -1,0, -1,0, -1,0, -1,0, -1,0, 0,0,0...]
        # c1 = c[:,1]-r   -> grad: [0,1,  0,-1, -1,0, -1,0, -1,0, -1,0, 0,0,0...]
        # b0 = 1-c[:,0]-r -> grad: [-1,0, 0,0,  1,0,  1,0,  1,0,  1,0, 1,0, 0,0,0...]
        # b1 = 1-c[:,1]-r -> grad: [0,-1, 0,0,  0,1,  0,1,  0,1,  0,1,  1,0, 0,0,0...]
        # r = r           -> grad: [0,0, 0,0,  0,0,  0,0,  0,0,  0,0,  1,0, 1,0, 1,0, 1,0, 1,0]
        
        # This is complex to do fully analytically in a few lines. 
        # SLSQP is robust; let's use the vectorized constraint with a simpler 
        # approach if needed, but vectorizing the sum is the priority.
        return jac

    # Reverting to vectorized constraint without full Jacobian for stability 
    # as SLSQP's internal finite diff is fast enough on small n.
    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", 
                    constraints={"type": "ineq", "fun": cons_fun},
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
