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
        c = v[:2 * n].reshape(n, 2)
        r = v[2 * n:]
        d_vec = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1))
        d = d_vec - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    def jac(v):
        c = v[:2 * n].reshape(n, 2)
        r = v[2 * n:]
        d_vec = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1))
        d_vec[d_vec == 0] = 1e-12
        
        num_dist = len(d_vec)
        grad = np.zeros((6 * n, 3 * n))
        
        # Pair distance constraints
        for k in range(num_dist):
            i, j = iu[0][k], iu[1][k]
            grad[k, 2*i : 2*i+2] = (c[i] - c[j]) / d_vec[k]
            grad[k, 2*j : 2*j+2] = (c[j] - c[i]) / d_vec[k]
            grad[k, 2*n + i] = -1.0
            grad[k, 2*n + j] = -1.0
            
        # Wall constraints: c-r >= 0, c-r <= 1, r >= 0
        for i in range(n):
            # c_i - r_i >= 0
            grad[num_dist + i, 2*i : 2*i+2] = [1.0, 1.0]
            grad[num_dist + i, 2*n + i] = -1.0
            # 1 - c_i - r_i >= 0
            grad[num_dist + n + i, 2*i : 2*i+2] = [-1.0, -1.0]
            grad[num_dist + n + i, 2*n + i] = -1.0
            # r_i >= 0
            grad[2 * num_dist + n + i, 2*n + i] = 1.0
            
        return grad

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP",
                    constraints={{"type": "ineq", "fun": cons, "jac": jac}},
                    options={{"maxiter": MAX_ITER, "ftol": 1e-9}})
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
