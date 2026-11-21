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
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def obj_and_grad(v, mu):
        c = v[:2 * n].reshape(n, 2)
        r = v[2 * n:]
        grad = np.zeros_like(v)
        grad[2 * n:] = -1.0
        
        d_ij = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1))
        s_ij = d_ij - (r[iu[0]] + r[iu[1]])
        s_wall = np.concatenate([c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r])
        
        eps = 1e-12
        for k in range(len(iu[0])):
            idx_i, idx_j = iu[0][k], iu[1][k]
            val = mu / (s_ij[k] + eps)
            grad[2 * n + idx_i] += val
            grad[2 * n + idx_j] += val
            
        for i in range(n):
            # c_x - r_x
            grad[i*2] += mu / (c[i,0] - r[i] + eps)
            grad[2*n + i] += mu / (c[i,0] - r[i] + eps)
            # c_y - r_y
            grad[i*2+1] += mu / (c[i,1] - r[i] + eps)
            grad[2*n + i] += mu / (c[i,1] - r[i] + eps)
            # 1 - c_x - r_x
            grad[i*2] += -mu / (1 - c[i,0] - r[i] + eps)
            grad[2*n + i] += -mu / (1 - c[i,0] - r[i] + eps)
            # 1 - c_y - r_y
            grad[i*2+1] += -mu / (1 - c[i,1] - r[i] + eps)
            grad[2*n + i] += -mu / (1 - c[i,1] - r[i] + eps)
            
        return -np.sum(r) + mu * (np.sum(np.log(s_ij + eps)) + np.sum(np.log(s_wall + eps))), grad

    v = x0.copy()
    mu = 1e-1
    for _ in range(10):
        res = minimize(lambda v_in: obj_and_grad(v_in, mu)[0], v, method='L-BFGS-B', jac=lambda v_in: obj_and_grad(v_in, mu)[1], 
                        options={'maxiter': 100, 'ftol': 1e-4})
        v = res.x
        mu *= 0.1
    return v[:2 * n].reshape(n, 2), v[2 * n:]


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
