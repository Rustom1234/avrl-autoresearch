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
    centers = np.zeros((n, 2))
    for i in range(n):
        success = False
        for _ in range(200):
            cand = rng.uniform(0, 1, 2)
            if i == 0:
                centers[i] = cand
                success = True
                break
            dist = np.linalg.norm(centers[:i] - cand, axis=1)
            if np.all(dist > 0.12):
                centers[i] = cand
                success = True
                break
        if not success:
            centers[i] = rng.uniform(0, 1, 2)

    x0 = np.concatenate([centers.flatten(), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)
    initial_c = centers
    initial_r = np.full(n, INIT_RADIUS)
    dist_matrix = np.sqrt(((initial_c[:, None] - initial_c[None, :]) ** 2).sum(-1))
    active_indices = [i for i in range(len(iu)) if dist_matrix[iu[0][i], iu[1][i]] < (initial_r[iu[0][i]] + initial_r[iu[1][i]] + 0.15)]
    active_iu = (iu[0][active_indices], iu[1][active_indices])

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[active_iu[0]] - c[active_iu[1]]) ** 2).sum(1)) - r[active_iu[0]] - r[active_iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": MAX_ITER})
    
    final_c = res.x[:2 * n].reshape(n, 2)
    final_r = res.x[2 * n:]
    # Final check to ensure no overlap was missed due to sparsity
    all_dist = np.sqrt(((final_c[:, None] - final_c[None, :]) ** 2).sum(-1))
    over = final_r[:, None] + final_r[None, :] - all_dist
    np.fill_diagonal(over, -1)
    if np.any(over > 0):
        # If sparse constraints failed, do one quick repair step
        res_repair = _repair(final_c, final_r)
        if len(res_repair) == 2:
            final_c, final_r = res_repair
    
    return final_c, final_r


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
    return np.maximum(r - SHRINK, 0), r


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
