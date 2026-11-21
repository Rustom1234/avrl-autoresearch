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
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)

        while time.process_time() - t0 < TIME_FRACTION * time_limit:
            idx = rng.randint(0, n)
            dists = np.linalg.norm(c - c[idx], axis=1)
            near_idx = np.argsort(dists)[:4]
            
            new_c = c.copy()
            new_r = r.copy()
            
            # Destroy and reinsert
            new_c[near_idx] = rng.uniform(0, 1, (4, 2))
            new_r[near_idx] = 0.02
            
            x0_local = np.concatenate([new_c.flatten(), new_r])
            iu = np.triu_indices(n, 1)
            
            def cons_local(v):
                cv, rv = v[:2 * n].reshape(n, 2), v[2 * n:]
                d = np.sqrt(((cv[iu[0]] - cv[iu[1]]) ** 2).sum(1)) - rv[iu[0]] - rv[iu[1]]
                return np.concatenate([d, cv[:, 0] - rv, cv[:, 1] - rv, 1 - cv[:, 0] - rv, 1 - cv[:, 1] - rv, rv])
            
            res_local = minimize(lambda v: -v[2 * n:].sum(), x0_local, method="SLSQP", 
                                  constraints={"type": "ineq", "fun": cons_local}, 
                                  options={"maxiter": 100})
            
            curr_c = res_local.x[:2 * n].reshape(n, 2)
            curr_r = res_local.x[2 * n:]
            curr_c = np.clip(curr_c, 0, 1)
            curr_r = _repair(curr_c, curr_r)
            
            if best is None or curr_r.sum() > best[1].sum():
                best = (curr_c, curr_r)
    return best
