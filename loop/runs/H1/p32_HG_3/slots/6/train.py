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

    SCALE = 10.0
    def cons(v):
        c, r_scaled = v[:2 * n].reshape(n, 2), v[2 * n:]
        r = r_scaled / SCALE
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -(v[2 * n:] / SCALE).sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": MAX_ITER})
    return res.x[:2 * n].reshape(n, 2), res.x[2 * n:] / SCALE


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

        if best is not None and (time.process_time() - t0 < 0.9 * time_limit):
            bc, br = best
            idx = np.argmin(br)
            bc_new = np.delete(bc, idx, axis=0)
            br_new = np.delete(br, idx)
            
            # Find farthest point from existing circles and walls
            pts = rng.uniform(0, 1, (2000, 2))
            dists = np.ones((2000, 1))
            for i in range(len(bc_new)):
                d = np.sqrt(np.sum((pts - bc_new[i])**2, axis=1))
                dists = np.minimum(dists, d - br_new[i])
                dists = np.minimum(dists, pts[:,0], axis=1)
                dists = np.minimum(dists, 1 - pts[:,0], axis=1)
                dists = np.minimum(dists, pts[:,1], axis=1)
                dists = np.minimum(dists, 1 - pts[:,1], axis=1)
            
            best_pt = pts[np.argmax(dists)]
            # Try to insert the removed circle
            new_c = np.vstack([bc_new, best_pt])
            new_r = np.append(br_new, br[idx])
            
            c_next = np.clip(new_c, 0, 1)
            r_next = _repair(c_next, new_r)
            if r_next.sum() > best[1].sum():
                best = (c_next, r_next)
                
    return best
