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
    def _solve_inner(count, rng_local, max_iter):
        x0 = np.concatenate([rng_local.uniform(INIT_LOW, 1 - INIT_LOW, 2 * count), np.full(count, INIT_RADIUS)])
        iu = np.triu_indices(count, 1)
        SCALE = 10.0
        def cons_inner(v):
            c, r_scaled = v[:2 * count].reshape(count, 2), v[2 * count:]
            r = r_scaled / SCALE
            d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
            return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])
        res = minimize(lambda v: -(v[2 * count:] / SCALE).sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons_inner},
                       options={"maxiter": max_iter})
        return res.x[:2 * count].reshape(count, 2), res.x[2 * count:] / SCALE

    if n == 1:
        return np.array([[0.5, 0.5]]), np.array([0.45])

    c_prev, r_prev = _solve_inner(n - 1, rng, MAX_ITER // 2)
    
    best_r, best_c = -1.0, np.array([0.5, 0.5])
    grid_size = 10
    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            tc = np.array([i / grid_size, j / grid_size])
            dist_wall = np.minimum(np.minimum(tc[0], 1 - tc[0]), np.minimum(tc[1], 1 - tc[1]))
            dist_circles = np.linalg.norm(c_prev - tc, axis=1) - r_prev
            r_possible = np.minimum(dist_wall, np.min(dist_circles))
            if r_possible > best_r:
                best_r, best_c = r_possible, tc

    x0 = np.concatenate([c_prev, best_c.reshape(1, 2)], axis=0)
    r_next = np.concatenate([r_prev, [max(best_r, 0.01)]])
    x0 = np.concatenate([x0, r_next], axis=0)
    
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
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
