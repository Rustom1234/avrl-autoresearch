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
    # Lubachevsky-Stillinger style growth
    c = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2))
    r = rng.uniform(0.01, INIT_RADIUS, n)
    
    growth_factor = 1.005
    for _ in range(200):
        r *= growth_factor
        # Push circles out of bounds
        for i in range(n):
            if c[i, 0] - r[i] < 0: c[i, 0] += (r[i] - c[i, 0]) * 0.5
            if c[i, 0] + r[i] > 1: c[i, 0] -= (c[i, 0] + r[i] - 1) * 0.5
            if c[i, 1] - r[i] < 0: c[i, 1] += (r[i] - c[i, 1]) * 0.5
            if c[i, 1] + r[i] > 1: c[i, 1] -= (c[i, 1] + r[i] - 1) * 0.5
            
        # Resolve overlaps
        for i in range(n):
            for j in range(i + 1, n):
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-9:
                    push = (diff / dist) * (min_dist - dist) * 0.5
                    c[i] += push
                    c[j] -= push
        growth_factor *= 0.995
        if growth_factor < 1.0001: break

    # SLSQP Polish
    x0 = np.concatenate([c.flatten(), r])
    iu = np.triu_indices(n, 1)
    def cons(v):
        cv = v[:2 * n].reshape(n, 2)
        rv = v[2 * n:]
        d = np.sqrt(((cv[iu[0]] - cv[iu[1]])**2).sum(1)) - rv[iu[0]] - rv[iu[1]]
        return np.concatenate([d, cv[:, 0] - rv, cv[:, 1] - rv, 1 - cv[:, 0] - rv, 1 - cv[:, 1] - rv, rv])

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
    return best
