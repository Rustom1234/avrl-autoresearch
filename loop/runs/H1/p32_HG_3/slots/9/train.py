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
    # Pre-relaxation: Gradient descent to spread centers
    c = rng.random((n, 2)) * (1 - 2 * INIT_LOW) + INIT_LOW
    r_initial = np.full(n, INIT_RADIUS)
    
    lr = 0.01
    for _ in range(100):
        # Repulsion from walls
        grad = np.zeros_like(c)
        grad[..., 0] += c[:, 0] * (1 - c[:, 0])
        grad[..., 1] += c[:, 1] * (1 - c[:, 1])
        # Simple pairwise repulsion
        for i in range(n):
            diff = c - c[i]
            dist = np.linalg.norm(diff, axis=1)
            dist[i] = 1e-6
            force = diff / (dist**3)
            grad[i] -= np.sum(force, axis=1) * 0.1
        c -= lr * grad
        c = np.clip(c, INIT_LOW, 1 - INIT_LOW)

    # Initialize radii based on nearest neighbors
    r_init = np.full(n, INIT_RADIUS)
    for i in range(n):
        dists = np.linalg.norm(c - c[i], axis=1)
        nearest = np.argmin(dists[1:]) + 1
        r_init[i] = min(r_init[i], dists[nearest] / 2.0)

    x0 = np.concatenate([c.flatten(), r_init])
    iu = np.triu_indices(n, 1)

    SCALE = 10.0
    def cons(v):
        c_v, r_scaled = v[:2 * n].reshape(n, 2), v[2 * n:]
        r = r_scaled / SCALE
        d = np.sqrt(((c_v[iu[0]] - c_v[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c_v[:, 0] - r, c_v[:, 1] - r, 1 - c_v[:, 0] - r, 1 - c_v[:, 1] - r, r])

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
