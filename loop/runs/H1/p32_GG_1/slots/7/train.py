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
    # Use a jittered grid for better coverage than pure random
    grid_size = int(np.ceil(np.sqrt(n)))
    x = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, x)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1).reshape(-1, 2)
    pts = pts[:n] # Take first n points
    
    # Jitter the points and keep them within bounds
    jitter = rng.uniform(-0.05, 0.05, pts.shape)
    pts = np.clip(pts + jitter, INIT_LOW, 1 - INIT_LOW)
    
    x0 = np.concatenate([pts.flatten(), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": MAX_ITER})
    x = res.x
    c = x[:2 * n].reshape(n, 2)
    r = x[2 * n:]

    # Gradient descent to resolve overlaps
    lr = 0.01
    for _ in range(50):
        grad_c = np.zeros_like(c)
        grad_r = np.zeros_like(r)
        for i in range(n):
            for j in range(i + 1, n):
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < (r[i] + r[j]):
                    # Move centers apart
                    # If dist is 0, move in random direction
                    direction = diff / (dist + 1e-9)
                    grad_c[i] += direction
                    grad_c[j] -= direction
                    # Shrink radii
                    grad_r[i] -= 1
                    grad_r[j] -= 1
        c -= lr * grad_c
        r -= lr * grad_r
        # Keep in bounds
        c = np.clip(c, r[:, None], 1 - r[:, None])

    return c, r


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
