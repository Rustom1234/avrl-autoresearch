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
    if rng.random() < 0.33:
        # Hexagonal lattice start
        # Approximate grid to cover unit square with ~32-40 points
        # Rows with alternating offsets
        rows = 6
        cols = 7
        pts = []
        for r in range(rows):
            for c in range(cols):
                x = c * (1.0 / (cols - 1))
                y = r * (1.0 / (rows - 1))
                if r % 2 == 1:
                    x += 0.5 / (cols - 1)
                pts.append([x, y])
        pts = np.array(pts)
        # Jitter and filter
        pts += rng.normal(0, 0.02, pts.shape)
        # Keep only n points that are most central
        dist_sq = np.sum((pts - 0.5)**2, axis=1)
        indices = np.argsort(dist_sq)[:n]
        pts = pts[indices]
        x0 = np.concatenate([pts.flatten(), np.full(n, INIT_RADIUS)])
    else:
        # Uniform random start
        c_init = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2))
        
        # Lloyd relaxation (briefly distribute centers)
        samples = rng.uniform(0, 1, (4000, 2))
        for _ in range(15):
            dists = np.linalg.norm(samples[:, None, :] - c_init[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centers = np.array([samples[labels == i].mean(axis=0) for i in range(n) if np.any(labels == i)])
            # Only update indices that actually got points
            for i in range(n):
                idx = np.where(labels == i)[0]
                if len(idx) > 0:
                    c_init[i] = samples[idx].mean(axis=0)
            c_init = np.clip(c_init, INIT_LOW, 1 - INIT_LOW)
            
        x0 = np.concatenate([c_init.flatten(), np.full(n, INIT_RADIUS)])

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
    return best
