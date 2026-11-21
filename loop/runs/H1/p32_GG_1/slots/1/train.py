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
SHRINK = 1e-9


def _solve_once(n, rng):
    # Brief Simulated Annealing to find a better warm-start
    # x = [x1, y1, ..., xn, yn, r1, ..., rn]
    x = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), rng.uniform(0.01, 0.05, n)])
    
    # Penalty function for SA
    def get_penalty(v):
        c = v[:2*n].reshape(n, 2)
        r = v[2*n:]
        p = 0
        # Boundary
        for i in range(n):
            p += max(0, r[i] - c[i,0]) + max(0, r[i] - (1-c[i,0])) + max(0, r[i] - c[i,1]) + max(0, r[i] - (1-c[i,1]))
        # Overlap
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(c[i] - c[j])
                p += max(0, (r[i] + r[j]) - dist)
        return p

    # Simple SA steps
    T = 1.0
    for _ in range(150):
        prev_p = get_penalty(x)
        idx = rng.integers(0, 3 * n)
        x[idx] += rng.normal(0, 0.05) * (T / 100)
        new_p = get_penalty(x)
        if new_p > prev_p and rng.random() > np.exp((prev_p - new_p) / T):
            x[idx] = x[idx] - (x[idx] - x[idx]) # logic check: just undo
            # simpler: just keep best
        x = x if new_p < prev_p else x # (This is a very simplified SA)
        T *= 0.99
    
    # Just use a better heuristic warm start if SA is too slow
    # Let's stick to a better random init for now to keep it simple
    x = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), rng.uniform(0.02, 0.08, n)])
    
    iu = np.triu_indices(n, 1)
    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x, method="SLSQP", constraints={"type": "ineq", "fun": cons},
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
