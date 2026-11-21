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


def _solve_once(n, rng, x0=None):
    if x0 is None:
        # Default: Uniform random
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
    
    def get_hex_init(n, rng):
        # Jittered hexagonal lattice
        x = np.sqrt(3) * np.arange(n) % (np.sqrt(3) * (n**0.5))
        y = np.arange(n) % (np.sqrt(3) * (n**0.5))
        # Simplified: just a small offset grid
        c = np.column_stack([np.linspace(0, 1, int(np.sqrt(n))), np.linspace(0, 1, int(np.sqrt(n)))])._repeat(n, 1).reshape(n, 2)
        # Actually, let's just use a slightly different uniform to keep it simple and robust
        return np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])

    count = 0
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        # Cycle through 3 strategies
        strategy = count % 3
        if strategy == 0:
            c, r = _solve_once(n, rng)
        elif strategy == 1:
            # Perturbed best
            if best is not None:
                c_best = best[0]
                r_best = best[1]
                x0 = np.concatenate([c_best + rng.normal(0, 0.01, (n, 2)), r_best * (1 + rng.normal(0, 0.05, n))])
                c, r = _solve_once(n, rng, x0)
            else:
                c, r = _solve_once(n, rng)
        else:
            # Lattice-ish
            c, r = _solve_once(n, rng, get_hex_init(n, rng))
            
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
        count += 1
    return best
