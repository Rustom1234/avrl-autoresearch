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


def _hex_init(n, rng):
    """Perturbed hexagonal-grid start: random n-subset of a staggered grid, scaled into the square, jittered."""
    m = int(np.ceil(np.sqrt(n))) + 1
    g = np.array([(col + 0.5 * (row % 2), row * np.sqrt(3) / 2) for row in range(m) for col in range(m)])
    c = g[rng.choice(len(g), n, replace=False)]
    c = (c - c.min(0)) / (c.max(0) - c.min(0))
    c = 0.1 + 0.8 * c + rng.normal(0, 0.01, c.shape)
    r = np.full(n, INIT_RADIUS) * (1 + 0.1 * rng.standard_normal(n))
    return np.clip(c, 0, 1), np.abs(r)


def _solve_once(n, rng, x0=None):
    if x0 is None:
        c0, r0 = _hex_init(n, rng)
        x0 = np.concatenate([c0.ravel(), r0])
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


def _jump(c, r, rng):
    """Move one random circle to a random spot and re-optimise from that layout."""
    c, r = c.copy(), r.copy()
    i = rng.integers(len(r))
    c[i] = rng.uniform(INIT_LOW, 1 - INIT_LOW, 2)
    r[i] = INIT_RADIUS
    return _solve_once(len(r), rng, np.concatenate([c.ravel(), r]))


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        if best is not None and time.process_time() - t0 > 0.5 * TIME_FRACTION * time_limit:
            c, r = _jump(best[0], best[1], rng)   # basin-hopping jump from the best packing
        else:
            c, r = _solve_once(n, rng)            # random restart
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
