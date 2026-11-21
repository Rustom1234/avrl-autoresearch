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
POOL = 16
STEP = 40
INIT_RADIUS = 0.05
SHRINK = 1e-9


def _start(n, rng):
    base = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2))
    return np.concatenate([base.ravel(), np.full(n, INIT_RADIUS)])


def _slsqp(n, x0, iters):
    """Run (or continue) SLSQP from the state x0 for at most iters iterations; returns the full state vector."""
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": iters})
    return res.x


def _evaluate(n, x):
    c = np.clip(x[:2 * n].reshape(n, 2), 0, 1)
    return c, _repair(c, x[2 * n:])


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
    limit = TIME_FRACTION * time_limit
    while time.process_time() - t0 < limit:   # successive halving over pools until the time is used
        pool, step = [_start(n, rng) for _ in range(POOL)], STEP
        while len(pool) > 1 and time.process_time() - t0 < limit:
            ranked = []
            for x in pool:
                if time.process_time() - t0 >= limit:
                    break
                x = _slsqp(n, x, step)
                c, r = _evaluate(n, x)
                ranked.append((r.sum(), x))
                if best is None or r.sum() > best[1].sum():
                    best = (c, r)
            if not ranked:
                break
            ranked.sort(key=lambda t: -t[0])
            pool = [x for _, x in ranked[:max(1, len(ranked) // 2)]]
            step *= 2
    return best
