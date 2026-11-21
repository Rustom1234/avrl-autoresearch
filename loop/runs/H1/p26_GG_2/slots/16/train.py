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
    for _ in range(100):
        d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
        over = r[:, None] + r[None] - d
        np.fill_diagonal(over, -1)
        if np.any(over > 0):
            i, j = np.unravel_index(np.argmax(over), over.shape)
            if over[i, j] <= 0:
                break
            # Shrink radii such that the smaller one takes more of the reduction
            # s = d / (r1 + r2). To favor smaller r, we use a weighted approach
            # where the larger r shrinks less. If r_i < r_j, r_i shrinks more.
            # A simple way: shrink both by a factor s_avg, but bias it.
            # Let's use a more aggressive shrink on the smaller one:
            s_base = d[i, j] / (r[i] + r[j])
            # If r[i] is much smaller than r[j], make r[i] shrink more.
            # We'll use a simple power: r_new = r * (s_base ^ (weight))
            # But for simplicity, just shrink the smaller one more by 1.2x weight
            # to get it out of the way faster.
            if r[i] < r[j]:
                # r[i] is smaller, shrink it more
                # r[i]*s1 + r[j]*s2 = d[i,j]
                # Set s2 = s_base * 0.9, solve for s1
                s2 = s_base * 0.9
                s1 = (d[i, j] - r[j] * s2) / r[i]
                r[i] *= s1
                r[j] *= s2
            else:
                s1 = s_base * 0.9
                s2 = (d[i, j] - r[i] * s1) / r[j]
                r[i] *= s1
                r[j] *= s2
            # Ensure radii stay positive
            r[i] = max(r[i], 1e-12)
            r[j] = max(r[j], 1e-12)
        else:
            break
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
