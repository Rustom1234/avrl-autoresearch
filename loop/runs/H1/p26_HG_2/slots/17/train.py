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
    def halton(seq, base):
        res = np.zeros(len(seq))
        for i, val in enumerate(seq):
            f = 1.0
            r = 0.0
            idx = i + 1
            while idx > 0:
                r += f / (idx & -idx)
                idx = (idx ^ (idx & (idx - 1))) # This logic is for bit manipulation, use standard radical inverse
                # Standard radical inverse for indices
                pass
            # Correct radical inverse:
            r = 0.0
            temp_idx = i + 1
            f = 1.0 / base
            while temp_idx > 0:
                r += f * (temp_idx % base)
                temp_idx //= base
                f /= base
            res[i] = r
        return res

    # Using a simpler radical inverse for brevity
    def radical_inverse(m, base):
        res = np.zeros(m)
        for i in range(m):
            idx = i + 1
            f = 1.0 / base
            r = 0.0
            while idx > 0:
                r += f * (idx % base)
                idx //= base
                f /= base
            res[i] = r
        return res

    # Halton sequence for centers
    seq_x = radical_inverse(n, 2)
    seq_y = radical_inverse(n, 3)
    offset = rng.random(2)
    centers = np.stack([seq_x, seq_y], axis=1) + offset
    # Shrink slightly toward center (0.5, 0.5)
    shrink_factors = rng.uniform(0.9, 1.0, (n, 1))
    centers = 0.5 + 0.5 * (centers - 0.5) * shrink_factors
    centers = np.clip(centers, 0, 1)

    x0 = np.concatenate([centers.flatten(), np.full(n, INIT_RADIUS)])
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
