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


def _inflate(n, rng):
    """Optimise centres for equal radii s on a smooth overlap penalty, with s growing in 6 steps."""
    def f(v, s):
        c = v.reshape(n, 2)
        diff = c[:, None] - c[None]
        d = np.sqrt((diff ** 2).sum(-1)) + np.eye(n)
        ov = np.maximum(2 * s - d, 0)
        np.fill_diagonal(ov, 0)
        lo, hi = np.maximum(s - c, 0), np.maximum(c + s - 1, 0)
        val = 0.5 * (ov ** 2).sum() + (lo ** 2).sum() + (hi ** 2).sum()
        g = -2 * ((ov / d)[:, :, None] * diff).sum(1) - 2 * lo + 2 * hi
        return val, g.ravel()

    v = rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n)
    for s in np.linspace(0.06, 0.11, 6):
        v = minimize(f, v, args=(s,), jac=True, method="L-BFGS-B", options={"maxiter": 100}).x
    c = np.clip(v.reshape(n, 2), 0, 1)
    dd = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1)) + 10 * np.eye(n)
    wall = np.minimum(c.min(1), (1 - c).min(1))
    r = np.maximum(np.minimum(wall, 0.5 * dd.min(1)), 0)
    return np.concatenate([c.ravel(), 0.9 * r])


def _solve_once(n, rng, start=None, sd=0.0):
    if start is None:
        x0 = _inflate(n, rng)
    else:
        x0 = np.concatenate([(start[0] + rng.normal(0, sd, start[0].shape)).ravel(), start[1] * 0.5])
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
        if best is not None and time.process_time() - t0 > 0.3 * time_limit:
            c, r = _solve_once(n, rng, best, rng.choice([0.005, 0.01, 0.02, 0.05]))
        else:
            c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
