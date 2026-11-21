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
RADIUS_CAPS = (0.1, 0.2, 0.35, 0.5, 1.0)
SHRINK = 1e-9


GD_ITER = 300
GD_LR = 0.01
GD_MOM = 0.9
PEN = 10.0


def _momentum(n, x):
    """Momentum gradient descent on a penalty objective, run before the SLSQP stages."""
    iu = np.triu_indices(n, 1)
    v = np.zeros_like(x)
    for _ in range(GD_ITER):
        c, r = x[:2 * n].reshape(n, 2), x[2 * n:]
        diff = c[iu[0]] - c[iu[1]]
        d = np.sqrt((diff ** 2).sum(1)) + 1e-12
        ov = np.maximum(r[iu[0]] + r[iu[1]] - d, 0)
        u = diff / d[:, None]
        gc, gr = np.zeros((n, 2)), np.zeros(n)
        np.add.at(gc, iu[0], -2 * ov[:, None] * u)
        np.add.at(gc, iu[1], 2 * ov[:, None] * u)
        np.add.at(gr, iu[0], 2 * ov)
        np.add.at(gr, iu[1], 2 * ov)
        lo = np.maximum(r[:, None] - c, 0)
        hi = np.maximum(c + r[:, None] - 1, 0)
        gc += -2 * lo + 2 * hi
        gr += 2 * lo.sum(1) + 2 * hi.sum(1)
        g = np.concatenate([PEN * gc.ravel(), PEN * gr - 1])
        v = GD_MOM * v - GD_LR * g
        x = x + v
    return x


def _solve_once(n, rng):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def cons(v, cap):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r, cap - r])

    x = _momentum(n, x0)
    for cap in RADIUS_CAPS:   # gradually raise the allowed radius, warm-starting each stage
        res = minimize(lambda v: -v[2 * n:].sum(), x, method="SLSQP",
                       constraints={"type": "ineq", "fun": cons, "args": (cap,)}, options={"maxiter": MAX_ITER})
        x = res.x
    return x[:2 * n].reshape(n, 2), x[2 * n:]


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
