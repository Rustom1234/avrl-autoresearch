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


PENALTY_MUS = (10, 100, 1000, 1e4)


def _presolve(n, iu, v):
    """L-BFGS-B on the penalised objective -sum(r) + mu * overlap^2 terms, for increasing mu."""
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n

    def fg(v, mu):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        diff = c[iu[0]] - c[iu[1]]
        d = np.sqrt((diff ** 2).sum(1)) + 1e-12
        g = np.maximum(r[iu[0]] + r[iu[1]] - d, 0)
        b1 = np.maximum(r - c[:, 0], 0)
        b2 = np.maximum(r - c[:, 1], 0)
        b3 = np.maximum(c[:, 0] + r - 1, 0)
        b4 = np.maximum(c[:, 1] + r - 1, 0)
        f = -r.sum() + mu * ((g ** 2).sum() + (b1 ** 2 + b2 ** 2 + b3 ** 2 + b4 ** 2).sum())
        gc = np.zeros((n, 2))
        gc[:, 0] = 2 * (b3 - b1)
        gc[:, 1] = 2 * (b4 - b2)
        gr = 2 * (b1 + b2 + b3 + b4)
        pair = 2 * g[:, None] * diff / d[:, None]
        np.add.at(gc, iu[0], -pair)
        np.add.at(gc, iu[1], pair)
        gr += np.bincount(iu[0], 2 * g, n) + np.bincount(iu[1], 2 * g, n)
        return f, np.concatenate([mu * gc.ravel(), -1 + mu * gr])

    v = np.asarray(v, dtype=float)
    for mu in PENALTY_MUS:
        v = minimize(fg, v, args=(mu,), jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200}).x
    return v


def _solve_once(n, rng, init=None):
    base = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2)) if init is None else np.clip(init + rng.normal(0, 0.03, (n, 2)), 0, 1)
    x0 = np.concatenate([base.ravel(), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)
    x0 = _presolve(n, iu, x0)

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
        guided = best is not None and time.process_time() - t0 > 0.5 * TIME_FRACTION * time_limit and rng.random() < 0.5
        c, r = _solve_once(n, rng, best[0] if guided else None)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
