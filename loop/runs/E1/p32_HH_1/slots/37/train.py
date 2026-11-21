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


def _solve_once(n, rng, init=None):
    base = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2)) if init is None else np.clip(init + rng.normal(0, 0.03, (n, 2)), 0, 1)
    x0 = np.concatenate([base.ravel(), np.full(n, INIT_RADIUS)])
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


WEIGHTS = np.geomspace(1, 100, 4)
SWAP_T = 0.005


def _solve_pen(n, rng, w, state=None):
    """SLSQP on sum(r) - w * sum(overlap^2), starting from state (c, r) plus noise, or from random centres."""
    if state is None:
        c, r = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2)), np.full(n, INIT_RADIUS)
    else:
        c, r = np.clip(state[0] + rng.normal(0, 0.03, (n, 2)), 0, 1), state[1].copy()
    x0 = np.concatenate([c.ravel(), r])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    def obj(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1))
        ov = np.maximum(r[iu[0]] + r[iu[1]] - d, 0)
        return -(r.sum() - w * (ov ** 2).sum())

    res = minimize(obj, x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": MAX_ITER})
    return res.x[:2 * n].reshape(n, 2), res.x[2 * n:]


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    cur = [None] * len(WEIGHTS)   # replica states: (c, r, repaired sum)
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        for k, w in enumerate(WEIGHTS):   # one sweep: each replica takes a local solve at its weight
            c, r = _solve_pen(n, rng, w, None if cur[k] is None else cur[k][:2])
            c = np.clip(c, 0, 1)
            r = _repair(c, r)
            cur[k] = (c, r, r.sum())
            if best is None or r.sum() > best[1].sum():
                best = (c, r)
        for k in range(len(WEIGHTS) - 1):   # Metropolis swap of adjacent weights on repaired sums
            a, b = cur[k][2], cur[k + 1][2]
            if rng.random() < np.exp(min(0.0, (b - a) / SWAP_T)):
                cur[k], cur[k + 1] = cur[k + 1], cur[k]
    return best
