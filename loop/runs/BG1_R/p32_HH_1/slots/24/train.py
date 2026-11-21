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


def _barrier_fg(v, n, iu, mu):
    """Barrier objective -sum(r) - mu * sum(log slack) and its gradient; infeasible points get a huge value."""
    c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
    i, j = iu
    diff = c[i] - c[j]
    dist = np.maximum(np.sqrt((diff ** 2).sum(1)), 1e-12)
    sp = dist - r[i] - r[j]
    sx0, sx1 = c[:, 0] - r, 1 - c[:, 0] - r
    sy0, sy1 = c[:, 1] - r, 1 - c[:, 1] - r
    slacks = (sp, sx0, sx1, sy0, sy1, r)
    if min(s.min() for s in slacks) <= 0:
        return 1e20, np.zeros_like(v)
    f = -r.sum() - mu * sum(np.log(s).sum() for s in slacks)
    g, gr = np.zeros((n, 2)), -np.ones(n)
    u = diff / dist[:, None]
    wp = mu / sp
    np.add.at(g, i, -wp[:, None] * u)
    np.add.at(g, j, wp[:, None] * u)
    np.add.at(gr, i, wp)
    np.add.at(gr, j, wp)
    g[:, 0] += -mu / sx0 + mu / sx1
    g[:, 1] += -mu / sy0 + mu / sy1
    gr += mu * (1 / sx0 + 1 / sx1 + 1 / sy0 + 1 / sy1) - mu / r
    return f, np.concatenate([g.ravel(), gr])


def _solve_once(n, rng, init=None):
    base = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2)) if init is None else np.clip(init + rng.normal(0, 0.03, (n, 2)), 0, 1)
    iu = np.triu_indices(n, 1)
    x = np.concatenate([np.clip(base, 1e-3, 1 - 1e-3).ravel(), np.full(n, 1e-4)])
    mu = 1e-2
    for _ in range(8):
        x = minimize(_barrier_fg, x, args=(n, iu, mu), jac=True, method="L-BFGS-B",
                     options={"maxiter": MAX_ITER}).x
        mu /= 5
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
        guided = best is not None and time.process_time() - t0 > 0.5 * TIME_FRACTION * time_limit and rng.random() < 0.5
        c, r = _solve_once(n, rng, best[0] if guided else None)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
