"""train.py: packs n circles into the unit square so that the sum of their radii is as large as possible.

The harness calls solve(n, seed, time_limit) and checks the returned packing exactly.
Returns (centers, radii): centers is an (n, 2) array, radii an (n,) array. Every circle must lie inside
[0, 1] x [0, 1] and no two circles may overlap.
"""
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

# Settings
TIME_FRACTION = 0.8
INIT_LOW = 0.1
MAX_ITER = 200
INIT_RADIUS = 0.05
SHRINK = 1e-9


def _soft_stage(n, x0):
    """Soft-min continuation: maximise sum(r) minus mu times a log-sum-exp violation, mu = 10 .. 1e4."""
    K = 300.0
    iu = np.triu_indices(n, 1)
    m = len(iu[0])

    def f(v, mu):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        diff = c[iu[0]] - c[iu[1]]
        d = np.sqrt((diff ** 2).sum(1)) + 1e-12
        g = np.concatenate([d - r[iu[0]] - r[iu[1]], c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r])
        lse = logsumexp(-K * g)
        L = lse / K
        w = np.exp(-K * g - lse)
        wp, wx, wy = w[:m], w[m:m + n], w[m + n:m + 2 * n]
        wl, wu = w[m + 2 * n:m + 3 * n], w[m + 3 * n:]
        q = (wp / d)[:, None] * diff
        gc = np.zeros((n, 2))
        np.add.at(gc, iu[0], q)
        np.add.at(gc, iu[1], -q)
        gc[:, 0] += wx - wl
        gc[:, 1] += wy - wu
        gr = -np.bincount(iu[0], weights=wp, minlength=n) - np.bincount(iu[1], weights=wp, minlength=n)
        gr -= wx + wy + wl + wu
        obj = -r.sum()
        grad = np.concatenate([np.zeros(2 * n), -np.ones(n)])
        if L > 0:
            obj += mu * L
            grad -= mu * np.concatenate([gc.ravel(), gr])
        return obj, grad

    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    v = np.asarray(x0, dtype=float)
    for mu in np.geomspace(10, 1e4, 6):
        v = minimize(f, v, args=(mu,), jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300}).x
    return v


def _solve_once(n, rng, x0=None):
    if x0 is None:
        x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    x0 = _soft_stage(n, x0)
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
    rng, t0 = np.random.default_rng(seed), time.process_time()
    budget = TIME_FRACTION * time_limit
    cur = best = None
    while time.process_time() - t0 < budget:
        if cur is None:
            c, r = _solve_once(n, rng)
        else:
            frac = (time.process_time() - t0) / budget
            T = 0.01 * (1e-4 / 0.01) ** frac
            c0 = np.clip(cur[0] + rng.normal(0, 0.05, (n, 2)), 0, 1)
            c, r = _solve_once(n, rng, np.concatenate([c0.ravel(), cur[1]]))
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if cur is None or r.sum() >= cur[1].sum() or np.exp((r.sum() - cur[1].sum()) / T) > rng.uniform():
            cur = (c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
