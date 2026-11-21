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


def _relax(n, x0, iu):
    """Penalty stages with L-BFGS-B: negative radius sum plus growing quadratic penalty on overlaps and boundary violations."""
    i, j = iu

    def pen(v, w):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        dc = c[i] - c[j]
        d = np.sqrt((dc ** 2).sum(1)) + 1e-12
        ov = np.maximum(r[i] + r[j] - d, 0)
        v1, v2 = np.maximum(r - c[:, 0], 0), np.maximum(r - (1 - c[:, 0]), 0)
        v3, v4 = np.maximum(r - c[:, 1], 0), np.maximum(r - (1 - c[:, 1]), 0)
        v5 = np.maximum(-r, 0)
        f = -r.sum() + w * ((ov ** 2).sum() + (v1 ** 2 + v2 ** 2 + v3 ** 2 + v4 ** 2 + v5 ** 2).sum())
        gd = 2 * w * ov
        gr = np.zeros(n)
        np.add.at(gr, i, gd)
        np.add.at(gr, j, gd)
        gr += 2 * w * (v1 + v2 + v3 + v4 - v5) - 1
        gc = np.zeros((n, 2))
        u = dc / d[:, None]
        np.add.at(gc, i, -gd[:, None] * u)
        np.add.at(gc, j, gd[:, None] * u)
        gc[:, 0] += 2 * w * (-v1 + v2)
        gc[:, 1] += 2 * w * (-v3 + v4)
        return f, np.concatenate([gc.ravel(), gr])

    bounds = [(None, None)] * (2 * n) + [(0, None)] * n
    v = x0
    for w in (10.0, 100.0, 1000.0):
        v = minimize(pen, v, args=(w,), jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 100}).x
    return v


def _solve_once(n, rng, start=None):
    x0 = start if start is not None else np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)
    if start is None:
        x0 = _relax(n, x0, iu)

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
    total = TIME_FRACTION * time_limit
    while time.process_time() - t0 < 0.5 * total:   # random restarts for half of the time
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    t1 = time.process_time()
    span = max(total - (t1 - t0), 1e-9)
    while time.process_time() - t0 < total:   # basin hopping from the best packing, perturbation decays linearly
        frac = min((time.process_time() - t1) / span, 1.0)
        scale = 0.1 + (0.005 - 0.1) * frac
        c, r = best
        c0 = np.clip(c + scale * rng.standard_normal(c.shape), 0, 1)
        c, r = _solve_once(n, rng, np.concatenate([c0.ravel(), r]))
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if r.sum() > best[1].sum():
            best = (c, r)
    return best
