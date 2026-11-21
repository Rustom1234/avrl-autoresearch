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


def _maximin_centres(n, rng):
    cand = rng.uniform(INIT_LOW, 1 - INIT_LOW, (400, 2))
    chosen = [cand[0]]
    dmin = np.linalg.norm(cand - cand[0], axis=1)
    for _ in range(n - 1):
        k = np.argmax(dmin)
        chosen.append(cand[k])
        dmin = np.minimum(dmin, np.linalg.norm(cand - cand[k], axis=1))
    return np.array(chosen)


def _solve_once(n, rng, start=None):
    if start is None and rng.random() < 0.5:
        x0 = np.concatenate([_maximin_centres(n, rng).ravel(), np.full(n, INIT_RADIUS)])
    else:
        x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    if start is not None:
        c0, r0 = start[0].copy(), start[1].copy()
        if rng.random() < 0.5:
            k = rng.integers(2, 4)
            idx = rng.choice(n, k, replace=False)
            c0[idx] = rng.uniform(INIT_LOW, 1 - INIT_LOW, (k, 2))
            r0[idx] = INIT_RADIUS * 0.5
        else:
            c0 = np.clip(c0 + rng.normal(0, 0.02, c0.shape), 0.01, 0.99)
        x0 = np.concatenate([c0.ravel(), r0])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    def cons_jac(v):
        c = v[:2 * n].reshape(n, 2)
        diff = c[iu[0]] - c[iu[1]]
        dist = np.maximum(np.sqrt((diff ** 2).sum(1)), 1e-12)
        g = diff / dist[:, None]
        P = len(iu[0])
        J = np.zeros((P + 5 * n, 3 * n))
        k = np.arange(P)
        J[k, 2 * iu[0]] = g[:, 0]
        J[k, 2 * iu[0] + 1] = g[:, 1]
        J[k, 2 * iu[1]] = -g[:, 0]
        J[k, 2 * iu[1] + 1] = -g[:, 1]
        J[k, 2 * n + iu[0]] = -1
        J[k, 2 * n + iu[1]] = -1
        w = np.arange(n)
        J[P + w, 2 * w] = 1
        J[P + n + w, 2 * w + 1] = 1
        J[P + 2 * n + w, 2 * w] = -1
        J[P + 3 * n + w, 2 * w + 1] = -1
        J[P + np.arange(4 * n), 2 * n + np.tile(w, 4)] = -1
        J[P + 4 * n + w, 2 * n + w] = 1
        return J

    def obj_jac(v):
        return np.concatenate([np.zeros(2 * n), -np.ones(n)])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, jac=obj_jac, method="SLSQP",
                   constraints={"type": "ineq", "fun": cons, "jac": cons_jac}, options={"maxiter": MAX_ITER})
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
        if best is not None and time.process_time() - t0 > 0.3 * TIME_FRACTION * time_limit:
            c, r = _solve_once(n, rng, best)
        else:
            c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
