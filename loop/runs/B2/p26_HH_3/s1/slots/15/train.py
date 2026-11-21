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
RESTART_FRACTION = 0.3


def _optimise(n, x0):
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": MAX_ITER})
    return res.x[:2 * n].reshape(n, 2), res.x[2 * n:]


def _optimise_local(c, r, idx):
    n = len(r)
    fixed = np.setdiff1d(np.arange(n), idx)
    m = len(idx)
    iu = np.triu_indices(m, 1)
    cfix, rfix = c[fixed], r[fixed]

    def cons(v):
        cc, rr = v[:2 * m].reshape(m, 2), v[2 * m:]
        d = np.sqrt(((cc[iu[0]] - cc[iu[1]]) ** 2).sum(1)) - rr[iu[0]] - rr[iu[1]]
        df = np.sqrt(((cc[:, None] - cfix[None]) ** 2).sum(-1)) - rr[:, None] - rfix[None]
        return np.concatenate([d, df.ravel(), cc[:, 0] - rr, cc[:, 1] - rr, 1 - cc[:, 0] - rr, 1 - cc[:, 1] - rr, rr])

    x0 = np.concatenate([c[idx].ravel(), r[idx]])
    res = minimize(lambda v: -v[2 * m:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": 100})
    c, r = c.copy(), r.copy()
    c[idx] = res.x[:2 * m].reshape(m, 2)
    r[idx] = res.x[2 * m:]
    return c, r


def _solve_once(n, rng):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    return _optimise(n, x0)


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
    budget = TIME_FRACTION * time_limit
    while time.process_time() - t0 < RESTART_FRACTION * budget or best is None:   # short random-restart phase
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    c, r = best
    step = 0
    while time.process_time() - t0 < budget:   # local reinsert of the smallest circle, periodic full polish
        step += 1
        c2, r2 = c.copy(), r.copy()
        i = np.argmin(r2)
        c2[i] = rng.uniform(0, 1, 2)
        r2[i] = 0
        idx = np.argsort(np.linalg.norm(c2 - c2[i], axis=1))[:9]
        c2, r2 = _optimise_local(c2, r2, idx)
        c2 = np.clip(c2, 0, 1)
        r2 = _repair(c2, r2)
        accepted = r2.sum() > r.sum()
        if accepted:
            c, r = c2, r2
        if accepted or step % 20 == 0:
            c2, r2 = _optimise(n, np.concatenate([c.ravel(), r]))
            c2 = np.clip(c2, 0, 1)
            r2 = _repair(c2, r2)
            if r2.sum() > r.sum():
                c, r = c2, r2
    return c, r
