"""train.py: packs n circles into the unit square so that the sum of their radii is as large as possible.

The harness calls solve(n, seed, time_limit) and checks the returned packing exactly.
Returns (centers, radii): centers is an (n, 2) array, radii an (n,) array. Every circle must lie inside
[0, 1] x [0, 1] and no two circles may overlap.
"""
import time

import numpy as np
from scipy.optimize import linprog, minimize

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


def _lp_polish(c, r, steps=30):
    """Sequential LP: linearise all constraints around (c, r) and take trust-region steps that grow sum(r)."""
    n = len(r)
    iu = np.triu_indices(n, 1)
    m = len(iu[0])
    idx = np.arange(n)
    k = np.arange(m)

    def slack(c, r):
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    t = 0.05
    for _ in range(steps):
        if t < 1e-7:
            break
        diff = c[iu[0]] - c[iu[1]]
        u = diff / np.maximum(np.sqrt((diff ** 2).sum(1)), 1e-12)[:, None]
        # variables: [dc (2n), dr (n)];  rows are written as A @ x <= g
        A = np.zeros((m + 5 * n, 3 * n))
        A[k, 2 * iu[0]] = -u[:, 0]
        A[k, 2 * iu[0] + 1] = -u[:, 1]
        A[k, 2 * iu[1]] = u[:, 0]
        A[k, 2 * iu[1] + 1] = u[:, 1]
        A[k, 2 * n + iu[0]] = 1
        A[k, 2 * n + iu[1]] = 1
        A[m + idx, 2 * idx] = -1                  # x - r >= 0
        A[m + idx, 2 * n + idx] = 1
        A[m + n + idx, 2 * idx + 1] = -1          # y - r >= 0
        A[m + n + idx, 2 * n + idx] = 1
        A[m + 2 * n + idx, 2 * idx] = 1           # 1 - x - r >= 0
        A[m + 2 * n + idx, 2 * n + idx] = 1
        A[m + 3 * n + idx, 2 * idx + 1] = 1       # 1 - y - r >= 0
        A[m + 3 * n + idx, 2 * n + idx] = 1
        A[m + 4 * n + idx, 2 * n + idx] = -1      # r >= 0
        g = slack(c, r)
        res = linprog(np.r_[np.zeros(2 * n), -np.ones(n)], A_ub=A, b_ub=g, bounds=(-t, t), method="highs")
        if res.status == 0:
            nc = c + res.x[:2 * n].reshape(n, 2)
            nr = r + res.x[2 * n:]
            if slack(nc, nr).min() >= -1e-12 and nr.sum() > r.sum():
                c, r = nc, nr
                continue
        t /= 2
    return c, r


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        guided = best is not None and time.process_time() - t0 > 0.5 * TIME_FRACTION * time_limit and rng.random() < 0.5
        c, r = _solve_once(n, rng, best[0] if guided else None)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    c, r = _lp_polish(best[0], best[1])
    c = np.clip(c, 0, 1)
    r = _repair(c, r)
    if r.sum() > best[1].sum():
        best = (c, r)
    return best
