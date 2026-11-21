"""train.py: packs n circles into the unit square so that the sum of their radii is as large as possible.

The harness calls solve(n, seed, time_limit) and checks the returned packing exactly.
Returns (centers, radii): centers is an (n, 2) array, radii an (n,) array. Every circle must lie inside
[0, 1] x [0, 1] and no two circles may overlap.
"""
import time

import numpy as np
from scipy.optimize import linprog

# Settings
TIME_FRACTION = 0.8
INIT_LOW = 0.1
MAX_ITER = 200
INIT_RADIUS = 0.05
SHRINK = 1e-9


def _slp(c, r, cap, iters=60):
    """Sequential LP with a box trust region on the step, linearising pair distances, walls and the radius cap."""
    n = len(r)
    i, j = np.triu_indices(n, 1)
    m = len(i)
    rows = np.arange(m)
    I2 = np.eye(2 * n)
    idx = np.arange(2 * n) // 2
    R = np.zeros((2 * n, n))
    R[np.arange(2 * n), idx] = 1
    Wl = np.hstack([-I2, R])
    Wu = np.hstack([I2, R])
    Rn = np.hstack([np.zeros((n, 2 * n)), -np.eye(n)])
    Rc = np.hstack([np.zeros((n, 2 * n)), np.eye(n)])
    cost = np.concatenate([np.zeros(2 * n), -np.ones(n)])

    def viol(c, r):
        d = np.sqrt(((c[i] - c[j]) ** 2).sum(1))
        g = np.concatenate([d - r[i] - r[j], c.ravel() - r[idx], 1 - c.ravel() - r[idx], r, cap - r])
        return max(0.0, -g.min())

    delta, obj = 0.05, r.sum()
    for _ in range(iters):
        if delta < 1e-9:
            break
        diff = c[i] - c[j]
        d = np.sqrt((diff ** 2).sum(1))
        u = diff / np.maximum(d, 1e-12)[:, None]
        P = np.zeros((m, 3 * n))
        P[rows, 2 * i] = -u[:, 0]
        P[rows, 2 * i + 1] = -u[:, 1]
        P[rows, 2 * j] = u[:, 0]
        P[rows, 2 * j + 1] = u[:, 1]
        P[rows, 2 * n + i] = 1
        P[rows, 2 * n + j] = 1
        A = np.vstack([P, Wl, Wu, Rn, Rc])
        b = np.concatenate([d - r[i] - r[j], c.ravel() - r[idx], 1 - c.ravel() - r[idx], r, cap - r])
        res = linprog(cost, A_ub=A, b_ub=b, bounds=(-delta, delta), method="highs")
        if res.status != 0:
            delta *= 0.5
            continue
        c_new = c + res.x[:2 * n].reshape(n, 2)
        r_new = r + res.x[2 * n:]
        new_obj = r_new.sum()
        if viol(c_new, r_new) <= max(viol(c, r), 1e-7) and new_obj > obj + 1e-10:
            c, r, obj = c_new, r_new, new_obj
            delta = min(delta * 1.5, 0.2)
        else:
            delta *= 0.5
    return c, r


def _solve_once(n, rng):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    c = np.clip(x0[:2 * n].reshape(n, 2), 0, 1)
    r = _repair(c, x0[2 * n:])   # feasible start for the LP steps
    for cap in list(np.arange(0.05, 0.4 + 1e-9, 0.02)) + [1.0]:
        c, r = _slp(c, np.minimum(r, cap), cap)
    return c, r


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
