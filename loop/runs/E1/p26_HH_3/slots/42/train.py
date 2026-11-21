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
RESTART_FRACTION = 0.3
LP_DELTA0 = 0.01
LP_MAX_DELTA = 0.1
LP_MIN_DELTA = 1e-7


def _optimise(n, x0):
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": MAX_ITER})
    return res.x[:2 * n].reshape(n, 2), res.x[2 * n:]


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


def _lp_polish(c, r, deadline):
    """Sequential LP with a trust region: linearised pair and wall constraints, maximise the sum of radii."""
    n = len(r)
    iu = np.triu_indices(n, 1)
    m = len(iu[0])
    k, i = np.arange(m), np.arange(n)
    delta = LP_DELTA0
    cost = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    while delta > LP_MIN_DELTA and time.process_time() < deadline:
        diff = c[iu[0]] - c[iu[1]]
        d = np.maximum(np.sqrt((diff ** 2).sum(1)), 1e-12)
        g = diff / d[:, None]
        A = np.zeros((m + 4 * n, 3 * n))
        b = np.zeros(m + 4 * n)
        # pair rows: d_ij + g.(dc_i - dc_j) - (r_i + dr_i + r_j + dr_j) >= 0
        A[k, 2 * iu[0]] = -g[:, 0]
        A[k, 2 * iu[0] + 1] = -g[:, 1]
        A[k, 2 * iu[1]] = g[:, 0]
        A[k, 2 * iu[1] + 1] = g[:, 1]
        A[k, 2 * n + iu[0]] = 1
        A[k, 2 * n + iu[1]] = 1
        b[:m] = d - r[iu[0]] - r[iu[1]]
        # wall rows: x - r >= 0, y - r >= 0, 1 - x - r >= 0, 1 - y - r >= 0
        A[m + i, 2 * i] = -1
        A[m + i, 2 * n + i] = 1
        b[m:m + n] = c[:, 0] - r
        A[m + n + i, 2 * i + 1] = -1
        A[m + n + i, 2 * n + i] = 1
        b[m + n:m + 2 * n] = c[:, 1] - r
        A[m + 2 * n + i, 2 * i] = 1
        A[m + 2 * n + i, 2 * n + i] = 1
        b[m + 2 * n:m + 3 * n] = 1 - c[:, 0] - r
        A[m + 3 * n + i, 2 * i + 1] = 1
        A[m + 3 * n + i, 2 * n + i] = 1
        b[m + 3 * n:] = 1 - c[:, 1] - r
        lb = np.concatenate([np.full(2 * n, -delta), np.maximum(-delta, -r)])
        ub = np.full(3 * n, delta)
        res = linprog(cost, A_ub=A, b_ub=b, bounds=np.column_stack([lb, ub]), method="highs")
        if res.status != 0:
            delta *= 0.5
            continue
        c_new = c + res.x[:2 * n].reshape(n, 2)
        r_new = r + res.x[2 * n:]
        if r_new.sum() - r.sum() > 1e-12:   # linearised pair constraints are lower bounds (d is convex), so the step stays valid
            c, r = c_new, r_new
            delta = min(1.5 * delta, LP_MAX_DELTA)
        else:
            delta *= 0.5
    return c, r


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
    while time.process_time() - t0 < budget:   # remove-and-reinsert the smallest circle
        c2, r2 = c.copy(), r.copy()
        i = np.argmin(r2)
        c2[i] = rng.uniform(0, 1, 2)
        r2[i] = 0
        c2, r2 = _lp_polish(c2, r2, t0 + budget)
        c2 = np.clip(c2, 0, 1)
        r2 = _repair(c2, r2)
        if r2.sum() > r.sum():
            c, r = c2, r2
    return c, r
