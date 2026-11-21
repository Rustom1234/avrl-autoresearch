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
SLP_ITER = 100
SLP_DELTA = 0.05


def _slp(c, r):
    """Sequential LP with a trust region: linearise the constraints and maximise sum(r) over bounded steps."""
    n = len(r)
    I, J = np.triu_indices(n, 1)
    m = len(I)
    ar = np.arange(n)
    k = np.arange(m)
    delta = SLP_DELTA
    r = _repair(c, r)
    cur_sum = r.sum()
    cost = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    for _ in range(SLP_ITER):
        if delta < 1e-7:
            break
        diff = c[I] - c[J]
        d = np.sqrt((diff ** 2).sum(1)) + 1e-12
        g = diff / d[:, None]
        A = np.zeros((m + 4 * n, 3 * n))
        b = np.empty(m + 4 * n)
        A[k, 2 * I] = -g[:, 0]
        A[k, 2 * I + 1] = -g[:, 1]
        A[k, 2 * J] = g[:, 0]
        A[k, 2 * J + 1] = g[:, 1]
        A[k, 2 * n + I] = 1
        A[k, 2 * n + J] = 1
        b[:m] = d - r[I] - r[J]
        walls = [(0, -1, c[:, 0] - r), (1, -1, c[:, 1] - r), (0, 1, 1 - c[:, 0] - r), (1, 1, 1 - c[:, 1] - r)]
        for q, (col, sgn, rhs) in enumerate(walls):
            rows = m + q * n + ar
            A[rows, 2 * ar + col] = sgn
            A[rows, 2 * n + ar] = 1
            b[rows] = rhs
        lo = np.concatenate([np.full(2 * n, -delta), np.maximum(-delta, -r)])
        bounds = np.column_stack([lo, np.full(3 * n, delta)])
        res = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if res.status != 0:
            delta *= 0.5
            continue
        nc = np.clip(c + res.x[:2 * n].reshape(n, 2), 0, 1)
        cand = _repair(nc, np.maximum(r + res.x[2 * n:], 0))
        if cand.sum() > cur_sum + 1e-12:
            c, r, cur_sum = nc, cand, cand.sum()
            delta = min(1.5 * delta, 0.2)
        else:
            delta *= 0.5
    return c, r


def _solve_once(n, rng):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    # Phase 1: all radii tied to one scalar t, maximise t
    def cons_eq(v):
        c, t = v[:2 * n].reshape(n, 2), v[2 * n]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - 2 * t
        return np.concatenate([d, c[:, 0] - t, c[:, 1] - t, 1 - c[:, 0] - t, 1 - c[:, 1] - t, [t]])

    x1 = np.concatenate([x0[:2 * n], [INIT_RADIUS]])
    res1 = minimize(lambda v: -v[2 * n], x1, method="SLSQP", constraints={"type": "ineq", "fun": cons_eq},
                    options={"maxiter": MAX_ITER})
    x2 = np.concatenate([res1.x[:2 * n], np.full(n, max(res1.x[2 * n], 0.0))])

    # Phase 2: sequential LP with a trust region replaces the free-radius SLSQP
    return _slp(x2[:2 * n].reshape(n, 2), x2[2 * n:])


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
