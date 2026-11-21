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


def _solve_once(n, rng):
    c = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2))
    r = np.zeros(n)  # zero radii: the start is feasible
    I, J = np.triu_indices(n, 1)
    m, idx, rows = len(I), np.arange(n), np.arange(len(I))
    delta, merit = 0.1, 0.0

    def violation(c, r):
        d = np.sqrt(((c[I] - c[J]) ** 2).sum(1))
        g = np.concatenate([d - r[I] - r[J], c[:, 0] - r, 1 - c[:, 0] - r, c[:, 1] - r, 1 - c[:, 1] - r])
        return min(0.0, g.min())

    cost = np.zeros(3 * n)
    cost[2 * n:] = -1
    for _ in range(MAX_ITER):
        diff = c[I] - c[J]
        dist = np.sqrt((diff ** 2).sum(1))
        u = diff / np.maximum(dist, 1e-12)[:, None]
        A = np.zeros((m + 5 * n, 3 * n))
        A[rows, 2 * I] = -u[:, 0]
        A[rows, 2 * I + 1] = -u[:, 1]
        A[rows, 2 * J] = u[:, 0]
        A[rows, 2 * J + 1] = u[:, 1]
        A[rows, 2 * n + I] = 1
        A[rows, 2 * n + J] = 1
        A[m + idx, 2 * idx] = -1                  # left wall
        A[m + idx, 2 * n + idx] = 1
        A[m + n + idx, 2 * idx] = 1               # right wall
        A[m + n + idx, 2 * n + idx] = 1
        A[m + 2 * n + idx, 2 * idx + 1] = -1      # bottom wall
        A[m + 2 * n + idx, 2 * n + idx] = 1
        A[m + 3 * n + idx, 2 * idx + 1] = 1       # top wall
        A[m + 3 * n + idx, 2 * n + idx] = 1
        A[m + 4 * n + idx, 2 * n + idx] = -1      # radius >= 0
        b = np.concatenate([dist - r[I] - r[J], c[:, 0] - r, 1 - c[:, 0] - r, c[:, 1] - r, 1 - c[:, 1] - r, r])
        res = linprog(cost, A_ub=A, b_ub=b, bounds=(-delta, delta), method="highs")
        if res.status == 0:
            c_new = c + res.x[:2 * n].reshape(n, 2)
            r_new = r + res.x[2 * n:]
            new_merit = violation(c_new, r_new)
            if new_merit >= merit - 1e-12:
                c, r, merit = c_new, r_new, new_merit
                delta = min(1.5 * delta, 0.1)
                continue
        delta *= 0.5
        if delta < 1e-7:
            break
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
