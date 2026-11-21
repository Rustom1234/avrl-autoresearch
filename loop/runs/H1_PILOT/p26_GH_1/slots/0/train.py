"""train.py: packs n circles into the unit square so that the sum of their radii is as large as possible.

The harness calls solve(n, seed, time_limit) and checks the returned packing exactly.
Returns (centers, radii): centers is an (n, 2) array, radii an (n,) array. Every circle must lie inside
[0, 1] x [0, 1] and no two circles may overlap.
"""
import time

import numpy as np
from scipy.optimize import minimize

# Settings
TIME_FRACTION = 0.5
INIT_LOW = 0.1
MAX_ITER = 200
INIT_RADIUS = 0.05
SHRINK = 1e-9


def _solve_once(n, rng):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
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


def _valid(c, r):
    d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair_ok = np.all(d >= r[:, None] + r[None] - 1e-12)
    box_ok = np.all(r >= 0) and np.all(r <= c.min(1)) and np.all(r <= 1 - c.max(1))
    return pair_ok and box_ok


def _anneal(c, r, rng, t_end):
    """Simulated annealing: move one centre or radius, repair, accept by Metropolis on sum(r)."""
    t_start, cur_c, cur_r = time.process_time(), c, r
    cur_s = best_s = r.sum()
    best_c, best_r = c, r
    span = max(t_end - t_start, 1e-9)
    while True:
        frac = (time.process_time() - t_start) / span
        if frac >= 1:
            break
        T = 0.01 * (0.01) ** frac   # geometric cooling from 1e-2 to 1e-4
        nc, nr = cur_c.copy(), cur_r.copy()
        i = rng.integers(len(nr))
        if rng.random() < 0.5:
            nc[i] = np.clip(nc[i] + rng.normal(0, T, 2), 0, 1)
        else:
            nr[i] = max(nr[i] + rng.normal(0, T), 0)
        nr = _repair(nc, nr)
        s = nr.sum()
        if (s >= cur_s or rng.random() < np.exp((s - cur_s) / T)) and _valid(nc, nr):
            cur_c, cur_r, cur_s = nc, nr, s
            if s > best_s:
                best_c, best_r, best_s = nc, nr, s
    return best_c, best_r


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts first
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return _anneal(best[0], best[1], rng, t0 + 0.9 * time_limit)   # anneal the best packing with the rest
