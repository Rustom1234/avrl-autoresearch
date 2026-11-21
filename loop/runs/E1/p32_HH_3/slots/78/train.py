"""train.py: packs n circles into the unit square so that the sum of their radii is as large as possible.

The harness calls solve(n, seed, time_limit) and checks the returned packing exactly.
Returns (centers, radii): centers is an (n, 2) array, radii an (n,) array. Every circle must lie inside
[0, 1] x [0, 1] and no two circles may overlap.
"""
import time

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

# Settings
TIME_FRACTION = 0.8
INIT_LOW = 0.1
MAX_ITER = 200
INIT_RADIUS = 0.05
SHRINK = 1e-9


def _solve_once(n, rng, x0=None):
    if x0 is None:
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


def _relocate(c, r, rng, trials=30):
    """Move the smallest circle to the best of several random centres, scored by the local sum of radii."""
    c, r = c.copy(), r.copy()
    k = int(np.argmin(r))
    best = None
    for _ in range(trials):
        p = rng.uniform(0, 1, 2)
        d = np.sqrt(((c - p) ** 2).sum(1))
        d[k] = np.inf
        rk = max(min(p.min(), (1 - p).min(), (d - r).min()), 0)
        nb = d < rk + r + 0.05
        score = rk + r[nb].sum()
        if best is None or score > best[0]:
            best = (score, p, rk)
    _, p, rk = best
    c[k], r[k] = p, rk
    return c, r


CROSS_PROB = 0.5
ELITE_SIZE = 4


def _sym(c, k):
    """Apply one of the 8 symmetries of the unit square (k in 0..7) to the centres."""
    x, y = (c[:, 1], c[:, 0]) if k // 4 else (c[:, 0], c[:, 1])
    x = 1 - x if k % 2 else x
    y = 1 - y if (k // 2) % 2 else y
    return np.stack([x, y], 1)


def _update_pool(pool, c, r):
    """Keep the ELITE_SIZE best distinct packings (distinct by sum of radii)."""
    s = r.sum()
    if any(abs(s - p[0]) < 1e-7 for p in pool):
        return
    pool.append((s, c, r))
    pool.sort(key=lambda p: -p[0])
    del pool[ELITE_SIZE:]


def _crossover(pool, n, rng):
    """Average two elite packings after matching circles of the second to the first."""
    i, j = rng.choice(len(pool), 2, replace=False)
    (_, c1, r1), (_, c2, r2) = pool[i], pool[j]
    c2s = _sym(c2, int(rng.integers(8)))
    rows, cols = linear_sum_assignment(np.sqrt(((c1[:, None] - c2s[None]) ** 2).sum(-1)))
    perm = np.empty(n, int)
    perm[rows] = cols
    c = 0.5 * (c1 + c2s[perm]) + rng.normal(0, 0.01, c1.shape)
    r = 0.5 * (r1 + r2[perm])
    return np.clip(c, 0, 1), np.maximum(r, 0)


def solve(n, seed, time_limit):
    rng, t0 = np.random.default_rng(seed), time.process_time()
    c, r = _solve_once(n, rng)
    c = np.clip(c, 0, 1)
    best = (c, _repair(c, r))
    pool = []
    _update_pool(pool, best[0], best[1])
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        if len(pool) >= 2 and rng.random() < CROSS_PROB:   # Procrustes consensus crossover start
            c0, r0 = _crossover(pool, n, rng)
            c, r = _solve_once(n, rng, np.concatenate([c0.ravel(), r0]))
        else:                                              # basin hopping from the incumbent
            bc, br = best
            c0 = np.clip(bc + rng.normal(0, 0.02, bc.shape), 0, 1)
            r0 = np.maximum(br + rng.normal(0, 0.02, br.shape), 0)
            c0, r0 = _relocate(c0, r0, rng)
            c, r = _solve_once(n, rng, np.concatenate([c0.ravel(), r0]))
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        _update_pool(pool, c, r)
        if r.sum() > best[1].sum():
            best = (c, r)
    return best
