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


def _solve_once(n, rng, start=None):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    if start is not None:
        c0, r0 = start[0].copy(), start[1].copy()
        if rng.random() < 0.5:
            idx = rng.choice(n, rng.integers(2, 5), replace=False)
            c0[idx] = rng.uniform(INIT_LOW, 1 - INIT_LOW, (len(idx), 2))
            r0[idx] = INIT_RADIUS * 0.5
        else:
            c0 = c0 + rng.normal(0, 0.02, c0.shape)
        x0 = np.concatenate([np.clip(c0, 0, 1).ravel(), r0 * 0.9])
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


def solve(n, seed, time_limit):
    rng, t0 = np.random.default_rng(seed), time.process_time()
    budget = TIME_FRACTION * time_limit

    def run(start):
        c, r = _solve_once(n, rng, start)
        c = np.clip(c, 0, 1)
        return c, _repair(c, r)

    pool = []
    while time.process_time() - t0 < 0.3 * budget or not pool:
        pool.append(run(None))
    pool.sort(key=lambda p: -p[1].sum())
    best = pool[0]
    second = next((p for p in pool[1:] if abs(p[1].sum() - best[1].sum()) > 1e-6), None)
    walkers = [best, second if second is not None else run(None)]
    fails = [0, 0]
    k = 0
    while time.process_time() - t0 < budget:
        w = k % 2
        k += 1
        cand = run(walkers[w])
        if cand[1].sum() > best[1].sum():
            best = cand
        if cand[1].sum() > walkers[w][1].sum():
            walkers[w], fails[w] = cand, 0
        else:
            fails[w] += 1
            if fails[w] >= 25:
                walkers[w], fails[w] = run(None), 0
                if walkers[w][1].sum() > best[1].sum():
                    best = walkers[w]
    return best
