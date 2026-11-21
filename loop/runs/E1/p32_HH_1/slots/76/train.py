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
ISLANDS = 4
ELITE = 3
PER_ROUND = 5


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


def solve(n, seed, time_limit):
    rngs = [np.random.default_rng(seed + k) for k in range(ISLANDS)]
    t0, best, rnd = time.process_time(), None, 0
    elites = [[] for _ in range(ISLANDS)]
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        for k in range(ISLANDS):
            rng = rngs[k]
            for _ in range(PER_ROUND):
                if time.process_time() - t0 >= TIME_FRACTION * time_limit:
                    break
                pool = elites[k] + elites[(k + 1) % ISLANDS]
                start = None
                if rnd >= 2 and pool and rng.random() < 0.5:
                    start = pool[rng.integers(len(pool))][0]
                c, r = _solve_once(n, rng, start)
                c = np.clip(c, 0, 1)
                r = _repair(c, r)
                if best is None or r.sum() > best[1].sum():
                    best = (c, r)
                elites[k] = sorted(elites[k] + [(c, r)], key=lambda e: -e[1].sum())[:ELITE]
        migrants = [e[0] if e else None for e in elites]
        for k in range(ISLANDS):
            dst = (k + 1) % ISLANDS
            if migrants[k] is None:
                continue
            if len(elites[dst]) < ELITE:
                elites[dst].append(migrants[k])
            elif migrants[k][1].sum() > elites[dst][-1][1].sum():
                elites[dst][-1] = migrants[k]
            elites[dst].sort(key=lambda e: -e[1].sum())
        rnd += 1
    return best
