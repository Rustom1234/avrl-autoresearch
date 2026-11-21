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
    x0 = start if start is not None else np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
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


ELITE = 4


def _push(pool, c, r):
    """Keep the ELITE best packings, sorted by sum of radii."""
    pool.append((c, r))
    pool.sort(key=lambda e: -e[1].sum())
    del pool[ELITE:]


def _crossover(pool, n, rng):
    """Child start: parent A's circles on one side of a random line, parent B's on the other, count fixed to n."""
    i, j = rng.choice(len(pool), 2, replace=False)
    (ca, ra), (cb, rb) = pool[i], pool[j]
    th = rng.uniform(0, np.pi)
    u = np.array([np.cos(th), np.sin(th)])
    t = u @ rng.uniform(0, 1, 2)
    pa, pb = ca @ u - t, cb @ u - t
    ka, kb = pa <= 0, pb > 0
    c = np.vstack([ca[ka], cb[kb]])
    r = np.concatenate([ra[ka], rb[kb]])
    dist = np.abs(np.concatenate([pa[ka], pb[kb]]))
    m = len(r)
    if m == 0:
        return None
    if m >= n:   # drop the circles nearest the line
        idx = np.argsort(dist)[m - n:]
        c, r = c[idx], r[idx]
    else:        # duplicate the circles nearest the line, with a small jitter
        idx = np.argsort(dist)[np.arange(n - m) % m]
        c = np.vstack([c, c[idx] + 1e-3 * rng.standard_normal((n - m, 2))])
        r = np.concatenate([r, r[idx]])
    return np.concatenate([np.clip(c, 0, 1).ravel(), r])


def solve(n, seed, time_limit):
    rng, t0, pool = np.random.default_rng(seed), time.process_time(), []
    total = TIME_FRACTION * time_limit
    while time.process_time() - t0 < 0.5 * total:   # restarts for half of the time, crossover children once two parents exist
        start = _crossover(pool, n, rng) if len(pool) >= 2 else None
        c, r = _solve_once(n, rng, start)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        _push(pool, c, r)
    t1 = time.process_time()
    span = max(total - (t1 - t0), 1e-9)
    while time.process_time() - t0 < total:   # basin hopping from the best packing, perturbation decays linearly
        frac = min((time.process_time() - t1) / span, 1.0)
        scale = 0.1 + (0.005 - 0.1) * frac
        c, r = pool[0]
        c0 = np.clip(c + scale * rng.standard_normal(c.shape), 0, 1)
        c, r = _solve_once(n, rng, np.concatenate([c0.ravel(), r]))
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        _push(pool, c, r)
    return pool[0]
