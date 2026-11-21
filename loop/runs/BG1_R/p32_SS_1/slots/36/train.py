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


def _cross(pool, n, rng):
    a, b = rng.choice(len(pool), 2, replace=False)
    t = rng.uniform(0.3, 0.7)
    (ca, ra), (cb, rb) = pool[a], pool[b]
    ma, mb = ca[:, 0] < t, cb[:, 0] >= t
    c, r = np.concatenate([ca[ma], cb[mb]]), np.concatenate([ra[ma], rb[mb]])
    if len(r) > n:
        k = rng.choice(len(r), n, replace=False)
        c, r = c[k], r[k]
    elif len(r) < n:
        m = n - len(r)
        c = np.vstack([c, rng.uniform(INIT_LOW, 1 - INIT_LOW, (m, 2))])
        r = np.concatenate([r, np.full(m, INIT_RADIUS)])
    return np.concatenate([c.ravel(), r])


def _reshuffle(c, r, rng):
    n = len(r)
    c, r = c.copy(), r.copy()
    p = rng.uniform(0, 1, 2)
    m = ((c - p) ** 2).sum(1) < 0.25 ** 2
    k = int(m.sum())
    if k:
        ang, rad = rng.uniform(0, 2 * np.pi, k), 0.25 * np.sqrt(rng.uniform(0, 1, k))
        c[m] = np.clip(p + rad[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1), 0.03, 0.97)
        r[m] = 0.03
    i, j = np.triu_indices(n, 1)

    def f(x):
        q = x.reshape(n, 2)
        d = q[i] - q[j]
        dist = np.sqrt((d ** 2).sum(1)) + 1e-12
        ov = np.maximum(r[i] + r[j] - dist, 0)
        lo, hi = np.maximum(r[:, None] - q, 0), np.maximum(q + r[:, None] - 1, 0)
        g = np.zeros((n, 2))
        w = (-2 * ov / dist)[:, None] * d
        np.add.at(g, i, w)
        np.add.at(g, j, -w)
        g += -2 * lo + 2 * hi
        return (ov ** 2).sum() + (lo ** 2).sum() + (hi ** 2).sum(), g.ravel()

    res = minimize(f, c.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 100})
    return np.concatenate([np.clip(res.x, 0, 1), r])


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


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    pool = []
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # restarts until the time is used
        if pool and time.process_time() - t0 > 0.6 * TIME_FRACTION * time_limit:
            x0 = _reshuffle(pool[0][0], pool[0][1], rng)
        else:
            x0 = _cross(pool, n, rng) if len(pool) >= 2 and rng.random() < 0.7 else None
        c, r = _solve_once(n, rng, x0)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        pool.append((c, r))
        pool.sort(key=lambda p: -p[1].sum())
        pool = pool[:5]
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
