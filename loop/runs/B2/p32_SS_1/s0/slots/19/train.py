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
ANNEAL = [0.06, 0.08, 0.095]


def _relax(c, rad, iters):
    """L-BFGS-B on a smooth overlap and wall penalty for fixed radii rad."""
    n = len(c)
    iu = np.triu_indices(n, 1)

    def energy(x):
        p = x.reshape(n, 2)
        diff = p[iu[0]] - p[iu[1]]
        d = np.sqrt((diff ** 2).sum(1)) + 1e-12
        ov = np.maximum(0, rad[iu[0]] + rad[iu[1]] - d)
        e = (ov ** 2).sum()
        g = np.zeros((n, 2))
        coef = (-2 * ov / d)[:, None] * diff
        np.add.at(g, iu[0], coef)
        np.add.at(g, iu[1], -coef)
        lo = np.maximum(0, rad[:, None] - p)
        hi = np.maximum(0, p + rad[:, None] - 1)
        e += (lo ** 2).sum() + (hi ** 2).sum()
        g += -2 * lo + 2 * hi
        return e, g.ravel()

    res = minimize(energy, c.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": iters})
    return res.x.reshape(n, 2)


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


def _solve_once(n, rng, x0=None):
    if x0 is None:
        c = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2))
        for t in ANNEAL:
            c = _relax(c, np.full(n, t), 30)
        x0 = np.concatenate([c.ravel(), np.full(n, 0.9 * ANNEAL[-1])])
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
