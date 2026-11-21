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
MAX_ITER = 100
MU = 50.0
RELAX_ITER = 50
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


def _relax(x0, n, iu):
    """L-BFGS-B on -sum(r) + MU * squared overlap and wall violations."""
    i, j = iu

    def f(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        g = np.zeros_like(v)
        gc, gr = g[:2 * n].reshape(n, 2), g[2 * n:]
        diff = c[i] - c[j]
        d = np.sqrt((diff ** 2).sum(1)) + 1e-12
        o = np.maximum(r[i] + r[j] - d, 0)
        e = (o ** 2).sum()
        np.add.at(gr, i, 2 * o)
        np.add.at(gr, j, 2 * o)
        u = (2 * o / d)[:, None] * diff
        np.add.at(gc, i, -u)
        np.add.at(gc, j, u)
        lo = np.maximum(r[:, None] - c, 0)
        hi = np.maximum(c + r[:, None] - 1, 0)
        e += (lo ** 2).sum() + (hi ** 2).sum()
        gc += -2 * lo + 2 * hi
        gr += 2 * lo.sum(1) + 2 * hi.sum(1)
        g *= MU
        gr -= 1
        return -r.sum() + MU * e, g

    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(0, 1)] * (2 * n) + [(0, 0.3)] * n,
                   options={"maxiter": RELAX_ITER})
    return res.x


def _solve_once(n, rng, x0=None):
    iu = np.triu_indices(n, 1)
    if x0 is None:
        x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
        x0 = _relax(x0, n, iu)

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
