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


def _relax(n, rng, R=0.09):
    """Relax uniform random centres with L-BFGS-B on a squared pair-overlap penalty (target radius R)."""
    iu = np.triu_indices(n, 1)

    def energy(x):
        c = x.reshape(n, 2)
        diff = c[iu[0]] - c[iu[1]]
        d = np.sqrt((diff ** 2).sum(1)) + 1e-12
        ov = np.maximum(0, 2 * R - d)
        g = np.zeros((n, 2))
        coef = (-2 * ov / d)[:, None] * diff
        np.add.at(g, iu[0], coef)
        np.add.at(g, iu[1], -coef)
        return (ov ** 2).sum(), g.ravel()

    x0 = rng.uniform(0.1, 0.9, 2 * n)
    res = minimize(energy, x0, jac=True, method="L-BFGS-B", bounds=[(R, 1 - R)] * (2 * n),
                   options={"maxiter": 60})
    return res.x.reshape(n, 2)


def _relocate(pool, n, rng):
    """Copy a random pool member and move one circle (chosen with prob ~ 1/radius) to the point of largest clearance."""
    c0, r0 = pool[rng.integers(len(pool))]
    c, r = c0.copy(), r0.copy()
    p = 1.0 / np.maximum(r, 1e-6)
    i = rng.choice(n, p=p / p.sum())
    others = np.delete(np.arange(n), i)
    cand = rng.uniform(0, 1, (400, 2))
    wall = np.minimum(np.minimum(cand[:, 0], cand[:, 1]), np.minimum(1 - cand[:, 0], 1 - cand[:, 1]))
    d = np.sqrt(((cand[:, None, :] - c[others][None]) ** 2).sum(-1))
    clear = np.minimum(wall, (d - r[others][None]).min(1))
    k = np.argmax(clear)
    c[i] = cand[k]
    r[i] = max(0.01, 0.9 * clear[k])
    return np.concatenate([c.ravel(), r])


def _solve_once(n, rng, x0=None):
    if x0 is None:
        x0 = np.concatenate([_relax(n, rng).ravel(), np.full(n, INIT_RADIUS)])
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
        x0 = _relocate(pool, n, rng) if len(pool) >= 2 and rng.random() < 0.7 else None
        c, r = _solve_once(n, rng, x0)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        pool.append((c, r))
        pool.sort(key=lambda p: -p[1].sum())
        pool = pool[:5]
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
