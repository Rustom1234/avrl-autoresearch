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


def _solve_once(n, rng):
    R = int(rng.integers(5, 8))
    m = int(np.ceil(n / R))
    pts = np.array([((j + 0.5 + 0.5 * (i % 2)) / (m + 0.5), (i + 0.5) / R) for i in range(R) for j in range(m)])
    pts = pts[rng.permutation(len(pts))[:n]] + rng.normal(0, 0.02, (n, 2))
    pts = np.clip(pts, 0.03, 0.97)
    x0 = np.concatenate([pts.ravel(), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    def energy(v, mu):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        diff = c[iu[0]] - c[iu[1]]
        d = np.sqrt((diff ** 2).sum(1)) + 1e-12
        o = np.maximum(r[iu[0]] + r[iu[1]] - d, 0)
        lo = np.maximum(r[:, None] - c, 0)
        hi = np.maximum(r[:, None] + c - 1, 0)
        val = -r.sum() + mu * ((o ** 2).sum() + (lo ** 2).sum() + (hi ** 2).sum())
        gr = -np.ones(n) + 2 * mu * (lo.sum(1) + hi.sum(1))
        np.add.at(gr, iu[0], 2 * mu * o)
        np.add.at(gr, iu[1], 2 * mu * o)
        gc = 2 * mu * (hi - lo)
        g = (-2 * mu * o / d)[:, None] * diff
        np.add.at(gc, iu[0], g)
        np.add.at(gc, iu[1], -g)
        return val, np.concatenate([gc.ravel(), gr])

    for mu in (10.0, 100.0, 1000.0):
        x0 = minimize(energy, x0, args=(mu,), jac=True, method="L-BFGS-B", options={"maxiter": 100}).x
    x0[:2 * n] = np.clip(x0[:2 * n], 0, 1)
    x0[2 * n:] = np.maximum(x0[2 * n:], 0.01)

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
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
