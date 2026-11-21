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


def _greedy_init(n, rng, cands=400):
    """Place circles one by one at the candidate point allowing the largest non-overlapping radius."""
    c, r = np.zeros((n, 2)), np.zeros(n)
    for k in range(n):
        p = rng.uniform(0, 1, (cands, 2))
        rad = np.minimum(np.minimum(p[:, 0], p[:, 1]), np.minimum(1 - p[:, 0], 1 - p[:, 1]))
        if k:
            d = np.sqrt(((p[:, None] - c[None, :k]) ** 2).sum(-1))
            rad = np.minimum(rad, (d - r[None, :k]).min(1))
        j = np.argmax(rad)
        c[k], r[k] = p[j], max(rad[j], 0)
    return c, r


def _solve_once(n, rng):
    c0, r0 = _greedy_init(n, rng)
    x = np.concatenate([c0.ravel(), r0])
    iu = np.triu_indices(n, 1)
    bounds = [(0, 1)] * (3 * n)   # centres and radii are box-bounded
    for lam in (10.0, 1e2, 1e3, 1e4, 1e5):   # increasing penalty weight on constraint violations

        def fg(v, lam=lam):
            c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
            diff = c[iu[0]] - c[iu[1]]
            d = np.sqrt((diff ** 2).sum(1)) + 1e-12
            p = np.maximum(0, r[iu[0]] + r[iu[1]] - d)
            f = -r.sum() + lam * (p ** 2).sum()
            gr = -np.ones(n)
            gc = np.zeros((n, 2))
            w = 2 * lam * p
            np.add.at(gr, iu[0], w)
            np.add.at(gr, iu[1], w)
            u = (diff / d[:, None]) * w[:, None]
            np.add.at(gc, iu[0], -u)
            np.add.at(gc, iu[1], u)
            for axis, sgn, off in ((0, 1, 0), (1, 1, 0), (0, -1, 1), (1, -1, 1)):
                q = np.maximum(0, r - (off + sgn * c[:, axis]))   # wall: r <= x, r <= y, r <= 1-x, r <= 1-y
                f += lam * (q ** 2).sum()
                gr += 2 * lam * q
                gc[:, axis] += -2 * lam * q * sgn
            return f, np.concatenate([gc.ravel(), gr])

        x = minimize(fg, x, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": MAX_ITER}).x
    return x[:2 * n].reshape(n, 2), x[2 * n:]


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
