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
    x = np.concatenate([pts.ravel(), np.full(n, INIT_RADIUS)])
    I, J = np.triu_indices(n, 1)
    bounds = [(0, 1)] * (2 * n) + [(0, 0.3)] * n

    def gaps(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        dvec = c[I] - c[J]
        d = np.maximum(np.sqrt((dvec ** 2).sum(1)), 1e-12)
        gp = d - r[I] - r[J]
        gw = np.concatenate([c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r])
        return dvec / d[:, None], gp, gw

    def fg(v, lam_p, lam_w, mu):
        unit, gp, gw = gaps(v)
        sp = np.maximum(0, lam_p - mu * gp)
        sw = np.maximum(0, lam_w - mu * gw)
        f = -v[2 * n:].sum() + ((sp ** 2).sum() + (sw ** 2).sum()) / (2 * mu)
        ap, aw = -sp, -sw
        gc = np.zeros((n, 2))
        gr = -np.ones(n)
        np.add.at(gc, I, ap[:, None] * unit)
        np.add.at(gc, J, -ap[:, None] * unit)
        np.add.at(gr, I, -ap)
        np.add.at(gr, J, -ap)
        a0, a1, a2, a3 = aw.reshape(4, n)
        gc[:, 0] += a0 - a2
        gc[:, 1] += a1 - a3
        gr -= a0 + a1 + a2 + a3
        return f, np.concatenate([gc.ravel(), gr])

    lam_p, lam_w, mu = np.zeros(len(I)), np.zeros(4 * n), 10.0
    for _ in range(8):
        x = minimize(fg, x, args=(lam_p, lam_w, mu), jac=True, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 200}).x
        _, gp, gw = gaps(x)
        lam_p = np.maximum(0, lam_p - mu * gp)
        lam_w = np.maximum(0, lam_w - mu * gw)
        mu *= 3
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
