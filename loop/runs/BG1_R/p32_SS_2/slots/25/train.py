"""train.py: packs n circles into the unit square so that the sum of their radii is as large as possible.

The harness calls solve(n, seed, time_limit) and checks the returned packing exactly.
Returns (centers, radii): centers is an (n, 2) array, radii an (n,) array. Every circle must lie inside
[0, 1] x [0, 1] and no two circles may overlap.
"""
import time

import numpy as np
from scipy.optimize import minimize, linprog

# Settings
TIME_FRACTION = 0.85
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


def _lp_radii(c):
    n = len(c)
    i, j = np.triu_indices(n, 1)
    d = np.sqrt(((c[i] - c[j]) ** 2).sum(1))
    A = np.zeros((len(i), n))
    A[np.arange(len(i)), i] = 1
    A[np.arange(len(i)), j] = 1
    wall = np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1 - c[:, 0], 1 - c[:, 1]))
    res = linprog(-np.ones(n), A_ub=A, b_ub=d, bounds=[(0, max(w, 0)) for w in wall], method="highs")
    return res.x if res.success else None


def _polish_round(c, r):
    n = len(r)
    iu = np.triu_indices(n, 1)
    d0 = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
    a, b = iu[0][d0 < 0.02], iu[1][d0 < 0.02]

    def cons(v):
        cc, rr = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((cc[a] - cc[b]) ** 2).sum(1)) - rr[a] - rr[b]
        return np.concatenate([d, cc[:, 0] - rr, cc[:, 1] - rr, 1 - cc[:, 0] - rr, 1 - cc[:, 1] - rr, rr])

    x0 = np.concatenate([c.ravel(), r])
    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": 50})
    return np.clip(res.x[:2 * n].reshape(n, 2), 0, 1), res.x[2 * n:]


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    c, r = best
    for _ in range(20):
        if time.process_time() - t0 > 0.97 * time_limit:
            break
        r1 = _lp_radii(c)
        if r1 is None:
            break
        c2, r2 = _polish_round(c, r1)
        r2 = _repair(c2, r2)
        if r2.sum() > r.sum():
            c, r = c2, r2
    r3 = _lp_radii(c)
    if r3 is not None:
        r3 = _repair(c, r3)
        if r3.sum() > r.sum():
            r = r3
    return c, r
