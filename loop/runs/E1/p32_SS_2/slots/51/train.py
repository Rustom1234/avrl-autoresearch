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


def _run(n, x0, maxiter):
    iu = np.triu_indices(n, 1)
    i, j = iu
    P = len(i)
    k = np.arange(P)
    idx = np.arange(n)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[i] - c[j]) ** 2).sum(1)) - r[i] - r[j]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    def jac(v):
        c = v[:2 * n].reshape(n, 2)
        diff = c[i] - c[j]
        u = diff / (np.sqrt((diff ** 2).sum(1)) + 1e-12)[:, None]
        J = np.zeros((P + 5 * n, 3 * n))
        J[k, 2 * i], J[k, 2 * i + 1] = u[:, 0], u[:, 1]
        J[k, 2 * j], J[k, 2 * j + 1] = -u[:, 0], -u[:, 1]
        J[k, 2 * n + i] = -1
        J[k, 2 * n + j] = -1
        for b, (col, s) in enumerate([(0, 1), (1, 1), (0, -1), (1, -1)]):
            J[P + b * n + idx, 2 * idx + col] = s
            J[P + b * n + idx, 2 * n + idx] = -1
        J[P + 4 * n + idx, 2 * n + idx] = 1
        return J

    def grad(v):
        g = np.zeros(3 * n)
        g[2 * n:] = -1
        return g

    res = minimize(lambda v: -v[2 * n:].sum(), x0, jac=grad, method="SLSQP",
                   constraints={"type": "ineq", "fun": cons, "jac": jac}, options={"maxiter": maxiter})
    return res.x


def _init(n, rng):
    R = int(rng.integers(5, 8))
    m = int(np.ceil(n / R))
    pts = np.array([((j + 0.5 + 0.5 * (i % 2)) / (m + 0.5), (i + 0.5) / R) for i in range(R) for j in range(m)])
    pts = pts[rng.permutation(len(pts))[:n]] + rng.normal(0, 0.02, (n, 2))
    pts = np.clip(pts, 0.03, 0.97)
    return np.concatenate([pts.ravel(), np.full(n, INIT_RADIUS)])


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
    states = []
    while time.process_time() - t0 < 0.5 * time_limit:   # cheap screening runs
        x = _run(n, _init(n, rng), 40)
        states.append((x[2 * n:].sum(), x))
    states.sort(key=lambda s: -s[0])
    for _, x in states[:8]:
        if best is not None and time.process_time() - t0 > TIME_FRACTION * time_limit:
            break
        x = _run(n, x, 300)
        c = np.clip(x[:2 * n].reshape(n, 2), 0, 1)
        r = _repair(c, x[2 * n:].copy())
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
