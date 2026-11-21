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


def _uniform_start(n, rng):
    return np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])


def _ring_start(n, rng):
    m = min(16, n)
    corners = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    s = np.arange(m) * 4.0 / m
    side = np.floor(s).astype(int)
    f = (s - side)[:, None]
    ring = corners[side] + f * (corners[(side + 1) % 4] - corners[side])
    ring += rng.normal(0, 0.02, ring.shape)
    c = np.vstack([ring, rng.uniform(0.25, 0.75, (n - m, 2))])
    return np.concatenate([c.ravel(), np.full(n, INIT_RADIUS)])


def _power_start(n, rng):
    r = np.arange(1, n + 1) ** -0.5
    r = np.clip(r / r.mean() * INIT_RADIUS, 0.02, 0.1)
    rng.shuffle(r)
    return np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), r])


def _shake_start(best, rng):
    c = best[0]
    A = np.eye(2) + rng.normal(0, 0.05, (2, 2))
    c2 = np.clip(0.5 + (c - 0.5) @ A, 0.05, 0.95)
    return np.concatenate([c2.ravel(), np.full(len(c), INIT_RADIUS)])


def _solve_once(n, rng, x0=None):
    if x0 is None:
        x0 = _uniform_start(n, rng)
    iu = np.triu_indices(n, 1)

    def cons(v, cap):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r, cap - r])

    x = x0
    for cap in list(np.arange(0.05, 0.4 + 1e-9, 0.02)) + [1.0]:
        res = minimize(lambda v: -v[2 * n:].sum(), x, method="SLSQP",
                       constraints={"type": "ineq", "fun": lambda v, cap=cap: cons(v, cap)},
                       options={"maxiter": MAX_ITER})
        x = res.x
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
        u = rng.integers(4)
        if u == 3 and best is None:
            u = 0
        if u == 0:
            x0 = None
        elif u == 1:
            x0 = _ring_start(n, rng)
        elif u == 2:
            x0 = _power_start(n, rng)
        else:
            x0 = _shake_start(best, rng)
        c, r = _solve_once(n, rng, x0)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
