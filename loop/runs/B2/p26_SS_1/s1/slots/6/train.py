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


def _rows_init(n, rng):
    k = rng.choice([5, 6])
    cnt = np.full(k, n // k)
    cnt[rng.choice(k, n - cnt.sum(), replace=False)] += 1
    pts = []
    for j, m in enumerate(cnt):
        w = 0.8 / m
        x = 0.1 + (np.arange(m) + 0.5 + 0.5 * (j % 2)) * w
        pts.append(np.stack([x, np.full(m, 0.1 + 0.8 * (j + 0.5) / k)], 1))
    c = np.concatenate(pts)
    return np.clip(c + rng.normal(0, 0.03, c.shape), 0.01, 0.99)


def _maximin(n, rng):
    cand = rng.uniform(0.02, 0.98, (400, 2))
    idx = [int(rng.integers(400))]
    d = np.linalg.norm(cand - cand[idx[0]], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(cand - cand[i], axis=1))
    return cand[idx]


def _fresh(n, rng):
    if rng.random() < 0.5:
        return _rows_init(n, rng), np.full(n, 0.06)
    return _maximin(n, rng), np.full(n, INIT_RADIUS)


def _solve_once(n, rng, start=None):
    if start is None:
        c0, r0 = _fresh(n, rng)
    else:
        c0, r0 = start[0].copy(), start[1].copy()
        if rng.random() < 0.5:
            k = rng.integers(2, 4)
            idx = rng.choice(n, k, replace=False)
            c0[idx] = rng.uniform(INIT_LOW, 1 - INIT_LOW, (k, 2))
            r0[idx] = INIT_RADIUS * 0.5
        else:
            c0 = np.clip(c0 + rng.normal(0, 0.02, c0.shape), 0.01, 0.99)
    x0 = np.concatenate([c0.ravel(), r0])
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
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        if best is not None and time.process_time() - t0 > 0.3 * TIME_FRACTION * time_limit:
            c, r = _solve_once(n, rng, best)
        else:
            c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
