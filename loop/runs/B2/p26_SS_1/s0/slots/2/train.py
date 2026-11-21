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


def _solve_once(n, rng, start=None):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    hot = np.zeros(n, dtype=bool)
    if start is None and rng.random() < 0.5:   # farthest-point (maximin) centres
        cand = rng.uniform(INIT_LOW, 1 - INIT_LOW, (400, 2))
        chosen = [0]
        dmin = np.linalg.norm(cand - cand[0], axis=1)
        for _ in range(n - 1):
            chosen.append(int(np.argmax(dmin)))
            dmin = np.minimum(dmin, np.linalg.norm(cand - cand[chosen[-1]], axis=1))
        x0 = np.concatenate([cand[chosen].ravel(), np.full(n, INIT_RADIUS)])
    if start is not None:
        c0, r0 = start[0].copy(), start[1].copy()
        if rng.random() < 0.5:
            k = rng.integers(2, 4)
            idx = rng.choice(n, k, replace=False)
            c0[idx] = rng.uniform(INIT_LOW, 1 - INIT_LOW, (k, 2))
            r0[idx] = INIT_RADIUS * 0.5
            hot[idx] = True
        else:
            c0 = np.clip(c0 + rng.normal(0, 0.02, c0.shape), 0.01, 0.99)
        x0 = np.concatenate([c0.ravel(), r0])
    iu = np.triu_indices(n, 1)
    if start is not None:   # keep only near pairs, plus all pairs touching a relocated circle
        cc = x0[:2 * n].reshape(n, 2)
        dist = np.sqrt(((cc[iu[0]] - cc[iu[1]]) ** 2).sum(1))
        keep = (dist < 0.45) | hot[iu[0]] | hot[iu[1]]
        iu = (iu[0][keep], iu[1][keep])

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
