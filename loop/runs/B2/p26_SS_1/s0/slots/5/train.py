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


def _maximin(n, rng):
    """Farthest-point sampling: greedily pick n of 400 random candidates, maximising the minimum distance."""
    cand = rng.uniform(INIT_LOW, 1 - INIT_LOW, (400, 2))
    pick = [int(rng.integers(400))]
    d = np.sqrt(((cand - cand[pick[0]]) ** 2).sum(1))
    for _ in range(n - 1):
        k = int(np.argmax(d))
        pick.append(k)
        d = np.minimum(d, np.sqrt(((cand - cand[k]) ** 2).sum(1)))
    return cand[pick]


def _slsqp(n, x0, maxiter):
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": maxiter})
    return res.x[:2 * n].reshape(n, 2), res.x[2 * n:]


def _solve_once(n, rng):
    c0 = _maximin(n, rng) if rng.random() < 0.5 else rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2))
    return _slsqp(n, np.concatenate([c0.ravel(), np.full(n, INIT_RADIUS)]), MAX_ITER)


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
            # shake all centres, short SLSQP; converge further only if close to the best
            c0 = np.clip(best[0] + rng.normal(0, 0.05, best[0].shape), 0.01, 0.99)
            c, r = _slsqp(n, np.concatenate([c0.ravel(), best[1]]), 60)
            c = np.clip(c, 0, 1)
            r = _repair(c, r)
            if r.sum() >= best[1].sum() - 0.01:
                c, r = _slsqp(n, np.concatenate([c.ravel(), r]), MAX_ITER)
        else:
            c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
