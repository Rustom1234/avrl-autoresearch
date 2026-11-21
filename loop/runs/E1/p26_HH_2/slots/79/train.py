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
EARLY_STOPS = 0  # debug: number of cap continuations that skipped to cap 1.0


def _solve_once(n, rng):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def cons(v, cap):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r, cap - r])

    def n_contacts(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        w = np.concatenate([c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r])
        return int((np.abs(d) < 1e-6).sum() + (np.abs(w) < 1e-6).sum())

    global EARLY_STOPS
    caps = list(np.arange(0.05, 0.4 + 1e-9, 0.02)) + [1.0]
    x, prev, i = x0, None, 0
    while i < len(caps):
        cap = caps[i]
        res = minimize(lambda v: -v[2 * n:].sum(), x, method="SLSQP",
                       constraints={"type": "ineq", "fun": lambda v, cap=cap: cons(v, cap)},
                       options={"maxiter": MAX_ITER})
        x = res.x
        cur = (n_contacts(x), x[2 * n:].sum())
        if prev is not None and cap < 1.0 and cur[0] == prev[0] and abs(cur[1] - prev[1]) < 1e-4:
            EARLY_STOPS += 1
            i, prev = len(caps) - 1, None   # skip remaining caps, go straight to cap 1.0
            continue
        prev = cur
        i += 1
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
