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
    iu = np.triu_indices(n, 1)
    if start is None:
        def cons_eq(v):
            c, r = v[:2 * n].reshape(n, 2), v[-1]
            d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - 2 * r
            return np.concatenate([d, (c - r).ravel(), (1 - c - r).ravel()])

        y0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), [INIT_RADIUS]])
        y = minimize(lambda v: -v[-1], y0, method="SLSQP", constraints={"type": "ineq", "fun": cons_eq},
                     options={"maxiter": 50}).x
        x0 = np.concatenate([np.clip(y[:2 * n], 0.01, 0.99), np.full(n, max(y[-1], 0.01))])
    else:
        c0 = np.clip(start[0] + rng.normal(0, 0.02, start[0].shape), 0.01, 0.99)
        r0 = start[1]
        x0 = np.concatenate([c0.ravel(), r0])
        d0 = np.sqrt(((c0[iu[0]] - c0[iu[1]]) ** 2).sum(1)) - r0[iu[0]] - r0[iu[1]]
        keep = d0 < 0.15
        iu = (iu[0][keep], iu[1][keep])

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    def run(x, it):
        return minimize(lambda v: -v[2 * n:].sum(), x, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                        options={"maxiter": it})

    if start is not None:
        res = run(x0, MAX_ITER)
        return res.x[:2 * n].reshape(n, 2), res.x[2 * n:]
    res = run(x0, 80)
    x1 = res.x.copy()
    small = x1[2 * n:] < np.median(x1[2 * n:])
    cj = x1[:2 * n].reshape(n, 2)
    cj[small] = np.clip(cj[small] + rng.normal(0, 0.005, (int(small.sum()), 2)), 0.01, 0.99)
    res2 = run(x1, 120)
    best = None
    for rr in (res, res2):
        c = np.clip(rr.x[:2 * n].reshape(n, 2), 0, 1)
        r = rr.x[2 * n:].copy()
        s = _repair(c, r.copy()).sum()
        if best is None or s > best[0]:
            best = (s, c, r)
    return best[1], best[2]


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
        late = best is not None and time.process_time() - t0 > 0.4 * time_limit
        c, r = _solve_once(n, rng, best if late else None)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
