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


def _splice(pa, pb, n, rng):
    """Deaven-Ho cut-and-splice: take pa on one side of a random line, pb on the other, then fix the count at n."""
    p = rng.uniform(0.2, 0.8, 2)
    th = rng.uniform(0, 2 * np.pi)
    nv = np.array([np.cos(th), np.sin(th)])
    sa = (pa - p) @ nv < 0
    sb = (pb - p) @ nv >= 0
    c = np.vstack([pa[sa], pb[sb]])
    side = np.concatenate([np.zeros(sa.sum(), int), np.ones(sb.sum(), int)])
    while len(c) > n:
        big = int(np.argmax(np.bincount(side, minlength=2)))
        idx = rng.choice(np.flatnonzero(side == big))
        keep = np.arange(len(c)) != idx
        c, side = c[keep], side[keep]
    while len(c) < n:
        deficit = int(np.argmin(np.bincount(side, minlength=2)))
        cand = rng.uniform(0, 1, (4000, 2))
        mask = (((cand - p) @ nv) < 0) == (deficit == 0)
        cand = cand[mask] if mask.any() else cand
        d = np.sqrt(((cand[:, None] - c[None]) ** 2).sum(-1)).min(1) if len(c) else np.ones(len(cand))
        c = np.vstack([c, cand[np.argmax(d)]])
        side = np.append(side, deficit)
    return c


def _solve_once(n, rng, start=None):
    if start is None:
        x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    else:
        x0 = np.concatenate([start.ravel(), np.full(n, INIT_RADIUS)])
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


def _update_pool(pool, item):
    """Keep the 6 best packings with distinct radius sums."""
    pool = sorted(pool + [item], key=lambda q: -q[1].sum())
    uniq = []
    for q in pool:
        if all(abs(q[1].sum() - u[1].sum()) > 1e-9 for u in uniq):
            uniq.append(q)
    return uniq[:6]


def solve(n, seed, time_limit):
    rng, t0, best, pool, k = np.random.default_rng(seed), time.process_time(), None, [], 0
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        start = None
        if k >= 2 and len(pool) >= 2:
            i, j = rng.choice(len(pool), 2, replace=False)
            start = _splice(pool[i][0], pool[j][0], n, rng)
        c, r = _solve_once(n, rng, start)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        k += 1
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
        pool = _update_pool(pool, (c, r))
    return best
