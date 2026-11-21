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


def _quad_start(n, rng):
    """Split the circles into four near-equal groups, pack each equal-radius inside its own quadrant."""
    sizes = np.full(4, n // 4) + (np.arange(4) < n % 4)[rng.permutation(4)]
    cs, rs = [], []
    for k, m in enumerate(sizes):
        ox, oy = 0.5 * (k % 2), 0.5 * (k // 2)
        iu = np.triu_indices(m, 1)

        def cons(v, iu=iu, m=m, ox=ox, oy=oy):
            c, t = v[:2 * m].reshape(m, 2), v[2 * m]
            d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - 2 * t
            return np.concatenate([d, c[:, 0] - ox - t, c[:, 1] - oy - t, ox + 0.5 - c[:, 0] - t,
                                   oy + 0.5 - c[:, 1] - t, [t]])

        pts = np.c_[ox + rng.uniform(0.1, 0.4, m), oy + rng.uniform(0.1, 0.4, m)]
        x0 = np.concatenate([pts.ravel(), [INIT_RADIUS]])
        res = minimize(lambda v, m=m: -v[2 * m], x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                       options={"maxiter": MAX_ITER})
        cs.append(res.x[:2 * m].reshape(m, 2))
        rs.append(np.full(m, max(res.x[2 * m], 0.0)))
    return np.concatenate([np.vstack(cs).ravel(), np.concatenate(rs)])


def _solve_once(n, rng, start=None):
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    if start is None:
        start = _quad_start(n, rng)
    x2 = start

    # Phase 2: release the radii and run the full free-radius SLSQP
    res = minimize(lambda v: -v[2 * n:].sum(), x2, method="SLSQP", constraints={"type": "ineq", "fun": cons},
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
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        if best is None:   # first start: a random restart
            c, r = _solve_once(n, rng)
        else:              # local kick: re-randomise the centres in a disc, then re-optimise everything
            c, r = best[0].copy(), best[1].copy()
            p, R = rng.uniform(0, 1, 2), rng.uniform(0.15, 0.35)
            m = np.hypot(*(c - p).T) < R
            k = m.sum()
            ang, rad = rng.uniform(0, 2 * np.pi, k), R * np.sqrt(rng.uniform(0, 1, k))
            c[m] = np.clip(p + np.c_[rad * np.cos(ang), rad * np.sin(ang)], 0, 1)
            r[m] = 0
            c, r = _solve_once(n, rng, np.concatenate([c.ravel(), r]))
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
