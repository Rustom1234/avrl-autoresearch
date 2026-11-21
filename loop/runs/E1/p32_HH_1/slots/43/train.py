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


def _solve_once(n, rng, init=None):
    base = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2)) if init is None else np.clip(init + rng.normal(0, 0.03, (n, 2)), 0, 1)
    x0 = np.concatenate([base.ravel(), np.full(n, INIT_RADIUS)])
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


def _radii_opt(c, r0):
    """Re-optimise radii for fixed centres with SLSQP, then repair so the packing is valid."""
    n = len(r0)
    iu = np.triu_indices(n, 1)
    D = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))[iu]

    def cons(r):
        return np.concatenate([D - r[iu[0]] - r[iu[1]], c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda r: -r.sum(), r0, jac=lambda r: -np.ones(n), method="SLSQP",
                   constraints={"type": "ineq", "fun": cons}, options={"maxiter": 100})
    return _repair(c, np.maximum(res.x, 0))


def _relocate(c, r, rng, deadline):
    """Hole-targeted relocation: move one random circle to the point of largest clearance, keep if sum improves."""
    n = len(r)
    g = np.linspace(0, 1, 41)
    G = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    for _ in range(40):
        if time.process_time() > deadline:
            break
        i = rng.integers(n)
        co, ro = np.delete(c, i, axis=0), np.delete(r, i)

        def clear(P):
            wall = np.minimum(np.minimum(P[:, 0], P[:, 1]), np.minimum(1 - P[:, 0], 1 - P[:, 1]))
            d = np.sqrt(((P[:, None] - co[None]) ** 2).sum(-1)) - ro[None]
            return np.minimum(wall, d.min(1))

        p0 = G[np.argmax(clear(G))]
        res = minimize(lambda p: -clear(np.clip(p[None], 0, 1))[0], p0, method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-10, "maxfev": 300})
        p = max([np.clip(p0, 0, 1), np.clip(res.x, 0, 1)], key=lambda q: clear(q[None])[0])
        rn = max(clear(p[None])[0], 0)
        cc, rr = c.copy(), r.copy()
        cc[i], rr[i] = p, rn
        rr = _radii_opt(cc, rr)
        if rr.sum() > r.sum() + 1e-9:
            c, r = cc, rr
    return c, r


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        guided = best is not None and time.process_time() - t0 > 0.5 * TIME_FRACTION * time_limit and rng.random() < 0.5
        c, r = _solve_once(n, rng, best[0] if guided else None)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    c, r = _relocate(best[0], best[1], rng, t0 + 0.95 * time_limit)
    return c, r
