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


def _solve_once(n, rng, init=None, init_r=None, iters=MAX_ITER):
    if init is None:
        base = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2))
    elif init_r is None:
        base = np.clip(init + rng.normal(0, 0.03, (n, 2)), 0, 1)
    else:
        base = init
    r0 = np.full(n, INIT_RADIUS) if init_r is None else init_r
    x0 = np.concatenate([base.ravel(), r0])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": iters})
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


def _lloyd(c, r, grid, iters=5):
    """Weighted (power-diagram) Lloyd iterations on the centres, radii fixed."""
    n = len(r)
    c = c.copy()
    for _ in range(iters):
        lab = np.argmin(((grid[:, None, :] - c[None]) ** 2).sum(-1) - r[None] ** 2, axis=1)
        cnt = np.bincount(lab, minlength=n)
        sx = np.bincount(lab, weights=grid[:, 0], minlength=n)
        sy = np.bincount(lab, weights=grid[:, 1], minlength=n)
        ok = cnt > 0
        c[ok] = np.column_stack([sx[ok] / cnt[ok], sy[ok] / cnt[ok]])
    return c


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    g = (np.arange(200) + 0.5) / 200
    X, Y = np.meshgrid(g, g)
    grid = np.column_stack([X.ravel(), Y.ravel()])
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        guided = best is not None and time.process_time() - t0 > 0.5 * TIME_FRACTION * time_limit and rng.random() < 0.5
        c, r = _solve_once(n, rng, best[0] if guided else None)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        cands = [(c, r)]
        c2 = np.clip(_lloyd(c, r, grid), 0, 1)
        r2 = _repair(c2, r)
        c3, r3 = _solve_once(n, rng, c2, r2, 100)
        c3 = np.clip(c3, 0, 1)
        cands.append((c3, _repair(c3, r3)))
        for cc, rr in cands:
            if best is None or rr.sum() > best[1].sum():
                best = (cc, rr)
    return best
