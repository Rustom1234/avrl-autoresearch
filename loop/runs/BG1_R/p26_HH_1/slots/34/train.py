"""train.py: packs n circles into the unit square so that the sum of their radii is as large as possible.

The harness calls solve(n, seed, time_limit) and checks the returned packing exactly.
Returns (centers, radii): centers is an (n, 2) array, radii an (n,) array. Every circle must lie inside
[0, 1] x [0, 1] and no two circles may overlap.
"""
import time

import numpy as np
from scipy.optimize import minimize, linprog

# Settings
TIME_FRACTION = 0.8
INIT_LOW = 0.1
MAX_ITER = 200
INIT_RADIUS = 0.05
SHRINK = 1e-9


def _solve_once(n, rng, c0=None):
    if c0 is None:
        c0 = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2))
    x0 = np.concatenate([np.ravel(c0), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    # Phase 1: all radii tied to one scalar t, maximise t
    def cons_eq(v):
        c, t = v[:2 * n].reshape(n, 2), v[2 * n]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - 2 * t
        return np.concatenate([d, c[:, 0] - t, c[:, 1] - t, 1 - c[:, 0] - t, 1 - c[:, 1] - t, [t]])

    x1 = np.concatenate([x0[:2 * n], [INIT_RADIUS]])
    res1 = minimize(lambda v: -v[2 * n], x1, method="SLSQP", constraints={"type": "ineq", "fun": cons_eq},
                    options={"maxiter": MAX_ITER})
    x2 = np.concatenate([res1.x[:2 * n], np.full(n, max(res1.x[2 * n], 0.0))])

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


def _lp_radii(c):
    """Max sum(r) for fixed centres (LP). Returns radii and a per-circle sensitivity from the duals."""
    n = len(c)
    iu = np.triu_indices(n, 1)
    d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1))
    wall = np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1 - c[:, 0], 1 - c[:, 1]))
    A = np.zeros((len(d), n))
    rows = np.arange(len(d))
    A[rows, iu[0]] = 1
    A[rows, iu[1]] = 1
    res = linprog(-np.ones(n), A_ub=A, b_ub=d, bounds=list(zip(np.zeros(n), np.maximum(wall, 0))),
                  method="highs")
    m = np.abs(res.ineqlin.marginals)
    score = np.zeros(n)
    np.add.at(score, iu[0], m)
    np.add.at(score, iu[1], m)
    score += np.abs(res.lower.marginals)
    return res.x, score


def _largest_hole(c, r):
    """Centre of the largest empty circle on a 100 by 100 grid."""
    g = np.linspace(0, 1, 100)
    gx, gy = np.meshgrid(g, g)
    p = np.stack([gx.ravel(), gy.ravel()], 1)
    free = (np.sqrt(((p[:, None] - c[None]) ** 2).sum(-1)) - r[None]).min(1)
    free = np.minimum(free, np.minimum(p.min(1), 1 - p.max(1)))
    return p[np.argmax(free)]


def _lns_move(c):
    """Remove the lowest-sensitivity circle and reinsert it at the largest hole. Returns new centres."""
    r, score = _lp_radii(c)
    k = np.argmin(score)
    keep = np.arange(len(c)) != k
    cc = c.copy()
    cc[k] = _largest_hole(c[keep], r[keep])
    return cc


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if rng.random() < 0.3:   # multiplier-guided removal and reinsertion, kept only if it improves
            c2, r2 = _solve_once(n, rng, _lns_move(c))
            c2 = np.clip(c2, 0, 1)
            r2 = _repair(c2, r2)
            if r2.sum() > r.sum():
                c, r = c2, r2
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
