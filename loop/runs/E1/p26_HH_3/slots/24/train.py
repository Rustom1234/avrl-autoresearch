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
RESTART_FRACTION = 0.3


def _optimise(n, x0):
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": MAX_ITER})
    return res.x[:2 * n].reshape(n, 2), res.x[2 * n:]


def _solve_once(n, rng):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    return _optimise(n, x0)


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


def _cut_child(cA, rA, cB, rB, n, rng):
    """Cut-line exchange: circles of A on one side of a random line, circles of B on the other, count fixed to n."""
    theta = rng.uniform(0, np.pi)
    nv = np.array([np.cos(theta), np.sin(theta)])
    p = rng.uniform(0.25, 0.75, 2)
    sA, sB = (cA - p) @ nv, (cB - p) @ nv
    c, r = np.vstack([cA, cB]), np.concatenate([rA, rB])
    side = np.concatenate([sA, sB])
    keep = np.concatenate([sA < 0, sB >= 0])
    kept = np.where(keep)[0]
    kept = kept[np.argsort(-np.abs(side[kept]))][:n]          # drop the ones nearest the line
    rest = np.where(~keep)[0]
    rest = rest[np.argsort(np.abs(side[rest]))][:n - len(kept)]   # add the ones nearest the line
    sel = np.concatenate([kept, rest])
    return c[sel], r[sel]


def solve(n, seed, time_limit):
    rng, t0, best, second = np.random.default_rng(seed), time.process_time(), None, None
    budget = TIME_FRACTION * time_limit
    while time.process_time() - t0 < RESTART_FRACTION * budget or best is None:   # short random-restart phase
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            second, best = best, (c, r)
        elif second is None or r.sum() > second[1].sum():
            second = (c, r)
    c, r = best
    step = 0
    while time.process_time() - t0 < budget:   # remove-and-reinsert the smallest circle
        step += 1
        c2, r2 = c.copy(), r.copy()
        i = np.argmin(r2)
        c2[i] = rng.uniform(0, 1, 2)
        r2[i] = 0
        c2, r2 = _optimise(n, np.concatenate([c2.ravel(), r2]))
        c2 = np.clip(c2, 0, 1)
        r2 = _repair(c2, r2)
        if r2.sum() > r.sum():
            c, r = c2, r2
        if second is not None and step % 5 == 0:   # one cut-line child per 5 reinsertion steps
            c3, r3 = _cut_child(c, r, second[0], second[1], n, rng)
            c3, r3 = _optimise(n, np.concatenate([c3.ravel(), r3]))
            c3 = np.clip(c3, 0, 1)
            r3 = _repair(c3, r3)
            second, (c, r) = sorted([(c, r), second, (c3, r3)], key=lambda p: p[1].sum())[1:]
    return c, r
