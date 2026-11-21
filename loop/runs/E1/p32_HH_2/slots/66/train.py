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
POOL = 20
KEEP = 3
HOP_START = 0.4
HOP_SD = 0.04


def _relax(p, n):
    """Cheap penalty energy minimisation of the centres at a fixed trial radius; returns (energy, centres)."""
    iu = np.triu_indices(n, 1)
    r0 = 0.5 / np.sqrt(n)

    def energy(p):
        c = p.reshape(n, 2)
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1) + 1e-12)
        ov = np.maximum(0, 2 * r0 - d)
        lo = np.maximum(0, r0 - c)
        hi = np.maximum(0, c - (1 - r0))
        return (ov ** 2).sum() + (lo ** 2).sum() + (hi ** 2).sum()

    res = minimize(energy, p, method="L-BFGS-B", options={"maxiter": 200})
    return res.fun, res.x


def _barrier_obj(v, n, iu, mu):
    """Objective -sum(r) - mu * sum(log(slack)) with analytic gradient; slacks are pair gaps and wall gaps."""
    c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
    dx = c[iu[0]] - c[iu[1]]
    d = np.sqrt((dx ** 2).sum(1))
    s = d - r[iu[0]] - r[iu[1]]
    w = np.stack([c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r])
    if s.min() <= 0 or w.min() <= 0:
        return 1e12, np.zeros_like(v)
    f = -r.sum() - mu * (np.log(s).sum() + np.log(w).sum())
    gc, gr = np.zeros((n, 2)), -np.ones(n)
    a = mu / s
    u = (dx / d[:, None]) * a[:, None]
    np.add.at(gc, iu[0], -u)
    np.add.at(gc, iu[1], u)
    np.add.at(gr, iu[0], a)
    np.add.at(gr, iu[1], a)
    b = mu / w
    gc[:, 0] += -b[0] + b[2]
    gc[:, 1] += -b[1] + b[3]
    gr += b.sum(0)
    return f, np.concatenate([gc.ravel(), gr])


def _barrier_start(n, p):
    """Log-barrier continuation (mu = 1e-2, then /10 for 4 stages) from a small feasible start."""
    iu = np.triu_indices(n, 1)
    v = np.concatenate([np.clip(p, 1e-3, 1 - 1e-3), np.full(n, 1e-3)])
    bounds = [(0, 1)] * (2 * n) + [(1e-6, None)] * n
    mu = 1e-2
    for _ in range(4):
        res = minimize(_barrier_obj, v, args=(n, iu, mu), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200})
        if res.fun >= 1e11:
            break
        v = res.x
        mu /= 10
    return v[:2 * n], v[2 * n:]


def _solve_once(n, centers0):
    c0, r0 = _barrier_start(n, centers0)
    x0 = np.concatenate([c0, r0])
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


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < HOP_START * TIME_FRACTION * time_limit:   # random restarts first
        pool = sorted((_relax(rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), n) for _ in range(POOL)), key=lambda t: t[0])
        for _, p in pool[:KEEP]:
            c, r = _solve_once(n, p)
            c = np.clip(c, 0, 1)
            r = _repair(c, r)
            if best is None or r.sum() > best[1].sum():
                best = (c, r)
    budget = TIME_FRACTION * time_limit
    while time.process_time() - t0 < budget:   # basin hopping from the best packing
        c, r = best
        p = (c + rng.normal(0, HOP_SD, c.shape)).ravel()
        c, r = _solve_once(n, p)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if r.sum() > best[1].sum():
            best = (c, r)
    return best
