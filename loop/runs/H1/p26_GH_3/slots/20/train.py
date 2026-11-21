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


def _hex_init(n, rng):
    """Hexagonal grid of candidate centres, randomly shifted, with jittered large starting radii."""
    m = int(np.ceil(np.sqrt(n))) + 1
    h = 1.0 / m
    dy = h * np.sqrt(3) / 2
    rows = int(1 / dy) + 1
    g = np.array([(j * h + 0.5 * h * (i % 2), i * dy) for i in range(rows) for j in range(m + 1)])
    g = g[(g[:, 0] <= 1) & (g[:, 1] <= 1)]
    g = g[rng.permutation(len(g))[:n]]
    c = np.clip(g + rng.uniform(-0.5 * h, 0.5 * h, 2), INIT_LOW, 1 - INIT_LOW)
    r = 0.5 * h * rng.uniform(0.8, 1.2, n)
    return c, r


def _soft_relax(n, rng):
    """Soft-body relaxation: maximise total radius minus spring energy of overlaps and wall contact,
    hardening the springs (increasing stiffness k) over several stages."""
    iu = np.triu_indices(n, 1)
    c0, r0 = _hex_init(n, rng)
    v = np.concatenate([c0.ravel(), r0])

    def energy(v, k):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        dv = c[iu[0]] - c[iu[1]]
        d = np.sqrt((dv ** 2).sum(1)) + 1e-12
        ov = np.maximum(r[iu[0]] + r[iu[1]] - d, 0)
        a = np.maximum(r[:, None] - c, 0)
        b = np.maximum(r[:, None] + c - 1, 0)
        E = -r.sum() + k * ((ov ** 2).sum() + (a ** 2).sum() + (b ** 2).sum())
        g = (2 * ov / d)[:, None] * dv
        gc = np.zeros((n, 2))
        gr = -np.ones(n)
        np.add.at(gc, iu[0], -k * g)
        np.add.at(gc, iu[1], k * g)
        np.add.at(gr, iu[0], 2 * k * ov)
        np.add.at(gr, iu[1], 2 * k * ov)
        gc += k * (-2 * a + 2 * b)
        gr += k * (2 * a.sum(1) + 2 * b.sum(1))
        return E, np.concatenate([gc.ravel(), gr])

    for k in (10.0, 100.0, 1e3, 1e4):
        v = minimize(energy, v, args=(k,), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
    return v


def _solve_once(n, rng):
    x0 = _soft_relax(n, rng)
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
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
