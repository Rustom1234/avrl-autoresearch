"""train.py: packs n circles into the unit square so that the sum of their radii is as large as possible.

The harness calls solve(n, seed, time_limit) and checks the returned packing exactly.
Returns (centers, radii): centers is an (n, 2) array, radii an (n,) array. Every circle must lie inside
[0, 1] x [0, 1] and no two circles may overlap.
"""
import time

import numpy as np
from scipy.optimize import minimize, basinhopping

# Settings
TIME_FRACTION = 0.8
INIT_LOW = 0.1
MAX_ITER = 200
INIT_RADIUS = 0.05
SHRINK = 1e-9
MU = 1e3
STEP = 0.05
TEMP = 0.01


def _solve_once(n, rng, x0=None):
    if x0 is None:
        x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
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


class _Out(Exception):
    pass


def _energy(v, n, iu):
    """Penalty energy -sum(r) + MU * squared violations of pair, wall and nonnegativity terms, with gradient."""
    c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
    i, j = iu
    diff = c[i] - c[j]
    d = np.sqrt((diff ** 2).sum(1)) + 1e-12
    p = np.maximum(0, r[i] + r[j] - d)
    a = np.maximum(0, r - c[:, 0])
    b = np.maximum(0, r - c[:, 1])
    e = np.maximum(0, c[:, 0] + r - 1)
    f = np.maximum(0, c[:, 1] + r - 1)
    q = np.maximum(0, -r)
    E = -r.sum() + MU * ((p ** 2).sum() + (a ** 2 + b ** 2 + e ** 2 + f ** 2 + q ** 2).sum())
    gc = np.zeros((n, 2))
    gr = np.zeros(n)
    u = diff / d[:, None]
    np.add.at(gc, i, -2 * p[:, None] * u)
    np.add.at(gc, j, 2 * p[:, None] * u)
    np.add.at(gr, i, 2 * p)
    np.add.at(gr, j, 2 * p)
    gc[:, 0] += -2 * a + 2 * e
    gc[:, 1] += -2 * b + 2 * f
    gr += 2 * (a + b + e + f) - 2 * q
    return E, np.concatenate([MU * gc.ravel(), MU * gr - 1])


def solve(n, seed, time_limit):
    rng, t0 = np.random.default_rng(seed), time.process_time()
    budget = TIME_FRACTION * time_limit
    iu = np.triu_indices(n, 1)
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    best = [np.inf, x0]

    def callback(x, f, accept):
        if f < best[0]:
            best[:] = [f, x.copy()]
        if time.process_time() - t0 > budget:
            raise _Out

    def take_step(x):
        return np.clip(x + rng.uniform(-STEP, STEP, x.shape), 0, 1)

    try:
        basinhopping(_energy, x0, niter=10 ** 9, T=TEMP, take_step=take_step, callback=callback,
                     minimizer_kwargs={"method": "L-BFGS-B", "jac": True, "bounds": [(0, 1)] * (3 * n),
                                       "args": (n, iu)})
    except _Out:
        pass
    c, r = _solve_once(n, rng, best[1])
    c = np.clip(c, 0, 1)
    return c, _repair(c, r)
