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
SYM_PROB = 0.5


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


def _relocate(c, r, rng, trials=30):
    """Move the smallest circle to the best of several random centres, scored by the local sum of radii."""
    c, r = c.copy(), r.copy()
    k = int(np.argmin(r))
    best = None
    for _ in range(trials):
        p = rng.uniform(0, 1, 2)
        d = np.sqrt(((c - p) ** 2).sum(1))
        d[k] = np.inf
        rk = max(min(p.min(), (1 - p).min(), (d - r).min()), 0)
        nb = d < rk + r + 0.05
        score = rk + r[nb].sum()
        if best is None or score > best[0]:
            best = (score, p, rk)
    _, p, rk = best
    c[k], r[k] = p, rk
    return c, r


def _solve_sym(n, rng):
    """Optimise under a random symmetry (reflection in x=y or 180-degree rotation) in reduced variables."""
    kind = int(rng.integers(2))   # 0: reflection in x=y, 1: rotation by 180 degrees about (0.5, 0.5)
    k, odd = n // 2, n % 2
    nz = 3 * k + (0 if not odd else (2 if kind == 0 else 1))
    A, b = np.zeros((3 * n, nz)), np.zeros(3 * n)   # full vector [centres.ravel(), radii] = A @ z + b
    for i in range(k):
        A[2 * i, 2 * i] = A[2 * i + 1, 2 * i + 1] = 1
        j = k + i
        if kind == 0:
            A[2 * j, 2 * i + 1] = A[2 * j + 1, 2 * i] = 1
        else:
            A[2 * j, 2 * i] = A[2 * j + 1, 2 * i + 1] = -1
            b[2 * j] = b[2 * j + 1] = 1
        A[2 * n + i, 2 * k + i] = A[2 * n + j, 2 * k + i] = 1
    if odd:
        j = 2 * k
        if kind == 0:
            A[2 * j, 3 * k] = A[2 * j + 1, 3 * k] = 1
            A[2 * n + j, 3 * k + 1] = 1
        else:
            b[2 * j] = b[2 * j + 1] = 0.5
            A[2 * n + j, 3 * k] = 1
    z0 = rng.uniform(INIT_LOW, 1 - INIT_LOW, nz)
    z0[2 * k:3 * k] = INIT_RADIUS
    if odd:
        z0[-1] = INIT_RADIUS
    iu = np.triu_indices(n, 1)

    def cons(z):
        v = A @ z + b
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda z: -(A @ z + b)[2 * n:].sum(), z0, method="SLSQP",
                   constraints={"type": "ineq", "fun": cons}, options={"maxiter": MAX_ITER})
    v = A @ res.x + b
    return v[:2 * n].reshape(n, 2), v[2 * n:]


def _polish(c, r, rng):
    """Unconstrained SLSQP polish from a (possibly symmetric) packing, then repair."""
    c, r = np.clip(c, 0, 1), np.maximum(r, 0)
    c, r = _solve_once(len(r), rng, np.concatenate([c.ravel(), r]))
    c = np.clip(c, 0, 1)
    return c, _repair(c, r)


def solve(n, seed, time_limit):
    rng, t0 = np.random.default_rng(seed), time.process_time()
    c, r = _solve_once(n, rng)
    c = np.clip(c, 0, 1)
    best = (c, _repair(c, r))
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # basin hopping from the incumbent
        bc, br = best
        c0 = np.clip(bc + rng.normal(0, 0.02, bc.shape), 0, 1)
        r0 = np.maximum(br + rng.normal(0, 0.02, br.shape), 0)
        c0, r0 = _relocate(c0, r0, rng)
        c, r = _solve_once(n, rng, np.concatenate([c0.ravel(), r0]))
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if r.sum() > best[1].sum():
            best = (c, r)
        if rng.random() < SYM_PROB:   # symmetry-reduced candidate, polished unconstrained
            sc, sr = _solve_sym(n, rng)
            sc, sr = _polish(sc, sr, rng)
            if sr.sum() > best[1].sum():
                best = (sc, sr)
    return best
