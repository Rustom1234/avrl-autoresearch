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
STALL_ITER = 30


class _Stall(Exception):
    pass


def _slsqp(obj, x0, cons, n):
    """SLSQP that stops early when the sum of radii has not improved for STALL_ITER iterations."""
    st = {"best": -np.inf, "last": 0, "k": 0, "x": x0}

    def cb(v):
        st["k"] += 1
        st["x"] = v.copy()
        if v[2 * n:].sum() > st["best"] + 1e-6:
            st["best"], st["last"] = v[2 * n:].sum(), st["k"]
        if st["k"] - st["last"] > STALL_ITER:
            raise _Stall

    try:
        return minimize(obj, x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                        callback=cb, options={"maxiter": MAX_ITER}).x
    except _Stall:
        return st["x"]


def _solve_once(n, rng):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)
    s0 = rng.uniform(0.7, 1.4)   # elliptical stretch used only in the initial search

    def make_cons(s):
        def cons(v):
            c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
            d = np.sqrt((c[iu[0], 0] - c[iu[1], 0]) ** 2 + (s * (c[iu[0], 1] - c[iu[1], 1])) ** 2) - r[iu[0]] - r[iu[1]]
            return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])
        return cons

    def obj(v):
        return -v[2 * n:].sum()

    x = _slsqp(obj, x0, make_cons(s0), n)
    # round to true circles: re-solve with the real circle constraints, starting from the stretched solution
    x = _slsqp(obj, x, make_cons(1.0), n)
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


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
