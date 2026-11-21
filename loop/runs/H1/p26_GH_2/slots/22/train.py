"""train.py: packs n circles into the unit square so that the sum of their radii is as large as possible.

The harness calls solve(n, seed, time_limit) and checks the returned packing exactly.
Returns (centers, radii): centers is an (n, 2) array, radii an (n,) array. Every circle must lie inside
[0, 1] x [0, 1] and no two circles may overlap.
"""
import time

import numpy as np
from scipy.optimize import linprog, minimize

# Settings
TIME_FRACTION = 0.8
INIT_LOW = 0.1
MAX_ITER = 200
INIT_RADIUS = 0.05
SHRINK = 1e-9


def _solve_once(n, rng):
    x = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)
    bounds = [(0, 1)] * (3 * n)   # centres and radii are box-bounded
    for lam in (10.0, 1e2, 1e3, 1e4, 1e5):   # increasing penalty weight on constraint violations

        def fg(v, lam=lam):
            c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
            diff = c[iu[0]] - c[iu[1]]
            d = np.sqrt((diff ** 2).sum(1)) + 1e-12
            p = np.maximum(0, r[iu[0]] + r[iu[1]] - d)
            f = -r.sum() + lam * (p ** 2).sum()
            gr = -np.ones(n)
            gc = np.zeros((n, 2))
            w = 2 * lam * p
            np.add.at(gr, iu[0], w)
            np.add.at(gr, iu[1], w)
            u = (diff / d[:, None]) * w[:, None]
            np.add.at(gc, iu[0], -u)
            np.add.at(gc, iu[1], u)
            for axis, sgn, off in ((0, 1, 0), (1, 1, 0), (0, -1, 1), (1, -1, 1)):
                q = np.maximum(0, r - (off + sgn * c[:, axis]))   # wall: r <= x, r <= y, r <= 1-x, r <= 1-y
                f += lam * (q ** 2).sum()
                gr += 2 * lam * q
                gc[:, axis] += -2 * lam * q * sgn
            return f, np.concatenate([gc.ravel(), gr])

        x = minimize(fg, x, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": MAX_ITER}).x
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


def _sca(c, r, iters=60, delta=0.05):
    """Successive convex approximation: linearise each distance (a lower bound by convexity), solve an LP, keep improving steps."""
    n = len(r)
    I, J = np.triu_indices(n, 1)
    m = len(I)
    k = np.arange(m)
    w = m + 4 * np.arange(n)
    xi, yi, ri = 2 * np.arange(n), 2 * np.arange(n) + 1, 2 * n + np.arange(n)
    best_c, best_r = c, _repair(c, r)
    best = best_r.sum()
    cost = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    for _ in range(iters):
        diff = best_c[I] - best_c[J]
        d = np.sqrt((diff ** 2).sum(1)) + 1e-12
        u = diff / d[:, None]
        A = np.zeros((m + 4 * n, 3 * n))
        b = np.zeros(m + 4 * n)
        A[k, 2 * I] = -u[:, 0]
        A[k, 2 * I + 1] = -u[:, 1]
        A[k, 2 * J] = u[:, 0]
        A[k, 2 * J + 1] = u[:, 1]
        A[k, 2 * n + I] = 1
        A[k, 2 * n + J] = 1
        A[w, xi] = -1
        A[w, ri] = 1
        A[w + 1, yi] = -1
        A[w + 1, ri] = 1
        A[w + 2, xi] = 1
        A[w + 2, ri] = 1
        b[w + 2] = 1
        A[w + 3, yi] = 1
        A[w + 3, ri] = 1
        b[w + 3] = 1
        lo = np.concatenate([np.maximum(0, best_c.ravel() - delta), np.zeros(n)])
        hi = np.concatenate([np.minimum(1, best_c.ravel() + delta), np.full(n, 0.5)])
        res = linprog(cost, A_ub=A, b_ub=b, bounds=list(zip(lo, hi)), method="highs")
        if res.status != 0:
            break
        cn = np.clip(res.x[:2 * n].reshape(n, 2), 0, 1)
        rn = _repair(cn, res.x[2 * n:])
        if rn.sum() > best:
            best, best_c, best_r = rn.sum(), cn, rn
            delta = min(2 * delta, 0.1)
        else:
            delta *= 0.5
            if delta < 1e-7:
                break
    return best_c, best_r


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        c, r = _sca(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
