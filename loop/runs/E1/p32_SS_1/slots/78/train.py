"""train.py: packs n circles into the unit square so that the sum of their radii is as large as possible.

The harness calls solve(n, seed, time_limit) and checks the returned packing exactly.
Returns (centers, radii): centers is an (n, 2) array, radii an (n,) array. Every circle must lie inside
[0, 1] x [0, 1] and no two circles may overlap.
"""
import time

import numpy as np
from scipy.optimize import minimize

# Settings
TIME_FRACTION = 0.65
POLISH_END = 0.8
INIT_LOW = 0.1
MAX_ITER = 200
INIT_RADIUS = 0.05
SHRINK = 1e-9


def _cross(pool, n, rng):
    a, b = rng.choice(len(pool), 2, replace=False)
    t = rng.uniform(0.3, 0.7)
    (ca, ra), (cb, rb) = pool[a], pool[b]
    ma, mb = ca[:, 0] < t, cb[:, 0] >= t
    c, r = np.concatenate([ca[ma], cb[mb]]), np.concatenate([ra[ma], rb[mb]])
    if len(r) > n:
        k = rng.choice(len(r), n, replace=False)
        c, r = c[k], r[k]
    elif len(r) < n:
        m = n - len(r)
        c = np.vstack([c, rng.uniform(INIT_LOW, 1 - INIT_LOW, (m, 2))])
        r = np.concatenate([r, np.full(m, INIT_RADIUS)])
    return np.concatenate([c.ravel(), r])


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


def _slp(c, r, t_end):
    """Sequential LP: maximise sum(dr) on linearised constraints inside a trust region."""
    from scipy.optimize import linprog
    n = len(r)
    i, j = np.triu_indices(n, 1)
    tr = 0.01
    I = np.eye(n)
    Z = np.zeros((n, n))
    while time.process_time() < t_end and tr > 1e-7:
        diff = c[i] - c[j]
        d = np.sqrt((diff ** 2).sum(1))
        sel = d - r[i] - r[j] < 4 * tr
        a, b, u, dd = i[sel], j[sel], diff[sel] / d[sel, None], d[sel]
        m = len(a)
        k = np.arange(m)
        A = np.zeros((m, 3 * n))
        A[k, 2 * a], A[k, 2 * a + 1], A[k, 2 * b], A[k, 2 * b + 1] = -u[:, 0], -u[:, 1], u[:, 0], u[:, 1]
        A[k, 2 * n + a] = 1
        A[k, 2 * n + b] = 1
        rhs = dd - r[a] - r[b]
        W, wb = [], []
        for ax in range(2):
            E = np.zeros((n, 3 * n))
            E[:, ax:2 * n:2] = -I
            E[:, 2 * n:] = I
            W.append(E)
            wb.append(c[:, ax] - r)
            E = np.zeros((n, 3 * n))
            E[:, ax:2 * n:2] = I
            E[:, 2 * n:] = I
            W.append(E)
            wb.append(1 - c[:, ax] - r)
        res = linprog(np.concatenate([np.zeros(2 * n), -np.ones(n)]), A_ub=np.vstack([A] + W),
                      b_ub=np.maximum(np.concatenate([rhs] + wb), 0), bounds=[(-tr, tr)] * (3 * n),
                      method="highs")
        if res.status != 0:
            tr /= 2
            continue
        c2 = np.clip(c + res.x[:2 * n].reshape(n, 2), 0, 1)
        r2 = _repair(c2, np.maximum(r + res.x[2 * n:], 0))
        if r2.sum() > r.sum():
            c, r = c2, r2
        else:
            tr /= 2
    return c, r


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    pool = []
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # restarts until the time is used
        x0 = _cross(pool, n, rng) if len(pool) >= 2 and rng.random() < 0.7 else None
        c, r = _solve_once(n, rng, x0)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        pool.append((c, r))
        pool.sort(key=lambda p: -p[1].sum())
        pool = pool[:5]
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    c, r = _slp(best[0], best[1], t0 + POLISH_END * time_limit)
    if r.sum() > best[1].sum():
        best = (c, r)
    return best
