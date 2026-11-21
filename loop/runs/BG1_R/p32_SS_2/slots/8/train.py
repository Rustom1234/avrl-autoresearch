"""train.py: packs n circles into the unit square so that the sum of their radii is as large as possible.

The harness calls solve(n, seed, time_limit) and checks the returned packing exactly.
Returns (centers, radii): centers is an (n, 2) array, radii an (n,) array. Every circle must lie inside
[0, 1] x [0, 1] and no two circles may overlap.
"""
import time

import numpy as np
from scipy.optimize import minimize, linprog, NonlinearConstraint

# Settings
TIME_FRACTION = 0.6
INIT_LOW = 0.1
MAX_ITER = 200
INIT_RADIUS = 0.05
SHRINK = 1e-9


def _solve_once(n, rng):
    R = int(rng.integers(5, 8))
    m = int(np.ceil(n / R))
    pts = np.array([((j + 0.5 + 0.5 * (i % 2)) / (m + 0.5), (i + 0.5) / R) for i in range(R) for j in range(m)])
    pts = pts[rng.permutation(len(pts))[:n]] + rng.normal(0, 0.02, (n, 2))
    pts = np.clip(pts, 0.03, 0.97)
    x0 = np.concatenate([pts.ravel(), np.full(n, INIT_RADIUS)])
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


def _jac(v, n, iu):
    c = v[:2 * n].reshape(n, 2)
    i, j = iu
    P = len(i)
    k, a = np.arange(P), np.arange(n)
    dd = c[i] - c[j]
    u = dd / np.sqrt((dd ** 2).sum(1))[:, None]
    J = np.zeros((P + 5 * n, 3 * n))
    J[k, 2 * i], J[k, 2 * i + 1], J[k, 2 * j], J[k, 2 * j + 1] = u[:, 0], u[:, 1], -u[:, 0], -u[:, 1]
    J[k, 2 * n + i] = -1
    J[k, 2 * n + j] = -1
    for q, (col, sgn) in enumerate([(0, 1), (1, 1), (0, -1), (1, -1)]):
        J[P + q * n + a, 2 * a + col] = sgn
        J[P + q * n + a, 2 * n + a] = -1
    J[P + 4 * n + a, 2 * n + a] = 1
    return J


def _refine(c, r, deadline):
    """trust-constr polish, then LP for the radii given the centres, then exact repair."""
    n = len(r)
    iu = np.triu_indices(n, 1)

    def cons(v):
        cc, rr = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((cc[iu[0]] - cc[iu[1]]) ** 2).sum(1)) - rr[iu[0]] - rr[iu[1]]
        return np.concatenate([d, cc[:, 0] - rr, cc[:, 1] - rr, 1 - cc[:, 0] - rr, 1 - cc[:, 1] - rr, rr])

    g = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    x0 = np.concatenate([c.ravel(), r])
    try:
        res = minimize(lambda v: -v[2 * n:].sum(), x0, jac=lambda v: g, method="trust-constr",
                       constraints=NonlinearConstraint(cons, 0, np.inf, jac=lambda v: _jac(v, n, iu)),
                       callback=lambda xk, st: time.process_time() > deadline,
                       options={"maxiter": 500})
        c2 = np.clip(res.x[:2 * n].reshape(n, 2), 0, 1)
    except Exception:
        c2 = c
    d = np.sqrt(((c2[iu[0]] - c2[iu[1]]) ** 2).sum(1))
    A = np.zeros((len(d), n))
    A[np.arange(len(d)), iu[0]] = 1
    A[np.arange(len(d)), iu[1]] = 1
    wall = np.minimum(np.minimum(c2[:, 0], c2[:, 1]), np.minimum(1 - c2[:, 0], 1 - c2[:, 1]))
    lp = linprog(-np.ones(n), A_ub=A, b_ub=d, bounds=list(zip(np.zeros(n), np.maximum(wall, 0))), method="highs")
    if not lp.success:
        return c, r
    r2 = _repair(c2, np.maximum(lp.x - 2e-7, 0))
    return c2, r2


def solve(n, seed, time_limit):
    rng, t0, cands = np.random.default_rng(seed), time.process_time(), []
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        cands.append((r.sum(), c, r))
    cands.sort(key=lambda t: -t[0])
    best = (cands[0][1], cands[0][2])
    for k, (_, c, r) in enumerate(cands[:3]):
        c2, r2 = _refine(c, r, t0 + time_limit * (0.72 + 0.09 * k))
        if r2.sum() > best[1].sum():
            best = (c2, r2)
    return best
