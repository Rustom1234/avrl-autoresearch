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


def _lp_polish(c, r, max_iter=100):
    """Sequential LP: linearise all constraints at (c, r), maximise the radius sum in a trust region."""
    n = len(r)
    iu = np.triu_indices(n, 1)
    k = np.arange(n)
    m = len(iu[0])
    p = np.arange(m)
    tr = 1e-3
    for _ in range(max_iter):
        diff = c[iu[0]] - c[iu[1]]
        d = np.sqrt((diff ** 2).sum(1))
        u = diff / d[:, None]
        A = np.zeros((m + 4 * n, 3 * n))
        b = np.empty(m + 4 * n)
        A[p, 2 * iu[0]] = -u[:, 0]
        A[p, 2 * iu[0] + 1] = -u[:, 1]
        A[p, 2 * iu[1]] = u[:, 0]
        A[p, 2 * iu[1] + 1] = u[:, 1]
        A[p, 2 * n + iu[0]] = 1
        A[p, 2 * n + iu[1]] = 1
        b[:m] = d - r[iu[0]] - r[iu[1]]
        for q, (ax, s) in enumerate([(0, -1), (1, -1), (0, 1), (1, 1)]):
            R = m + q * n + k
            A[R, 2 * k + ax] = s
            A[R, 2 * n + k] = 1
            b[R] = c[:, ax] - r if s < 0 else 1 - c[:, ax] - r
        lo = np.concatenate([np.full(2 * n, -tr), np.maximum(-tr, -r)])
        cost = np.concatenate([np.zeros(2 * n), -np.ones(n)])
        res = linprog(cost, A_ub=A, b_ub=b, bounds=np.column_stack([lo, np.full(3 * n, tr)]), method="highs")
        if res.status != 0:
            break
        cn = np.clip(c + res.x[:2 * n].reshape(n, 2), 0, 1)
        rn = _repair(cn, r + res.x[2 * n:])
        if rn.sum() - r.sum() > 1e-12:
            c, r = cn, rn
        else:
            tr *= 0.3
            if tr < 1e-8:
                break
    return c, r


def solve(n, seed, time_limit):
    rng, t0 = np.random.default_rng(seed), time.process_time()
    budget = TIME_FRACTION * time_limit
    cur = best = None
    while time.process_time() - t0 < budget:
        if cur is None:
            c, r = _solve_once(n, rng)
        else:
            frac = (time.process_time() - t0) / budget
            T = 0.01 * (1e-4 / 0.01) ** frac
            c0 = np.clip(cur[0] + rng.normal(0, 0.05, (n, 2)), 0, 1)
            c, r = _solve_once(n, rng, np.concatenate([c0.ravel(), cur[1]]))
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if cur is None or r.sum() >= cur[1].sum() or np.exp((r.sum() - cur[1].sum()) / T) > rng.uniform():
            cur = (c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return _lp_polish(*best)
