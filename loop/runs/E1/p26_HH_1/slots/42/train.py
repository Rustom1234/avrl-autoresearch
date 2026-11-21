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


def _solve_once(n, rng, start=None):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    # Phase 1: all radii tied to one scalar t, maximise t
    def cons_eq(v):
        c, t = v[:2 * n].reshape(n, 2), v[2 * n]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - 2 * t
        return np.concatenate([d, c[:, 0] - t, c[:, 1] - t, 1 - c[:, 0] - t, 1 - c[:, 1] - t, [t]])

    if start is None:
        x1 = np.concatenate([x0[:2 * n], [INIT_RADIUS]])
        res1 = minimize(lambda v: -v[2 * n], x1, method="SLSQP", constraints={"type": "ineq", "fun": cons_eq},
                        options={"maxiter": MAX_ITER})
        start = np.concatenate([res1.x[:2 * n], np.full(n, max(res1.x[2 * n], 0.0))])
    x2 = start

    # Phase 2: release the radii and run the full free-radius SLSQP
    res = minimize(lambda v: -v[2 * n:].sum(), x2, method="SLSQP", constraints={"type": "ineq", "fun": cons},
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


def _soft_kick(n, c, r, rng):
    """Move the centres along 1 to 3 of the softest nonzero modes of the active-contact Jacobian."""
    iu = np.triu_indices(n, 1)
    diff = c[iu[0]] - c[iu[1]]
    d = np.sqrt((diff ** 2).sum(1))
    act = np.where(d - r[iu[0]] - r[iu[1]] < 1e-7)[0]
    u = diff[act] / d[act, None]
    J = np.zeros((len(act), 2 * n))
    k = np.arange(len(act))
    for q in range(2):
        J[k, 2 * iu[0][act] + q] = u[:, q]
        J[k, 2 * iu[1][act] + q] = -u[:, q]
    rows = [J]
    for i in range(n):
        for q in range(2):
            if c[i, q] - r[i] < 1e-7:
                e = np.zeros(2 * n)
                e[2 * i + q] = 1
                rows.append(e[None])
            if 1 - c[i, q] - r[i] < 1e-7:
                e = np.zeros(2 * n)
                e[2 * i + q] = -1
                rows.append(e[None])
    J = np.vstack(rows)
    w, V = np.linalg.eigh(J.T @ J)
    soft = V[:, w > 1e-9][:, :6]
    if soft.shape[1] == 0:
        return c.copy()
    m = rng.choice(soft.shape[1], size=rng.integers(1, 4), replace=False)
    v = soft[:, m] @ rng.choice([-1.0, 1.0], len(m))
    v *= rng.uniform(0.02, 0.05) / np.abs(v).max()
    return np.clip(c + v.reshape(n, 2), 0, 1)


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        if best is None:   # first start: a random restart
            c, r = _solve_once(n, rng)
        else:              # local kick: soft-mode perturbation of the incumbent, then re-optimise everything
            c = _soft_kick(n, best[0], best[1], rng)
            r = best[1].copy()
            c, r = _solve_once(n, rng, np.concatenate([c.ravel(), r]))
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
