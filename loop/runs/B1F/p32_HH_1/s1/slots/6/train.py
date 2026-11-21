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


def _solve_once(n, rng, init=None, init_r=None):
    base = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2)) if init is None else np.clip(init + rng.normal(0, 0.03, (n, 2)), 0, 1)
    rad = np.full(n, INIT_RADIUS) if init_r is None else init_r
    x0 = np.concatenate([base.ravel(), rad])
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


ELITES = 4


def _sym(c, k):
    """Apply one of the 8 symmetries of the unit square to the centres."""
    x, y = c[:, 0], c[:, 1]
    if k & 1:
        x = 1 - x
    if k & 2:
        y = 1 - y
    if k & 4:
        x, y = y, x
    return np.stack([x, y], 1)


def _child(a, b, n, rng):
    """Cut-and-splice crossover of two elites (centres, radii); returns starts with radii at 0.85x."""
    cb = _sym(b[0], rng.integers(8))
    axis, cut = rng.integers(2), rng.uniform(0.35, 0.65)
    ka, kb = a[0][:, axis] < cut, cb[:, axis] >= cut
    c = np.concatenate([a[0][ka], cb[kb]])
    r = np.concatenate([a[1][ka], b[1][kb]])
    if len(r) > n:
        keep = np.argsort(-r)[:n]
        c, r = c[keep], r[keep]
    while len(r) < n:
        cand = rng.uniform(0, 1, (200, 2))
        clearance = np.sqrt(((cand[:, None] - c[None]) ** 2).sum(-1)).min(1)
        c = np.vstack([c, cand[np.argmax(clearance)]])
        r = np.append(r, 0.01)
    return c, 0.85 * r


def solve(n, seed, time_limit):
    rng, t0, elites = np.random.default_rng(seed), time.process_time(), []
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        late = time.process_time() - t0 > 0.4 * TIME_FRACTION * time_limit
        if late and len(elites) >= 2 and rng.random() < 0.5:
            i, j = rng.choice(len(elites), 2, replace=False)
            c0, r0 = _child(elites[i], elites[j], n, rng)
            c, r = _solve_once(n, rng, c0, r0)
        else:
            c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        s = r.sum()
        if all(abs(s - e[1].sum()) > 1e-7 for e in elites) and (len(elites) < ELITES or s > elites[-1][1].sum()):
            elites = sorted(elites + [(c, r)], key=lambda e: -e[1].sum())[:ELITES]
    return elites[0]
