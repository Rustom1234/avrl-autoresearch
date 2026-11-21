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


def _solve_once(n, rng, init=None):
    base = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2)) if init is None else np.clip(init + rng.normal(0, 0.03, (n, 2)), 0, 1)
    x0 = np.concatenate([base.ravel(), np.full(n, INIT_RADIUS)])
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


def _splice(elite, n, rng):
    """Cut-and-splice crossover: circles of parent A on one side of a random line, circles of parent B on the other."""
    i, j = rng.choice(len(elite), 2, replace=False)
    (ca, ra), (cb, rb) = elite[i], elite[j]
    u = rng.normal(size=2)
    u /= np.linalg.norm(u)
    cut = rng.uniform(0.3, 0.7) - 0.5
    keep_a = (ca - 0.5) @ u < cut
    keep_b = (cb - 0.5) @ u >= cut
    c = np.vstack([ca[keep_a], cb[keep_b]])
    r = np.concatenate([ra[keep_a], rb[keep_b]])
    if len(r) > n:
        c = c[np.argsort(-r)[:n]]
    elif len(r) < n:
        c = np.vstack([c, rng.uniform(INIT_LOW, 1 - INIT_LOW, (n - len(r), 2))])
    return c


def _add_elite(elite, c, r, k=4):
    if any(abs(r.sum() - e[1].sum()) < 1e-9 for e in elite):
        return
    elite.append((c, r))
    elite.sort(key=lambda e: -e[1].sum())
    del elite[k:]


def solve(n, seed, time_limit):
    rng, t0, best, elite = np.random.default_rng(seed), time.process_time(), None, []
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        guided = best is not None and time.process_time() - t0 > 0.5 * TIME_FRACTION * time_limit and rng.random() < 0.5
        init = None
        if guided:
            init = _splice(elite, n, rng) if len(elite) > 1 else best[0]
        c, r = _solve_once(n, rng, init)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        _add_elite(elite, c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
