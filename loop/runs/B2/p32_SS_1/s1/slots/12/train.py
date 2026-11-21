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
RELAX_R = 0.08
POOL = 12
DIV_TOL = 1e-3


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


def _lattice(n, rng):
    """Noisy staggered lattice with 5 to 7 rows."""
    k = rng.integers(5, 8)
    ys = np.linspace(0.1, 0.9, k)
    counts = np.full(k, n // k)
    counts[:n % k] += 1
    pts = []
    for row, (y, m) in enumerate(zip(ys, counts)):
        xs = (np.arange(m) + 0.5 + 0.5 * (row % 2)) / (m + 0.5)
        pts.append(np.column_stack([xs, np.full(m, y)]))
    c = np.vstack(pts) + rng.normal(0, 0.02, (n, 2))
    return np.clip(c, 0.05, 0.95)


def _relax(c, rr=RELAX_R):
    """Spread centres by minimising a smooth overlap and wall penalty at fixed radius rr."""
    n = len(c)
    i, j = np.triu_indices(n, 1)

    def energy(v):
        p = v.reshape(n, 2)
        diff = p[i] - p[j]
        d = np.sqrt((diff ** 2).sum(1)) + 1e-12
        ov = np.maximum(0, 2 * rr - d)
        lo = np.maximum(0, rr - p)
        hi = np.maximum(0, p - (1 - rr))
        e = (ov ** 2).sum() + (lo ** 2).sum() + (hi ** 2).sum()
        gi = (-2 * ov / d)[:, None] * diff
        g = np.zeros((n, 2))
        np.add.at(g, i, gi)
        np.add.at(g, j, -gi)
        g += -2 * lo + 2 * hi
        return e, g.ravel()

    res = minimize(energy, c.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 50})
    return np.clip(res.x.reshape(n, 2), 0, 1)


def _solve_once(n, rng, x0=None):
    if x0 is None:
        if rng.random() < 0.5:
            c0 = _lattice(n, rng)
        else:
            c0 = rng.uniform(INIT_LOW, 1 - INIT_LOW, (n, 2))
        x0 = np.concatenate([_relax(c0).ravel(), np.full(n, INIT_RADIUS)])
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


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    pool = []
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # restarts until the time is used
        x0 = _cross(pool, n, rng) if len(pool) >= 2 and rng.random() < 0.7 else None
        c, r = _solve_once(n, rng, x0)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if all(np.linalg.norm(np.sort(r) - np.sort(q)) >= DIV_TOL for _, q in pool):
            pool.append((c, r))
            pool.sort(key=lambda p: -p[1].sum())
            pool = pool[:POOL]
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
