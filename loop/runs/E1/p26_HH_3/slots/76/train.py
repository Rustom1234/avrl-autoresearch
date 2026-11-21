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
RESTART_FRACTION = 0.3
BEAM_WIDTH = 3
BEAM_EXTRA = 2
BEAM_SHARE = 0.25


def _optimise(n, x0):
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        d = np.sqrt(((c[iu[0]] - c[iu[1]]) ** 2).sum(1)) - r[iu[0]] - r[iu[1]]
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r])

    res = minimize(lambda v: -v[2 * n:].sum(), x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": MAX_ITER})
    return res.x[:2 * n].reshape(n, 2), res.x[2 * n:]


def _solve_once(n, rng):
    x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    return _optimise(n, x0)


def _grow(n, rng):
    """Grow a packing from 4 circles to n, inserting each new circle (radius 0) at the emptiest sampled point."""
    k = min(4, n)
    c = rng.uniform(INIT_LOW, 1 - INIT_LOW, (k, 2))
    c, r = _optimise(k, np.concatenate([c.ravel(), np.full(k, INIT_RADIUS)]))
    while len(r) < n:
        pts = rng.uniform(0, 1, (2000, 2))
        dc = np.sqrt(((pts[:, None] - c[None]) ** 2).sum(-1)).min(1)
        dw = np.minimum(pts, 1 - pts).min(1)
        best = pts[np.argmax(np.minimum(dc, dw))]
        c = np.vstack([c, best])
        r = np.append(r, 0.0)
        c, r = _optimise(len(r), np.concatenate([c.ravel(), r]))
    return c, r


def _beam(n, rng, deadline):
    """Overpack to n + BEAM_EXTRA circles, then beam-search removals down to n circles.

    After the deadline, candidates are only repaired (no SLSQP), so the search always finishes.
    """
    m = n + BEAM_EXTRA
    c0 = rng.uniform(0, 1, (m, 2))
    c, r = _optimise(m, np.concatenate([c0.ravel(), np.full(m, 0.02)]))
    c = np.clip(c, 0, 1)
    beam = [(c, _repair(c, r))]
    while len(beam[0][1]) > n:
        k = len(beam[0][1])
        cands = []
        for c, r in beam:
            for i in range(k):
                keep = np.arange(k) != i
                c2, r2 = c[keep], r[keep]
                if time.process_time() < deadline:
                    c2, r2 = _optimise(k - 1, np.concatenate([c2.ravel(), r2]))
                    c2 = np.clip(c2, 0, 1)
                cands.append((c2, _repair(c2, r2)))
        cands.sort(key=lambda p: -p[1].sum())
        beam = cands[:BEAM_WIDTH]
    return beam[0]


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
    rng, t0 = np.random.default_rng(seed), time.process_time()
    budget = TIME_FRACTION * time_limit
    c, r = _beam(n, rng, t0 + BEAM_SHARE * RESTART_FRACTION * budget)
    best = (c, r)
    while time.process_time() - t0 < RESTART_FRACTION * budget or best is None:   # short random-restart phase
        c, r = _grow(n, rng) if rng.random() < 0.5 else _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    c, r = best
    while time.process_time() - t0 < budget:   # remove-and-reinsert the smallest circle
        c2, r2 = c.copy(), r.copy()
        i = np.argmin(r2)
        c2[i] = rng.uniform(0, 1, 2)
        r2[i] = 0
        c2, r2 = _optimise(n, np.concatenate([c2.ravel(), r2]))
        c2 = np.clip(c2, 0, 1)
        r2 = _repair(c2, r2)
        if r2.sum() > r.sum():
            c, r = c2, r2
    return c, r
