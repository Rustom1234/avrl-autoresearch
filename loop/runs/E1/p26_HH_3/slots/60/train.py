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
    budget = TIME_FRACTION * time_limit
    while time.process_time() - t0 < RESTART_FRACTION * budget or best is None:   # short random-restart phase
        c, r = _grow(n, rng) if rng.random() < 0.5 else _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    T = [0.0005 * 3 ** k for k in range(4)]          # fixed chain temperatures (hot chains have larger T)
    S = [0.01 * 15 ** (k / 3) for k in range(4)]     # geometric jitter scales from 0.01 (cold) to 0.15 (hot)
    P = [0.25 * (k + 1) for k in range(4)]           # probability of jittering per proposal
    chains = [(best[0].copy(), best[1].copy()) for _ in range(4)]
    step = 0
    while time.process_time() - t0 < budget:   # parallel tempering over perturbation strength
        for k in range(4):
            c, r = chains[k]
            c2, r2 = c.copy(), r.copy()
            i = np.argmin(r2)
            c2[i] = rng.uniform(0, 1, 2)
            r2[i] = 0
            if rng.random() < P[k]:
                m = rng.random(n) < 0.5
                c2[m] += rng.normal(0, S[k], (m.sum(), 2))
            c2, r2 = _optimise(n, np.concatenate([c2.ravel(), r2]))
            c2 = np.clip(c2, 0, 1)
            r2 = _repair(c2, r2)
            d = r2.sum() - r.sum()
            if d > 0 or rng.random() < np.exp(d / T[k]):
                chains[k] = (c2, r2)
                if r2.sum() > best[1].sum():
                    best = (c2, r2)
        step += 1
        if step % 10 == 0:
            for k in range(3):
                Ei, Ej = chains[k][1].sum(), chains[k + 1][1].sum()
                a = (Ej - Ei) * (1 / T[k] - 1 / T[k + 1])
                if a >= 0 or rng.random() < np.exp(a):
                    chains[k], chains[k + 1] = chains[k + 1], chains[k]
    return best
