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
TEMPS = [1e-3, 3e-3, 1e-2, 3e-2]
SIGMAS = [0.01, 0.03, 0.08, 0.2]
SWAP_EVERY = 20


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


def solve(n, seed, time_limit):
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    states = []
    for _ in TEMPS:   # one chain per temperature, each starts from a random restart
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        states.append((c, r))
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    rounds = 0
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        for k, (T, sig) in enumerate(zip(TEMPS, SIGMAS)):
            c, r = states[k]
            c2 = np.clip(c + rng.normal(0, sig, c.shape), 0, 1)
            c2, r2 = _solve_once(n, rng, np.concatenate([c2.ravel(), r]))
            c2 = np.clip(c2, 0, 1)
            r2 = _repair(c2, r2)
            d = r2.sum() - r.sum()
            if d >= 0 or rng.uniform() < np.exp(d / T):   # Metropolis on sum(r)
                states[k] = (c2, r2)
            if r2.sum() > best[1].sum():
                best = (c2, r2)
        rounds += 1
        if rounds % SWAP_EVERY == 0:   # replica exchange between adjacent temperatures
            for k in range(len(states) - 1):
                f_a, f_b = states[k][1].sum(), states[k + 1][1].sum()
                a = (1 / TEMPS[k] - 1 / TEMPS[k + 1]) * (f_b - f_a)
                if rng.uniform() < np.exp(min(a, 0.0)):
                    states[k], states[k + 1] = states[k + 1], states[k]
    return best
