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


def _solve_once(n, rng, start=None, sd=0.0):
    if start is None:
        x0 = np.concatenate([rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n), np.full(n, INIT_RADIUS)])
    else:
        x0 = np.concatenate([(start[0] + rng.normal(0, sd, start[0].shape)).ravel(), start[1] * 0.5])
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


def _improve(n, rng, best):
    """One perturbation step from a packing; keeps the result only if it is better."""
    c, r = _solve_once(n, rng, best, rng.choice([0.005, 0.01, 0.02, 0.05]))
    c = np.clip(c, 0, 1)
    r = _repair(c, r)
    return (c, r) if r.sum() > best[1].sum() else best


def solve(n, seed, time_limit):
    rng, t0 = np.random.default_rng(seed), time.process_time()
    elite = []  # two best packings from the restart phase
    while time.process_time() - t0 < 0.3 * time_limit:   # random restarts
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        elite = sorted(elite + [(c, r)], key=lambda p: -p[1].sum())[:2]
    chains = list(elite)
    k = 0
    while len(chains) == 2 and time.process_time() - t0 < 0.55 * time_limit:   # two-chain tournament
        chains[k % 2] = _improve(n, rng, chains[k % 2])
        k += 1
    if len(chains) == 2:   # drop the chain with the lower sum of radii
        chains = [max(chains, key=lambda p: p[1].sum())]
    best = chains[0]
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        best = _improve(n, rng, best)
    return best
