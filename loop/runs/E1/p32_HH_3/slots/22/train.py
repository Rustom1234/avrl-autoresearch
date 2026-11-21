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
N_CHAINS = 4
T_LADDER = np.geomspace(0.0005, 0.01, N_CHAINS)


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


def solve(n, seed, time_limit):
    rng, t0 = np.random.default_rng(seed), time.process_time()
    c, r = _solve_once(n, rng)
    c = np.clip(c, 0, 1)
    r = _repair(c, r)
    best = (c, r)
    chains = [(c, r) for _ in T_LADDER]
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # replica-exchange basin hopping
        for k, T in enumerate(T_LADDER):
            bc, br = chains[k]
            c0 = np.clip(bc + rng.normal(0, 0.02, bc.shape), 0, 1)
            r0 = np.maximum(br + rng.normal(0, 0.02, br.shape), 0)
            c, r = _solve_once(n, rng, np.concatenate([c0.ravel(), r0]))
            c = np.clip(c, 0, 1)
            r = _repair(c, r)
            delta = r.sum() - br.sum()
            if delta > 0 or rng.random() < np.exp(delta / T):
                chains[k] = (c, r)
            if r.sum() > best[1].sum():
                best = (c, r)
        for k in range(N_CHAINS - 1):   # neighbour swap, standard replica-exchange acceptance
            fi, fj = chains[k][1].sum(), chains[k + 1][1].sum()
            log_a = (fj - fi) * (1 / T_LADDER[k] - 1 / T_LADDER[k + 1])
            if np.log(rng.random()) < log_a:
                chains[k], chains[k + 1] = chains[k + 1], chains[k]
    return best
