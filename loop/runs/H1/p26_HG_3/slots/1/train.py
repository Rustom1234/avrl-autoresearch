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
POP_SIZE = 5


def _solve_once(n, rng):
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
    rng, t0, population = np.random.default_rng(seed), time.process_time(), []
    
    def update_pop(c, r):
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if len(population) < POP_SIZE:
            population.append((c, r))
        else:
            population.sort(key=lambda x: x[1].sum(), reverse=True)
            best_score = population[-1][1].sum()
            if r.sum() > best_score:
                population[-1] = (c, r)

    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        if rng.random() < 0.8:
            c, r = _solve_once(n, rng)
            update_pop(c, r)
        elif len(population) >= 2:
            idx = rng.choice(len(population), 2, replace=False)
            p1, p2 = population[idx[0]], population[idx[1]]
            c1, r1 = p1
            c2, r2 = p2
            mid = n // 2
            c_child = np.concatenate([c1[:mid], c2[mid:]])
            r_child = np.concatenate([r1[:mid], r2[mid:]])
            # Re-adjust if n is odd
            if len(c_child) < n:
                c_child = np.concatenate([c_child, rng.uniform(INIT_LOW, 1-INIT_LOW, (n-len(c_child), 2))], axis=1)
                r_child = np.concatenate([r_child, np.full(n-len(r_child), INIT_RADIUS)])
            
            c_child = np.clip(c_child, 0, 1)
            r_child = _repair(c_child, r_child)
            update_pop(c_child, r_child)
            
    population.sort(key=lambda x: x[1].sum(), reverse=True)
    return population[0]
