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
MOVES = 4


def _perturb(move, best_c, best_r, rng):
    """Build a start from the incumbent: 1 = Gaussian kick, 2 = slice-and-rotate, 3 = remove-and-reinsert."""
    c = best_c.copy()
    if move == 1:
        c = c + rng.normal(0, 0.03, c.shape)
    elif move == 2:
        cut, axis = rng.uniform(0.3, 0.7), rng.integers(2)
        mask = c[:, axis] > cut
        if mask.any():
            cen = c[mask].mean(0)
            th = rng.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            c[mask] = (c[mask] - cen) @ R.T + cen
    else:
        c[np.argmin(best_r)] = rng.uniform(0.1, 0.9, 2)
    return np.clip(c, 0, 1)


def _solve_once(n, rng, init=None):
    pos = rng.uniform(INIT_LOW, 1 - INIT_LOW, 2 * n) if init is None else init.ravel()
    x0 = np.concatenate([pos, np.full(n, INIT_RADIUS)])
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

    x1 = np.concatenate([x0[:2 * n], [INIT_RADIUS]])
    res1 = minimize(lambda v: -v[2 * n], x1, method="SLSQP", constraints={"type": "ineq", "fun": cons_eq},
                    options={"maxiter": MAX_ITER})
    x2 = np.concatenate([res1.x[:2 * n], np.full(n, max(res1.x[2 * n], 0.0))])

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
    counts, sums = np.zeros(MOVES), np.zeros(MOVES)
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        if best is None:
            move = 0
        else:   # UCB1 over move types
            ucb = sums / np.maximum(counts, 1) + np.sqrt(2 * np.log(counts.sum() + 1) / np.maximum(counts, 1))
            ucb[counts == 0] = np.inf
            move = int(np.argmax(ucb))
        init = None if move == 0 or best is None else _perturb(move, best[0], best[1], rng)
        c, r = _solve_once(n, rng, init)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        reward = 1.0 if best is not None and r.sum() > best[1].sum() + 1e-6 else 0.0
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
        counts[move] += 1
        sums[move] += reward
    return best
