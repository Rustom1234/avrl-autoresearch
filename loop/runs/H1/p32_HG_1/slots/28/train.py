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


def _solve_once(n, rng):
    if rng.random() < 0.33:
        # Hexagonal lattice start
        # Approximate grid to cover unit square with ~32-40 points
        # Rows with alternating offsets
        rows = 6
        cols = 7
        pts = []
        for r in range(rows):
            for c in range(cols):
                x = c * (1.0 / (cols - 1))
                y = r * (1.0 / (rows - 1))
                if r % 2 == 1:
                    x += 0.5 / (cols - 1)
                pts.append([x, y])
        pts = np.array(pts)
        # Jitter and filter
        pts += rng.normal(0, 0.02, pts.shape)
        # Keep only n points that are most central
        dist_sq = np.sum((pts - 0.5)**2, axis=1)
        indices = np.argsort(dist_sq)[:n]
        pts = pts[indices]
        x0 = np.concatenate([pts.flatten(), np.full(n, INIT_RADIUS)])
    else:
        if rng.random() < 0.5:
            # Halton low-discrepancy sequence
            def get_halton(indices, base):
                res = np.zeros(len(indices))
                for i, idx in enumerate(indices):
                    f = 1.0
                    factor = 1.0 / base
                    while f > 1e-6:
                        res[i] += f * (idx % base)
                        idx //= base
                        f *= factor
                    return res
            
            # Simple scrambled Halton
            indices = np.arange(n)
            # Use the rng object to shuffle indices
            temp_rng = np.random.default_rng(rng.bit_generator.state[0], key=None) # Not ideal but we can just use a simple shuffle
            # Since we can't easily get a new rng from old, just use a simple shuffle or just use the rng object
            # Let's just use a standard shuffle and skip the seed issue
            idx_arr = np.arange(n)
            np.random.Generator.shuffle(idx_arr, rng)
            pts = np.column_stack([get_halton(idx_arr, 2), get_halton(idx_arr, 3)])
            pts = pts * (1.0 - 2 * INIT_LOW) + INIT_LOW
            x0 = np.concatenate([pts.flatten(), np.full(n, INIT_RADIUS)])
        else:
            # Uniform random start
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
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    while time.process_time() - t0 < TIME_FRACTION * time_limit:   # random restarts until the time is used
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
    return best
