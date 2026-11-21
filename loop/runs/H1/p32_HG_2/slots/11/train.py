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
    rng, t0, best = np.random.default_rng(seed), time.process_time(), None
    hops_since_improvement = 0
    
    pool = []
    
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        # Standard search or perturbation
        if best is None or hops_since_improvement > 15:
            c, r = _solve_once(n, rng)
            c = np.clip(c, 0, 1)
            r = _repair(c, r)
            hops_since_improvement = 0
        else:
            c_curr, r_curr = best
            c_new = c_curr.copy()
            num_to_move = rng.integers(1, 4)
            indices = rng.choice(n, num_to_move, replace=False)
            if rng.random() < 0.5:
                c_new[indices] = rng.uniform(INIT_LOW, 1 - INIT_LOW, (num_to_move, 2))
            else:
                c_new += rng.normal(0, 0.02, (n, 2))
                c_new = np.clip(c_new, 0, 1)
            x0 = np.concatenate([c_new.ravel(), r_curr])
            c, r = _solve_once(n, rng, x0=x0)
            c = np.clip(c, 0, 1)
            r = _repair(c, r)
            if r.sum() > best[1].sum():
                best = (c, r)
                hops_since_improvement = 0
            else:
                hops_since_improvement += 1
        
        if best is None or r.sum() > best[1].sum():
            best = (c, r)
            
        # Maintain pool and perform crossover
        if best is not None:
            pool.append(best)
            pool = sorted(pool, key=lambda x: x[1].sum(), reverse=True)[:5]
            
            if len(pool) >= 2 and rng.random() < 0.3:
                p1, p2 = rng.sample(pool, 2)
                # Create a cut line
                theta = rng.uniform(0, np.pi)
                dx, dy = np.cos(theta), np.sin(theta)
                # Project centers onto normal
                proj1 = (p1[0][:, 0] * dx + p1[0][:, 1] * dy)
                proj2 = (p2[0][:, 0] * dx + p2[0][:, 1] * dy)
                
                mid = (proj1.mean() + proj2.mean()) / 2
                mask1 = proj1 < mid
                mask2 = proj2 > mid
                
                # Combine
                c_child = np.empty((n, 2))
                r_child = np.empty(n)
                
                # Simple recombination
                # This is a simplified crossover: take centers from p1 where mask1 is true
                # and fill the rest from p2
                indices1 = np.where(mask1)[0]
                indices2 = np.where(mask2)[0]
                
                # Fill child
                c_child = np.zeros((n, 2))
                r_child = np.zeros(n)
                
                # Just take pieces and fix count
                c_child[indices1] = p1[0][indices1]
                r_child[indices1] = p1[1][indices1]
                
                rem = np.setdiff1d(np.arange(n), indices1)
                # Fill remainder with p2
                # Use simple replacement to maintain count n
                c_child[rem] = p2[0][indices2[:len(rem)]] if len(indices2) >= len(rem) else p2[0][indices2]
                r_child[rem] = p2[1][indices2[:len(rem)]] if len(indices2) >= len(rem) else p2[1][indices2]
                
                # Repair and check
                c_child = np.clip(c_child, 0, 1)
                r_child = _repair(c_child, r_child)
                
                if r_child.sum() > pool[-1][1].sum():
                    pool[-1] = (c_child, r_child)
                    pool = sorted(pool, key=lambda x: x[1].sum(), reverse=True)[:5]

    return best
