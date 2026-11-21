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
        pts = rng.uniform(INIT_LOW, 1 - INIT_LOW, (4 * n, 2))
        seeds = pts[rng.choice(4 * n, n, replace=False)]
        for _ in range(30):
            weights = rng.uniform(0.7, 1.3, size=(n,))
            # For each point, find distance to each seed
            dist = np.linalg.norm(pts[:, np.newaxis, :] - seeds[np.newaxis, :, :], axis=2)
            # For each seed, find points in its Voronoi cell
            # We use the weighted approach: points are assigned to nearest seed
            # To vary cell sizes, we use the weights to influence the final mean
            # But the spec says 'random target weight per seed... so that cell sizes vary'
            # Interpretation: move seeds to the mean of assigned points, weighted by target weights
            # Actually, a standard Lloyd-relaxed centroidal Voronoi with varying weights:
            # A point p belongs to seed i if dist(p, i) is minimal.
            # Seed i moves to mean of points in its cell, adjusted by weight.
            # To keep it simple and valid: 
            # 1. Assign points to nearest seed.
            # 2. Move seed to weighted mean of its points.
            # Since we want to vary sizes, let's use a simpler interpretation: 
            # Use the points to form a Voronoi partition, move seeds to centroids.
            # To incorporate weights: move seeds to the mean of points in its cell, 
            # but the 'weight' can be seen as a multiplier on the displacement or a target weight.
            # Let's stick to: move seed to weighted mean of points in its Voronoi cell.
            # Wait, the prompt says 'random target weight... so that cell sizes vary'. 
            # This suggests the 'weight' might be the target area or a pull factor. 
            # Let's use weights as a multiplier on the contribution of the points to the mean.
            # Let's simplify: Just do the Lloyd relaxation and use weights to perturb the weights.
            # Correct interpretation of Lloyd with weights: seeds move to weighted centroids.
            # We'll use the weights as the 'mass' of the points in that cell.
            assignments = np.argmin(dist, axis=1)
            for i in range(n):
                mask = (assignments == i)
                if np.any(mask):
                    # Weighted mean: sum(point * weight) / sum(weights)
                    # But the spec says weight per seed. Let's use weights[i] as a target weight.
                    # Let's just use the standard Lloyd and let the initial distribution of points 
                    # (which is uniform) provide the 'variation'. 
                    # Re-reading: 'moves each seed to the mean of its assigned points, 
                    # with a random target weight per seed drawn from rng in [0.7, 1.3]'
                    # This implies: new_seed_pos = (mean_pos_of_points_in_cell) * weight_i
                    # But we must stay in the square. Let's use weights as a multiplier on the 
                    # displacement from the current position.
                    # Actually, let's just move it to the mean and scale the result by the weight.
                    # To ensure it doesn't drift too much, let's just do the mean.
                    # The weights are meant to make the Voronoi cells different sizes.
                    # If we move seeds to the mean of assigned points, the weights should 
                    # probably influence which points are 'captured' or how much they pull. 
                    # Let's just use the weight as a multiplier on the resulting mean.
                    mean_pos = pts[mask].mean(axis=0)
                    seeds[i] = mean_pos * weights[i]
            seeds = np.clip(seeds, INIT_LOW, 1 - INIT_LOW)
        x0 = np.concatenate([seeds.ravel(), np.full(n, INIT_RADIUS)])
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
    
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        if best is None or hops_since_improvement > 15:
            c, r = _solve_once(n, rng)
            c = np.clip(c, 0, 1)
            r = _repair(c, r)
            hops_since_improvement = 0
        else:
            # Perturb the best packing
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
            
    return best
