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
    
    # Parallel Tempering setup
    temps = [0.0, 0.002, 0.005, 0.01]
    replicas = []
    for t in temps:
        c, r = _solve_once(n, rng)
        c = np.clip(c, 0, 1)
        r = _repair(c, r)
        replicas.append({'c': c, 'r': r, 'temp': t})
    
    best = max(replicas, key=lambda x: x['r'].sum())
    
    while time.process_time() - t0 < TIME_FRACTION * time_limit:
        # Metropolis-Hastings step for each replica
        for i in range(len(replicas)):
            rep = replicas[i]
            c_curr, r_curr = rep['c'], rep['r']
            
            # Perturbation
            c_new = c_curr.copy()
            num_to_move = rng.integers(1, 4)
            indices = rng.choice(n, num_to_move, replace=False)
            if rng.random() < 0.5:
                c_new[indices] = rng.uniform(INIT_LOW, 1 - INIT_LOW, (num_to_move, 2))
            else:
                c_new += rng.normal(0, 0.02, (n, 2))
                c_new = np.clip(c_new, 0, 1)
            
            x0 = np.concatenate([c_new.ravel(), r_curr])
            c_sol, r_sol = _solve_once(n, rng, x0=x0)
            c_sol = np.clip(c_sol, 0, 1)
            r_sol = _repair(c_sol, r_sol)
            
            delta = r_sol.sum() - r_curr.sum()
            if delta > 0 or (rep['temp'] > 0 and rng.random() < np.exp(delta / rep['temp'])):
                rep['c'], rep['r'] = c_sol, r_sol
        
        # Swap adjacent replicas
        for i in range(len(temps) - 1):
            r1, r2 = replicas[i], replicas[i+1]
            delta = (r2['r'].sum() - r1['r'].sum()) * (1.0 / (r1['temp'] + r2['temp'])) # simplified swap prob
            # Standard exchange: exp( (S2 - S1) / (1/T1 + 1/T2) )
            # But since T_i are small and fixed, we use a simpler transition
            if rng.random() < 0.1: # Small chance to swap to maintain diversity
                replicas[i], replicas[i+1] = replicas[i+1], replicas[i]
        
        best = max(replicas, key=lambda x: x['r'].sum())
        if best['r'].sum() > best['r'].sum(): # Safety
             pass
             
    return best['c'], best['r']
