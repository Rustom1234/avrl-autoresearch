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
    centers = np.zeros((n, 2))
    for i in range(n):
        success = False
        for _ in range(200):
            cand = rng.uniform(0, 1, 2)
            if i == 0:
                centers[i] = cand
                success = True
                break
            dist = np.linalg.norm(centers[:i] - cand, axis=1)
            if np.all(dist > 0.12):
                centers[i] = cand
                success = True
                break
        if not success:
            centers[i] = rng.uniform(0, 1, 2)

    x0 = np.concatenate([centers.flatten(), np.full(n, INIT_RADIUS)])
    iu = np.triu_indices(n, 1)

    def cons(v):
        c, r = v[:2 * n].reshape(n, 2), v[2 * n:]
        # Constraints: 
        # d = sqrt((c1x-c2x)^2 + (c1y-c2y)^2) - r1 - r2 >= 0
        # c1x - r1 >= 0, c1y - r1 >= 0, 1 - c1x - r1 >= 0, 1 - c1y - r1 >= 0
        # Same for c2 and r2
        # r >= 0
        dx = c[iu[0], 0] - c[iu[1], 0]
        dy = c[iu[0], 1] - c[iu[1], 1]
        dist = np.sqrt(dx**2 + dy**2)
        d = dist - r[iu[0]] - r[iu[1]]
        
        # Construct Jacobian
        # Number of constraints: len(iu) + 4*n + n
        # Variables: 3*n (c1x, c1y, r1, c2x, c2y, r2, ...)
        # Wait, variables are c (n,2) and r (n,), so 3*n total variables.
        # The original code used x0 = concat(centers.flatten(), radii) -> 3n variables.
        
        # Let's re-calculate Jacobian for: 
        # f = [d, c0x-r0, c0y-r0, 1-c0x-r0, 1-c0y-r0, ..., r0, r1, ...]
        
        # Variable indices: 
        # c[i, 0] is 2*i, c[i, 1] is 2*i+1, r[i] is 2*n + i
        # Total variables = 3n. Let's use 3n indices.
        # Wait, the original code used x0 = np.concatenate([centers.flatten(), np.full(n, INIT_RADIUS)])
        # So indices are: 0..2n-1 are centers (c[0x, c[0y], c[1x], c[1y]...]), 2n..3n-1 are radii.
        
        jac = np.zeros((len(iu) + 5 * n, 3 * n))
        
        # Part 1: Overlap constraints (d >= 0)
        # d = sqrt((c[i_0x] - c[j_x])^2 + (c[i_0y] - c[j_y])^2) - r[i_0] - r[j]
        # grad wrt c[i_0x]: (c[i_0x] - c[j_x]) / dist
        # grad wrt c[j_x]: -(c[i_0x] - c[j_x]) / dist
        # grad wrt r[i_0]: -1
        # grad wrt r[j]: -1
        
        # Note: original code used centers.flatten(), so c[i,0] is index 2*i, c[i,1] is 2*i+1
        # radii r[i] is index 2*n + i
        
        idx_i = iu[0]
        idx_j = iu[1]
        
        for k in range(len(iu)):
            i, j = idx_i[k], idx_j[k]
            # Use explicit indices for center and radius
            # i_c = 2*i, i_r = 2*n + i
            # j_c = 2*j, j_r = 2*n + j
            # However, the input is centers.flatten() -> [c0x, c0y, c1x, c1y...]
            # So c[i, 0] is 2*i, c[i, 1] is 2*i+1
            
            # Let's simplify: c_i is (c[i,0], c[i,1]), r_i is r[i]
            # Indices in v: c[i,0] is 2*i, c[i,1] is 2*i+1, r[i] is 2*n+i
            
            # Oh, centers.flatten() means c[0,0] is 0, c[0,1] is 1, c[1,0] is 2, c[1,1] is 3
            # So center c[i] are at 2*i and 2*i+1
            # Radius r[i] is at 2*n + i
            
            dist_val = dist[k]
            # Grad wrt c[i,0]
            jac[k, 2*i] += dx[k] / dist_val
            # Grad wrt c[i,1]
            jac[k, 2*i+1] += dy[k] / dist_val
            # Grad wrt c[j,0]
            jac[k, 2*j] -= dx[k] / dist_val
            # Grad wrt c[j,1]
            jac[k, 2*j+1] -= dy[k] / dist_val
            # Grad wrt r[i]
            jac[k, 2*n + i] -= 1
            # Grad wrt r[j]
            jac[k, 2*n + j] -= 1
            
        # Part 2: Boundary constraints (c0x-r0 >= 0, c0y-r0 >= 0, 1-c0x-r0 >= 0, 1-c0y-r0 >= 0)
        # Indices for c[i,0]-r[i] is i_c = 2*i, i_r = 2*n+i
        # Indices for c[i,1]-r[i] is i_c' = 2*i+1, i_r = 2*n+i
        # Indices for 1-c[i,0]-r[i] is i_c = 2*i, i_r = 2*n+i
        # Indices for 1-c[i,1]-r[i] is i_c' = 2*i+1, i_r = 2*n+i
        
        for i in range(n):
            # c[i,0] - r[i] >= 0
            jac[len(iu) + 2*i, 2*i] = -1
            jac[len(iu) + 2*i, 2*n + i] = -1
            # c[i,1] - r[i] >= 0
            jac[len(iu) + 2*i + 1, 2*i+1] = -1
            jac[len(iu) + 2*i + 1, 2*n + i] = -1
            # 1 - c[i,0] - r[i] >= 0
            jac[len(iu) + 2*n + 2*i, 2*i] = -1
            jac[len(iu) + 2*n + 2*i, 2*n + i] = -1
            # 1 - c[i,1] - r[i] >= 0
            jac[len(iu) + 2*n + 2*i + 1, 2*i+1] = -1
            jac[len(iu) + 2*n + 2*i + 1, 2*n + i] = -1
            
        # Part 3: r[i] >= 0
        for i in range(n):
            jac[len(iu) + 4*n + i, 2*n + i] = 1
            
        return np.concatenate([d, c[:, 0] - r, c[:, 1] - r, 1 - c[:, 0] - r, 1 - c[:, 1] - r, r]), jac

    # The original code used a lambda, but SLSQP can take a Jacobian function.
    # We need to adjust the call.
    def obj(v): return -v[2*n:].sum()
    
    # Since we need the Jacobian of the constraints, we pass the jac function.
    # But the constraint 'fun' is what returns the vector, and 'jac' is its gradient.
    res = minimize(obj, x0, method="SLSQP", 
                    constraintsMes_grad={"type": "ineq", "fun": cons, "jac": lambda v: cons(v)[1]},
                    options={"maxiter": MAX_ITER})
    # Wait, the constraint 'jac' parameter takes the Jacobian of the constraint function.
    # My 'cons' returns (values, jacobian). So I can just wrap it.
    
    # Let's re-write the call to be cleaner:
    
    def constraint_fun(v):
        return cons(v)[0]
    
    def constraint_jac(v):
        return cons(v)[1]
    
    res = minimize(obj, x0, method="SLSQP", 
                    constraints={{"type": "ineq", "fun": constraint_fun, "jac": constraint_jac}},
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
