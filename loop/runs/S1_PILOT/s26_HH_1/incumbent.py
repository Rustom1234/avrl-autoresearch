# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""

import numpy as np
from scipy.optimize import linprog, minimize


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Place circles in a structured pattern
    # This is a simple pattern - evolution will improve this

    # First, place a large circle in the center
    centers[0] = [0.5, 0.5]

    # Place 8 circles around it in a ring
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    # Place 16 more circles in an outer ring
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    # Additional positioning adjustment to make sure all circles
    # are inside the square and don't overlap
    # Clip to ensure everything is inside the unit square
    centers = equal_radius_start(n)
    centers = batch_search(centers)

    # Annealed coordinate-noise search: noise sigma decays geometrically
    # from 0.05 to 1e-4; accept a perturbed packing only if its LP sum is higher.
    radii = lp_radii(centers)
    best_c = centers.copy()
    best_r = radii.copy()
    best_s = best_r.sum()
    rng = np.random.default_rng(12345)
    steps = 2000
    for k in range(steps):
        sigma = 0.05 * (1e-4 / 0.05) ** (k / (steps - 1))
        cand = np.clip(best_c + rng.normal(0.0, sigma, best_c.shape), 0.0, 1.0)
        r = lp_radii(cand)
        s = r.sum()
        if s > best_s:
            best_c, best_r, best_s = cand, r, s
    best_c, best_r = joint_slsqp(best_c, best_r)
    return best_c, best_r


def equal_radius_start(n, seed=7):
    """Stage 1: bisection on a common radius r; for each r, minimise the
    overlap/wall-violation penalty over centers. Returns centers for the
    largest r found feasible."""
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, 1)

    def energy(z, r):
        P = z.reshape(n, 2)
        d = P[:, None, :] - P[None, :, :]
        D = np.sqrt((d ** 2).sum(-1) + 1e-18)[iu]
        ov = np.maximum(0.0, 2 * r - D)
        wall = np.maximum(0.0, r - P) + np.maximum(0.0, P - (1.0 - r))
        return (ov ** 2).sum() + (wall ** 2).sum()

    def feasible(r):
        best = None
        for _ in range(2):
            z0 = rng.uniform(r, 1 - r, 2 * n)
            res = minimize(energy, z0, args=(r,), method="L-BFGS-B",
                           options={"maxiter": 200})
            if best is None or res.fun < best.fun:
                best = res
            if best.fun < 1e-14:
                break
        return best

    lo, hi = 0.02, 0.12
    best_z = None
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        res = feasible(mid)
        if res.fun < 1e-12:
            lo = mid
            best_z = res.x.copy()
        else:
            hi = mid
    if best_z is None:
        best_z = rng.uniform(0.05, 0.95, 2 * n)
    return np.clip(best_z.reshape(n, 2), 0.0, 1.0)


def batch_search(centers, B=32, gens=8, iters=150, seed=99):
    """Batched population search: vectorized penalty descent over all B
    candidates at a fixed target radius, exact LP scoring, top-half
    selection and perturbed replacement of the bottom half."""
    rng = np.random.default_rng(seed)
    n = centers.shape[0]
    eye = np.eye(n, dtype=bool)
    rt = 0.1
    pop = np.clip(np.repeat(centers[None], B, axis=0)
                  + rng.normal(0.0, 0.02, (B, n, 2)), 0.0, 1.0)
    best_c = centers.copy()
    best_s = lp_radii(centers).sum()
    for _ in range(gens):
        P = pop.copy()
        for _ in range(iters):
            d = P[:, :, None, :] - P[:, None, :, :]
            D = np.sqrt((d ** 2).sum(-1) + 1e-18)
            ov = np.where(eye, 0.0, np.maximum(0.0, 2 * rt - D))
            g_pair = np.sum((-2 * ov / D)[..., None] * d, axis=2)
            g_wall = -2 * np.maximum(0.0, rt - P) + 2 * np.maximum(0.0, P - (1 - rt))
            P = np.clip(P - 0.02 * (g_pair + g_wall), 0.0, 1.0)
        sc = np.array([lp_radii(P[b]).sum() for b in range(B)])
        order = np.argsort(-sc)
        if sc[order[0]] > best_s:
            best_s = sc[order[0]]
            best_c = P[order[0]].copy()
        half = B // 2
        top = P[order[:half]]
        new = np.empty_like(P)
        new[:half] = top
        for k in range(half, B):
            src = top[(k - half) % half]
            new[k] = np.clip(src + rng.normal(0.0, 0.02, src.shape), 0.0, 1.0)
        pop = new
    return best_c


def lp_radii(centers):
    """Exact max radii for fixed centers via LP: maximize sum r subject to
    r_i + r_j <= d_ij and r_i <= distance to each wall."""
    n = centers.shape[0]
    rows = []
    b = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            b.append(d)
    for i in range(n):
        x, y = centers[i]
        for bound in (x, y, 1.0 - x, 1.0 - y):
            row = np.zeros(n)
            row[i] = 1.0
            rows.append(row)
            b.append(max(bound, 0.0))
    res = linprog(-np.ones(n), A_ub=np.array(rows), b_ub=np.array(b),
                  bounds=[(0, None)] * n, method="highs")
    if res.status != 0:
        return np.zeros(n)
    # tiny safety margin for floating point feasibility
    return np.maximum(res.x - 1e-9, 0.0)


def joint_slsqp(centers, radii, rounds=3):
    """Joint SLSQP over 2n centre coords and n radii, maximise sum(r).
    Pair:  |p_i-p_j|^2 - (r_i+r_j)^2 >= 0.  Wall: x_i-r_i, y_i-r_i,
    1-x_i-r_i, 1-y_i-r_i >= 0.  Final radii are re-solved exactly by LP."""
    n = centers.shape[0]
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    idx = np.arange(n)
    base_sum = radii.sum()

    def cons(z):
        P = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        d = P[iu] - P[ju]
        pair = (d ** 2).sum(1) - (r[iu] + r[ju]) ** 2
        wall = np.concatenate([P[:, 0] - r, P[:, 1] - r,
                               1.0 - P[:, 0] - r, 1.0 - P[:, 1] - r])
        return np.concatenate([pair, wall])

    def cons_jac(z):
        P = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        d = P[iu] - P[ju]
        s = r[iu] + r[ju]
        J = np.zeros((m + 4 * n, 3 * n))
        k = np.arange(m)
        J[k, 2 * iu] = 2 * d[:, 0]
        J[k, 2 * ju] = -2 * d[:, 0]
        J[k, 2 * iu + 1] = 2 * d[:, 1]
        J[k, 2 * ju + 1] = -2 * d[:, 1]
        J[k, 2 * n + iu] = -2 * s
        J[k, 2 * n + ju] = -2 * s
        b = m
        J[b + idx, 2 * idx] = 1.0
        J[b + idx, 2 * n + idx] = -1.0
        J[b + n + idx, 2 * idx + 1] = 1.0
        J[b + n + idx, 2 * n + idx] = -1.0
        J[b + 2 * n + idx, 2 * idx] = -1.0
        J[b + 2 * n + idx, 2 * n + idx] = -1.0
        J[b + 3 * n + idx, 2 * idx + 1] = -1.0
        J[b + 3 * n + idx, 2 * n + idx] = -1.0
        return J

    def obj(z):
        return -np.sum(z[2 * n:])

    def obj_jac(z):
        g = np.zeros(3 * n)
        g[2 * n:] = -1.0
        return g

    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    z = np.concatenate([centers.ravel(), radii])
    prev = obj(z)
    for _ in range(rounds):
        try:
            res = minimize(obj, z, jac=obj_jac, method="SLSQP", bounds=bounds,
                           constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
                           options={"maxiter": 300, "ftol": 1e-12})
        except Exception:
            break
        if not np.all(np.isfinite(res.x)):
            break
        z = res.x
        cur = obj(z)
        if prev - cur < 1e-10:
            break
        prev = cur

    c = np.clip(z[:2 * n].reshape(n, 2), 0.0, 1.0)
    r = lp_radii(c)
    if r.sum() > base_sum:
        return c, r
    return centers, radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # Then, limit by distance to other circles
            # Each pair of circles with centers at distance d can have
            # sum of radii at most d to avoid overlap
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

                    # If current radii would cause overlap
                    if radii[i] + radii[j] > dist:
                        # Scale both radii proportionally
                        scale = dist / (radii[i] + radii[j])
                        radii[i] *= scale
                        radii[j] *= scale

    # Ensure no negative radii due to numerical issues
    radii = np.maximum(radii, 0.0)
    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
