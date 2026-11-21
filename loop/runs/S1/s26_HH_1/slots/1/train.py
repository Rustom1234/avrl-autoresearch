# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""

import numpy as np


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
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)
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


import time
from scipy.optimize import minimize, linprog


def _lp_radii(centers):
    # Maximise sum of radii for fixed centers (LP), then shrink by a tiny margin.
    n = centers.shape[0]
    x = np.clip(centers[:, 0], 0.0, 1.0)
    y = np.clip(centers[:, 1], 0.0, 1.0)
    border = np.maximum(np.minimum(np.minimum(x, y), np.minimum(1 - x, 1 - y)), 0.0)
    I, J = np.triu_indices(n, 1)
    d = np.sqrt((x[I] - x[J]) ** 2 + (y[I] - y[J]) ** 2)
    A = np.zeros((len(I), n))
    A[np.arange(len(I)), I] = 1.0
    A[np.arange(len(I)), J] = 1.0
    bounds = [(0.0, float(b)) for b in border]
    res = linprog(-np.ones(n), A_ub=A, b_ub=d, bounds=bounds, method="highs")
    if res.status != 0:
        return np.zeros(n)
    return np.maximum(res.x - 1e-10, 0.0)


def _relax(c, iters=120):
    # Force-directed relaxation: pairs repel with strength ~ (r_i + r_j), walls push inward,
    # damping decreases over time; radii from the LP are refreshed every 10 steps.
    n = c.shape[0]
    c = c.copy()
    I, J = np.triu_indices(n, 1)
    r = _lp_radii(c)
    for t in range(iters):
        if t % 10 == 0:
            r = _lp_radii(c)
        damp = 0.05 * (1.0 - t / iters) + 0.002
        dvec = c[I] - c[J]
        d = np.sqrt(np.sum(dvec * dvec, axis=1)) + 1e-9
        s = r[I] + r[J]
        pf = (s / d ** 3)[:, None] * dvec
        F = np.zeros_like(c)
        np.add.at(F, I, pf)
        np.add.at(F, J, -pf)
        F[:, 0] += np.maximum(r - c[:, 0], 0.0) - np.maximum(r - (1.0 - c[:, 0]), 0.0)
        F[:, 1] += np.maximum(r - c[:, 1], 0.0) - np.maximum(r - (1.0 - c[:, 1]), 0.0)
        m = np.max(np.abs(F)) + 1e-12
        c = np.clip(c + damp * F / m, 0.0, 1.0)
    return c


def _local_opt(c0):
    c0 = _relax(c0)
    # SLSQP on (x, y, r) maximising sum r, then LP polish; keep the better of start/solver.
    n = 26
    I, J = np.triu_indices(n, 1)
    r0 = _lp_radii(c0)
    z0 = np.concatenate([c0[:, 0], c0[:, 1], r0])

    def obj(z):
        return -np.sum(z[2 * n:])

    def obj_grad(z):
        g = np.zeros_like(z)
        g[2 * n:] = -1.0
        return g

    def cons(z):
        x = z[:n]
        y = z[n:2 * n]
        r = z[2 * n:]
        dx = x[I] - x[J]
        dy = y[I] - y[J]
        rs = r[I] + r[J]
        pair = dx * dx + dy * dy - rs * rs
        bnd = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
        return np.concatenate([pair, bnd])

    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    candidates = [c0.copy()]
    try:
        res = minimize(obj, z0, jac=obj_grad, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "ineq", "fun": cons}],
                       options={"maxiter": 300, "ftol": 1e-12})
        z = res.x
        candidates.append(np.clip(np.stack([z[:n], z[n:2 * n]], axis=1), 0.0, 1.0))
    except Exception:
        pass
    best_c, best_r, best_s = None, None, -1.0
    for cand in candidates:
        rr = _lp_radii(cand)
        s = float(np.sum(rr))
        if s > best_s:
            best_c, best_r, best_s = cand, rr, s
    return best_c, best_r


def jitter_anneal_search():
    n = 26
    rng = np.random.default_rng(12345)
    t0 = time.time()
    c = rng.uniform(0.1, 0.9, (n, 2))
    best_c, best_r = _local_opt(c)
    best_s = float(np.sum(best_r))
    K = 8
    for k in range(K):
        if time.time() - t0 > 150.0:
            break
        frac = k / (K - 1)
        sigma = 0.05 * (1e-4 / 0.05) ** frac
        cand = np.clip(best_c + rng.normal(0.0, sigma, (n, 2)), 0.0, 1.0)
        cc, rr = _local_opt(cand)
        s = float(np.sum(rr))
        if s > best_s:
            best_c, best_r, best_s = cc, rr, s
    return best_c, best_r


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = jitter_anneal_search()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
