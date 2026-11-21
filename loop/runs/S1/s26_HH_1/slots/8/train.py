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


def _equal_opt(c0):
    # Phase 1: maximise a single shared radius s for 26 equal circles (SLSQP on x, y, s).
    n = 26
    I, J = np.triu_indices(n, 1)
    s0 = 0.01
    z0 = np.concatenate([c0[:, 0], c0[:, 1], [s0]])

    def obj(z):
        return -z[-1]

    def obj_grad(z):
        g = np.zeros_like(z)
        g[-1] = -1.0
        return g

    def cons(z):
        x = z[:n]
        y = z[n:2 * n]
        s = z[-1]
        dx = x[I] - x[J]
        dy = y[I] - y[J]
        pair = dx * dx + dy * dy - 4.0 * s * s
        bnd = np.concatenate([x - s, 1 - x - s, y - s, 1 - y - s])
        return np.concatenate([pair, bnd])

    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)]
    try:
        res = minimize(obj, z0, jac=obj_grad, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "ineq", "fun": cons}],
                       options={"maxiter": 300, "ftol": 1e-12})
        z = res.x
        return np.clip(np.stack([z[:n], z[n:2 * n]], axis=1), 0.0, 1.0)
    except Exception:
        return c0


def _local_opt(c0, w=None):
    W = np.ones(26) if w is None else np.asarray(w, dtype=float)
    # Two-phase: equal-radius warm start, then free-radii SLSQP; LP polish; keep the better.
    c_orig = c0.copy()
    c0 = _equal_opt(c0)
    n = 26
    I, J = np.triu_indices(n, 1)
    r0 = _lp_radii(c0)
    z0 = np.concatenate([c0[:, 0], c0[:, 1], r0])

    def obj(z):
        return -np.sum(W * z[2 * n:])

    def obj_grad(z):
        g = np.zeros_like(z)
        g[2 * n:] = -W
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
    candidates = [c0.copy(), c_orig]
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


def _ls_growth(rng, steps=400):
    # Lubachevsky-Stillinger style growth: grow radii at a decaying rate,
    # push overlapping centers apart along the line of centers.
    n = 26
    I, J = np.triu_indices(n, 1)
    c = rng.uniform(0.1, 0.9, (n, 2))
    r = np.full(n, 0.01)
    rate = 0.01
    for _ in range(steps):
        r = r + rate * rng.uniform(0.5, 1.5, n)
        for _ in range(30):
            d = c[I] - c[J]
            dist = np.sqrt(np.sum(d * d, axis=1)) + 1e-12
            ov = r[I] + r[J] - dist
            m = ov > 1e-9
            if not np.any(m):
                break
            u = d[m] / dist[m, None]
            w = 0.5 * ov[m, None] * u
            np.add.at(c, I[m], w)
            np.add.at(c, J[m], -w)
            c = np.clip(c, 0.0, 1.0)
        rate *= 0.99
    return c


def jitter_anneal_search(seed=12345, budget=150.0):
    n = 26
    rng = np.random.default_rng(seed)
    t0 = time.time()
    c = _ls_growth(rng)
    best_c, best_r = _local_opt(c)
    best_s = float(np.sum(best_r))
    K = 20
    for k in range(K):
        if time.time() - t0 > budget:
            break
        frac = k / (K - 1)
        sigma = 0.1 * (0.002 / 0.1) ** frac
        cand = np.clip(best_c + rng.normal(0.0, sigma, (n, 2)), 0.0, 1.0)
        w = rng.uniform(0.8, 1.2, n)
        cc, rr = _local_opt(cand, w)
        s = float(np.sum(rr))
        if s > best_s:
            best_c, best_r, best_s = cc, rr, s
    return best_c, best_r


# This part remains fixed (not evolved)
def _valid_packing(centers, radii):
    # Strict double-precision check: inside the square, no overlaps.
    if np.any(radii < 0):
        return False
    x = centers[:, 0]
    y = centers[:, 1]
    if np.any(x - radii < 0) or np.any(x + radii > 1) or np.any(y - radii < 0) or np.any(y + radii > 1):
        return False
    n = centers.shape[0]
    I, J = np.triu_indices(n, 1)
    d = np.sqrt((x[I] - x[J]) ** 2 + (y[I] - y[J]) ** 2)
    if np.any(radii[I] + radii[J] > d):
        return False
    return True


def run_packing():
    """Run the circle packing search under a ~200 s wall-time budget, keeping the best valid packing"""
    t_start = time.perf_counter()
    total_budget = 200.0
    best_c, best_r, best_s = None, None, -1.0
    k = 0
    while True:
        remaining = total_budget - (time.perf_counter() - t_start)
        if remaining <= 5.0 or (k > 0 and remaining < 30.0):
            break
        c, r = jitter_anneal_search(seed=12345 + 7919 * k, budget=min(150.0, remaining - 10.0))
        s = float(np.sum(r))
        if s > best_s and _valid_packing(c, r):
            best_c, best_r, best_s = c, r, s
        k += 1
    if best_c is None:
        # Last-resort valid output (all radii zero).
        best_c = np.full((26, 2), 0.5)
        best_r = np.zeros(26)
        best_s = 0.0
    sum_radii = float(np.sum(best_r))
    return best_c, best_r, sum_radii
