# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""

import numpy as np


import time
from scipy.optimize import minimize


def _repair(centers, radii):
    """Shrink radii so every circle is inside the square and no pair overlaps."""
    n = centers.shape[0]
    x = centers[:, 0]
    y = centers[:, 1]
    r = np.minimum.reduce([radii, x, 1 - x, y, 1 - y])
    r = np.maximum(r, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            s = r[i] + r[j]
            if s > d and s > 0:
                f = d / s
                r[i] *= f
                r[j] *= f
    return r * (1 - 1e-9)


def _pack_once(seed):
    """Mirror-symmetric random start, then unconstrained-symmetry SLSQP polish."""
    n = 26
    rng = np.random.default_rng(seed)
    rows_patterns = [(5, 6, 5, 6, 4), (6, 5, 6, 5, 4), (5, 5, 6, 5, 5)]
    rows = rows_patterns[seed % len(rows_patterns)]
    hex_pts = []
    nr = len(rows)
    for ri, cnt in enumerate(rows):
        off = 0.5 * (ri % 2)
        yy = 0.1 + 0.8 * ri / (nr - 1)
        for ci in range(cnt):
            xx = 0.1 + 0.8 * (ci + off) / (cnt - 1 + off) if cnt > 1 else 0.5
            hex_pts.append([xx, yy])
    pts = np.array(hex_pts) + rng.normal(0.0, 0.01, size=(n, 2))
    pts = np.clip(pts, 0.02, 0.98)
    iu, ju = np.triu_indices(n, 1)
    m = iu.size
    z0 = np.concatenate([pts[:, 0], pts[:, 1], np.full(n, 0.01)])
    k = np.arange(n)

    def obj(z):
        return -np.sum(z[2 * n:])

    def obj_jac(z):
        g = np.zeros_like(z)
        g[2 * n:] = -1.0
        return g

    def cons(z):
        x = z[:n]
        y = z[n:2 * n]
        r = z[2 * n:]
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        rs = r[iu] + r[ju]
        pair = dx * dx + dy * dy - rs * rs
        bnd = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
        return np.concatenate([pair, bnd])

    def cons_jac(z):
        x = z[:n]
        y = z[n:2 * n]
        r = z[2 * n:]
        J = np.zeros((m + 4 * n, 3 * n))
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        rs = r[iu] + r[ju]
        rows = np.arange(m)
        J[rows, iu] = 2 * dx
        J[rows, ju] = -2 * dx
        J[rows, n + iu] = 2 * dy
        J[rows, n + ju] = -2 * dy
        J[rows, 2 * n + iu] = -2 * rs
        J[rows, 2 * n + ju] = -2 * rs
        b = m
        J[b + k, k] = 1.0
        J[b + k, 2 * n + k] = -1.0
        J[b + n + k, k] = -1.0
        J[b + n + k, 2 * n + k] = -1.0
        J[b + 2 * n + k, n + k] = 1.0
        J[b + 2 * n + k, 2 * n + k] = -1.0
        J[b + 3 * n + k, n + k] = -1.0
        J[b + 3 * n + k, 2 * n + k] = -1.0
        return J

    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    try:
        res = minimize(obj, z0, jac=obj_jac, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
                       options={"maxiter": 500, "ftol": 1e-12})
        z = res.x
    except Exception:
        z = z0
    centers = np.column_stack([np.clip(z[:n], 0.0, 1.0), np.clip(z[n:2 * n], 0.0, 1.0)])
    radii = _repair(centers, np.maximum(z[2 * n:], 0.0))
    return centers, radii


def construct_packing():
    """Multi-start mirror-symmetric init + SLSQP; returns best (centers, radii)."""
    t0 = time.perf_counter()
    best = None
    best_sum = -1.0
    for seed in range(12):
        if seed > 0 and time.perf_counter() - t0 > 150.0:
            break
        c, r = _pack_once(seed)
        s = float(np.sum(r))
        if s > best_sum:
            best_sum = s
            best = (c, r)
    return best


def _old_construct_packing():
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


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
