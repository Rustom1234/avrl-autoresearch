# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""

import numpy as np
from scipy.optimize import linprog


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
    return best_c, best_r


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
