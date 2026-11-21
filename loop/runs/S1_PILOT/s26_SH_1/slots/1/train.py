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
    import time
    from scipy.optimize import minimize

    t0 = time.time()
    n_c = centers.shape[0]
    I, J = np.triu_indices(n_c, 1)

    def _safe(c, r):
        # Shrink radii so that every circle is inside the square and no pair overlaps.
        c = np.clip(c, 0.0, 1.0)
        bound = np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1 - c[:, 0], 1 - c[:, 1]))
        r = np.maximum(np.minimum(r, bound), 0.0)
        for a in range(n_c):
            for b in range(a + 1, n_c):
                d = np.sqrt(np.sum((c[a] - c[b]) ** 2))
                s = r[a] + r[b]
                if s > d:
                    f = d / s
                    r[a] *= f
                    r[b] *= f
        return c, r * (1 - 1e-9)

    def _cons(x):
        c = x[:2 * n_c].reshape(n_c, 2)
        r = x[2 * n_c:]
        d = np.sqrt(np.sum((c[I] - c[J]) ** 2, axis=1) + 1e-18)
        return np.concatenate([
            d - r[I] - r[J],
            c[:, 0] - r,
            1 - c[:, 0] - r,
            c[:, 1] - r,
            1 - c[:, 1] - r,
        ])

    def _obj(x):
        return -np.sum(x[2 * n_c:])

    def _obj_grad(x):
        return np.concatenate([np.zeros(2 * n_c), -np.ones(n_c)])

    def _opt(c, r):
        x0 = np.concatenate([c.ravel(), r])
        res = minimize(_obj, x0, jac=_obj_grad, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": _cons}],
                       options={"maxiter": 300, "ftol": 1e-12})
        return _safe(res.x[:2 * n_c].reshape(n_c, 2), res.x[2 * n_c:])

    def _hole(c, r, k):
        # Grid point with the largest clearance from all other circles and the walls.
        g = np.linspace(0.0, 1.0, 81)
        gx, gy = np.meshgrid(g, g)
        P = np.stack([gx.ravel(), gy.ravel()], axis=1)
        others = np.array([m for m in range(n_c) if m != k])
        D = np.sqrt(np.sum((P[:, None, :] - c[None, others, :]) ** 2, axis=2))
        clear = np.min(D - r[others][None, :], axis=1)
        border = np.min(np.minimum(P, 1 - P), axis=1)
        val = np.minimum(clear, border)
        return P[np.argmax(val)]

    rng = np.random.default_rng(0)
    c0 = rng.uniform(0.05, 0.95, (n_c, 2))
    r0 = np.exp(np.log(0.1) + 0.7 * rng.standard_normal(n_c))
    c_best, r_best = _opt(*_safe(c0, r0))
    for _ in range(60):
        if time.time() - t0 > 150:
            break
        k = int(np.argmin(r_best))
        c = c_best.copy()
        r = r_best.copy()
        c[k] = _hole(c_best, r_best, k)
        r[k] = 0.0
        c_new, r_new = _opt(c, r)
        if np.sum(r_new) > np.sum(r_best) + 1e-10:
            c_best, r_best = c_new, r_new

    centers, radii = c_best, r_best
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
