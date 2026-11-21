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
    n = 26
    I, Jp = np.triu_indices(n, 1)
    m = len(I)
    rows = np.arange(m)
    ar = np.arange(n)

    def cons(z):
        x = z[:n]
        y = z[n:2 * n]
        r = z[2 * n:]
        dx = x[I] - x[Jp]
        dy = y[I] - y[Jp]
        rs = r[I] + r[Jp]
        return np.concatenate([dx * dx + dy * dy - rs * rs, x - r, 1 - x - r, y - r, 1 - y - r])

    def cjac(z):
        x = z[:n]
        y = z[n:2 * n]
        r = z[2 * n:]
        dx = x[I] - x[Jp]
        dy = y[I] - y[Jp]
        rs = r[I] + r[Jp]
        Jm = np.zeros((m + 4 * n, 3 * n))
        Jm[rows, I] = 2 * dx
        Jm[rows, Jp] = -2 * dx
        Jm[rows, n + I] = 2 * dy
        Jm[rows, n + Jp] = -2 * dy
        Jm[rows, 2 * n + I] = -2 * rs
        Jm[rows, 2 * n + Jp] = -2 * rs
        b = m
        Jm[b + ar, ar] = 1
        Jm[b + ar, 2 * n + ar] = -1
        Jm[b + n + ar, ar] = -1
        Jm[b + n + ar, 2 * n + ar] = -1
        Jm[b + 2 * n + ar, n + ar] = 1
        Jm[b + 2 * n + ar, 2 * n + ar] = -1
        Jm[b + 3 * n + ar, n + ar] = -1
        Jm[b + 3 * n + ar, 2 * n + ar] = -1
        return Jm

    obj_grad = np.concatenate([np.zeros(2 * n), -np.ones(n)])

    def optimize(z0):
        res = minimize(lambda z: -np.sum(z[2 * n:]), z0, jac=lambda z: obj_grad,
                       constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                       bounds=[(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n,
                       method="SLSQP", options={"maxiter": 300, "ftol": 1e-12})
        z = res.x
        if not np.all(np.isfinite(z)):
            return z0
        return z

    def safe_radii(c, r):
        r = np.minimum(r, np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1 - c[:, 0], 1 - c[:, 1])))
        r = np.maximum(r, 0.0)
        for _ in range(500):
            bad = False
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.sqrt(np.sum((c[i] - c[j]) ** 2))
                    s = r[i] + r[j]
                    if s > d:
                        f = d / s
                        r[i] *= f
                        r[j] *= f
                        bad = True
            if not bad:
                break
        return r * (1 - 1e-9)

    rng = np.random.default_rng(12345)
    best_val = -1.0
    best_c = None
    best_r = None
    for start in range(4):
        if start > 0 and time.time() - t0 > 150:
            break
        c0 = rng.uniform(0.05, 0.95, size=(n, 2))
        z = np.concatenate([c0[:, 0], c0[:, 1], np.full(n, 0.05)])
        cur = optimize(z)
        cur_val = np.sum(cur[2 * n:])
        # shrink-and-grow cycles
        for cyc in range(8):
            if time.time() - t0 > 180:
                break
            zs = cur.copy()
            zs[2 * n:] *= 0.9
            zs[:2 * n] = np.clip(zs[:2 * n] + rng.normal(0, 0.02, 2 * n), 0.0, 1.0)
            zn = optimize(zs)
            v = np.sum(zn[2 * n:])
            if v > cur_val:
                cur, cur_val = zn, v
        if cur_val > best_val:
            best_val = cur_val
            best_c = np.stack([cur[:n], cur[n:2 * n]], axis=1)
            best_r = cur[2 * n:].copy()

    centers = np.clip(best_c, 0.0, 1.0)
    radii = safe_radii(centers, np.maximum(best_r, 0.0))
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
