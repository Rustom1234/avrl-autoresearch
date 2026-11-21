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


def _relax(pts, rng, steps=300):
    """Soft-disk repulsion: overlaps of growing disks push centers apart; walls clip."""
    n = pts.shape[0]
    iu2, ju2 = np.triu_indices(n, 1)
    p = np.clip(pts + rng.normal(0.0, 0.01, size=pts.shape), 0.0, 1.0)
    for s in range(steps):
        R = 0.09 * (s + 1) / steps
        dx = p[iu2, 0] - p[ju2, 0]
        dy = p[iu2, 1] - p[ju2, 1]
        d = np.sqrt(dx * dx + dy * dy) + 1e-12
        ov = np.maximum(2.0 * R - d, 0.0)
        fx = ov * dx / d
        fy = ov * dy / d
        disp = np.zeros_like(p)
        np.add.at(disp[:, 0], iu2, 0.3 * fx)
        np.add.at(disp[:, 0], ju2, -0.3 * fx)
        np.add.at(disp[:, 1], iu2, 0.3 * fy)
        np.add.at(disp[:, 1], ju2, -0.3 * fy)
        p = np.clip(p + disp, R, 1.0 - R)
    return p


def _pack_once(seed):
    """Mirror-symmetric random start, then unconstrained-symmetry SLSQP polish."""
    n = 26
    rng = np.random.default_rng(seed)
    ring_sets = [(1, 6, 12, 7), (1, 7, 12, 6), (1, 6, 7, 12)]
    ring_r = [0.0, 0.2, 0.36, 0.46]
    rings = ring_sets[seed % len(ring_sets)]
    hex_pts = []
    for ri, cnt in enumerate(rings):
        rr = ring_r[ri]
        phase = 0.5 * np.pi / cnt * (ri % 2)
        for ci in range(cnt):
            a = 2 * np.pi * ci / cnt + phase
            hex_pts.append([0.5 + rr * np.cos(a), 0.5 + rr * np.sin(a)])
    base = np.array([[0.13, 0.13], [0.87, 0.13], [0.13, 0.87], [0.87, 0.87],
                     [0.5, 0.13], [0.5, 0.87], [0.13, 0.5], [0.87, 0.5]])
    interior = []
    for row, yy in enumerate([0.3, 0.5, 0.7]):
        shift = 0.04 if row % 2 else 0.0
        for xx in np.linspace(0.3, 0.7, 6) + shift:
            interior.append([xx, yy])
    pts = np.vstack([base, np.array(interior)])
    pts = pts + rng.normal(0.0, 0.02, size=pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    iu, ju = np.triu_indices(n, 1)
    m = iu.size
    pts = _relax(pts, rng)
    z0 = np.concatenate([pts[:, 0], pts[:, 1], np.full(n, 0.05)])
    k = np.arange(n)
    pbounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

    def pen(z, mu):
        x = z[:n]
        y = z[n:2 * n]
        r = z[2 * n:]
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        d = np.sqrt(dx * dx + dy * dy + 1e-18)
        ov = np.maximum(r[iu] + r[ju] - d, 0.0)
        bl = np.maximum(r - x, 0.0)
        br = np.maximum(r + x - 1.0, 0.0)
        bb = np.maximum(r - y, 0.0)
        bt = np.maximum(r + y - 1.0, 0.0)
        f = -np.sum(r) + mu * (np.sum(ov * ov) + np.sum(bl * bl + br * br + bb * bb + bt * bt))
        g = np.zeros_like(z)
        g[2 * n:] = -1.0
        gd = 2.0 * mu * ov
        ux = dx / d
        uy = dy / d
        np.add.at(g[2 * n:], iu, gd)
        np.add.at(g[2 * n:], ju, gd)
        np.add.at(g[:n], iu, -gd * ux)
        np.add.at(g[:n], ju, gd * ux)
        np.add.at(g[n:2 * n], iu, -gd * uy)
        np.add.at(g[n:2 * n], ju, gd * uy)
        g[2 * n:] += 2.0 * mu * (bl + br + bb + bt)
        g[:n] += 2.0 * mu * (br - bl)
        g[n:2 * n] += 2.0 * mu * (bt - bb)
        return f, g

    # Symmetric-constrained stage: circle j (j>=13) is the mirror of circle j-13 across y=x.
    # Reduced variables w = [u(13), v(13), s(13)] with centers (u_i, v_i), mirror (v_i, u_i).
    try:
        h = 13
        A = np.zeros((3 * n, 3 * h))
        idx = np.arange(h)
        A[idx, idx] = 1.0
        A[h + idx, h + idx] = 1.0
        A[n + idx, h + idx] = 1.0
        A[n + h + idx, idx] = 1.0
        A[2 * n + idx, 2 * h + idx] = 1.0
        A[2 * n + h + idx, 2 * h + idx] = 1.0

        def pen_w(w, mu):
            f, g = pen(A @ w, mu)
            return f, A.T @ g

        wb = [(0.0, 1.0)] * (2 * h) + [(0.0, 0.5)] * h
        ws = np.concatenate([pts[:h, 0], pts[:h, 1], np.full(h, 0.05)])
        for mu in (10.0, 1e2, 1e3, 1e4):
            rw = minimize(pen_w, ws, args=(mu,), jac=True, method="L-BFGS-B",
                          bounds=wb, options={"maxiter": 300})
            ws = rw.x
        z0 = A @ ws
    except Exception:
        pass

    try:
        zp = z0.copy()
        for mu in (10.0, 1e2, 1e3, 1e4, 1e5):
            rp = minimize(pen, zp, args=(mu,), jac=True, method="L-BFGS-B",
                          bounds=pbounds, options={"maxiter": 300})
            zp = rp.x
        z0 = zp
    except Exception:
        pass

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
    t0 = time.time()
    best = None
    best_sum = -1.0
    for seed in range(10000):
        if seed > 0 and time.time() - t0 > 200.0:
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
