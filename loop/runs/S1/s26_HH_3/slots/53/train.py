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
    return _best_packing()


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


import time
from scipy.optimize import minimize


def _repair(centers, radii):
    """Shrink radii so every circle is inside the square and no pair overlaps."""
    n = len(radii)
    c = np.clip(centers, 0.0, 1.0)
    r = np.maximum(np.asarray(radii, dtype=float), 0.0)
    b = np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1 - c[:, 0], 1 - c[:, 1]))
    r = np.minimum(r, np.maximum(b, 0.0))
    for _ in range(1000):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
                s = r[i] + r[j]
                if s > d * (1 - 1e-10):
                    f = d * (1 - 1e-10) / s if s > 0 else 0.0
                    r[i] *= f
                    r[j] *= f
                    changed = True
        if not changed:
            break
    r = r * (1 - 1e-9)
    return np.maximum(r, 0.0)


def _optimize(centers0, radii0):
    """Joint SLSQP over (x, y, r): maximize sum r subject to containment and non-overlap."""
    n = len(radii0)
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    idx = np.arange(n)

    def unpack(z):
        return z[:n], z[n:2 * n], z[2 * n:]

    def obj(z):
        return -np.sum(z[2 * n:])

    def obj_grad(z):
        g = np.zeros_like(z)
        g[2 * n:] = -1.0
        return g

    def cons(z):
        x, y, r = unpack(z)
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        pair = dx * dx + dy * dy - (r[iu] + r[ju]) ** 2
        bnd = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
        return np.concatenate([pair, bnd])

    def cons_jac(z):
        x, y, r = unpack(z)
        J = np.zeros((m + 4 * n, 3 * n))
        rows = np.arange(m)
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        s = r[iu] + r[ju]
        J[rows, iu] = 2 * dx
        J[rows, ju] = -2 * dx
        J[rows, n + iu] = 2 * dy
        J[rows, n + ju] = -2 * dy
        J[rows, 2 * n + iu] = -2 * s
        J[rows, 2 * n + ju] = -2 * s
        J[m + idx, idx] = 1
        J[m + idx, 2 * n + idx] = -1
        J[m + n + idx, idx] = -1
        J[m + n + idx, 2 * n + idx] = -1
        J[m + 2 * n + idx, n + idx] = 1
        J[m + 2 * n + idx, 2 * n + idx] = -1
        J[m + 3 * n + idx, n + idx] = -1
        J[m + 3 * n + idx, 2 * n + idx] = -1
        return J

    z0 = np.concatenate([centers0[:, 0], centers0[:, 1], radii0])
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    res = minimize(obj, z0, jac=obj_grad, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
                   bounds=bounds, options={"maxiter": 300, "ftol": 1e-12})
    x, y, r = unpack(res.x)
    return np.column_stack([x, y]), r


def _relax(centers, radii, iters=150):
    """Repulsion relaxation: per-circle step scaled by radius (big circles move less),
    radii reset to the feasible max then grown slightly, decaying global step,
    stop once the sum of radii stabilises."""
    n = len(radii)
    c = np.clip(np.asarray(centers, dtype=float).copy(), 0.0, 1.0)
    r = np.maximum(np.asarray(radii, dtype=float).copy(), 0.0)
    step = 0.02
    prev = -1.0
    t0 = time.time()
    for _ in range(iters):
        if time.time() - t0 > 20:
            break
        b = np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1 - c[:, 0], 1 - c[:, 1]))
        r = np.maximum(np.minimum(r, b), 0.0)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
                s = r[i] + r[j]
                if s > d and s > 0:
                    f = d / s
                    r[i] *= f
                    r[j] *= f
        total = float(np.sum(r))
        if abs(total - prev) < 1e-7:
            break
        prev = total
        disp = np.zeros_like(c)
        rmean = float(np.mean(r)) + 1e-12
        for i in range(n):
            for j in range(i + 1, n):
                dv = c[i] - c[j]
                d = np.hypot(dv[0], dv[1]) + 1e-12
                gap = d - (r[i] + r[j])
                if gap < 0.05:
                    push = (0.05 - gap) * dv / d
                    fi = min(rmean / max(r[i], 1e-6), 3.0)
                    fj = min(rmean / max(r[j], 1e-6), 3.0)
                    disp[i] += step * push * fi
                    disp[j] -= step * push * fj
        c = np.clip(c + disp, 0.0, 1.0)
        r = r * 1.01
        step *= 0.97
    b = np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1 - c[:, 0], 1 - c[:, 1]))
    r = np.maximum(np.minimum(r, b), 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
            s = r[i] + r[j]
            if s > d and s > 0:
                f = d / s
                r[i] *= f
                r[j] *= f
    return c, r


def _sym_layout(k, m):
    """Affine map z = A @ v + b from reduced vars v to full (x, y, r) of length 3n.
    v = [x(k), y(k), r(k), ya(m), ra(m)]; mirror of left circle i is circle k+i at (1-x, y, r);
    axis circles sit at x = 0.5."""
    n = 2 * k + m
    nv = 3 * k + 2 * m
    A = np.zeros((3 * n, nv))
    b = np.zeros(3 * n)
    for i in range(k):
        A[i, i] = 1.0
        A[n + i, k + i] = 1.0
        A[2 * n + i, 2 * k + i] = 1.0
        j = k + i
        A[j, i] = -1.0
        b[j] = 1.0
        A[n + j, k + i] = 1.0
        A[2 * n + j, 2 * k + i] = 1.0
    for a in range(m):
        j = 2 * k + a
        b[j] = 0.5
        A[n + j, 3 * k + a] = 1.0
        A[2 * n + j, 3 * k + m + a] = 1.0
    return A, b, n


def _sym_search(rng):
    """Mirror-symmetric SLSQP over reduced variables, then expand to 26 circles."""
    k, m = (13, 0) if rng.random() < 0.5 else (12, 2)
    A, b, n = _sym_layout(k, m)
    cl = rng.uniform(0.08, 0.45, size=(k, 2))
    ca = rng.uniform(0.2, 0.8, size=m)
    ra = rng.uniform(0.06, 0.1, size=m)
    rl = rng.uniform(0.06, 0.1, size=k)
    v0 = np.concatenate([cl[:, 0], cl[:, 1], rl, ca, ra])
    nv = len(v0)
    iu, ju = np.triu_indices(n, 1)
    mp = len(iu)
    idx = np.arange(n)

    def expand(v):
        return A @ v + b

    def obj(v):
        return -np.sum(expand(v)[2 * n:])

    def obj_grad(v):
        return -A[2 * n:].sum(axis=0)

    def cons(v):
        z = expand(v)
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        pair = dx * dx + dy * dy - (r[iu] + r[ju]) ** 2
        bnd = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
        return np.concatenate([pair, bnd])

    def cons_jac(v):
        z = expand(v)
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        J = np.zeros((mp + 4 * n, 3 * n))
        rows = np.arange(mp)
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        s = r[iu] + r[ju]
        J[rows, iu] = 2 * dx
        J[rows, ju] = -2 * dx
        J[rows, n + iu] = 2 * dy
        J[rows, n + ju] = -2 * dy
        J[rows, 2 * n + iu] = -2 * s
        J[rows, 2 * n + ju] = -2 * s
        J[mp + idx, idx] = 1
        J[mp + idx, 2 * n + idx] = -1
        J[mp + n + idx, idx] = -1
        J[mp + n + idx, 2 * n + idx] = -1
        J[mp + 2 * n + idx, n + idx] = 1
        J[mp + 2 * n + idx, 2 * n + idx] = -1
        J[mp + 3 * n + idx, n + idx] = -1
        J[mp + 3 * n + idx, 2 * n + idx] = -1
        return J @ A

    bounds = [(0.0, 1.0)] * (2 * k) + [(0.0, 0.5)] * k + [(0.0, 1.0)] * m + [(0.0, 0.5)] * m
    res = minimize(obj, v0, jac=obj_grad, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
                   bounds=bounds, options={"maxiter": 300, "ftol": 1e-12})
    z = expand(res.x)
    centers = np.column_stack([z[:n], z[n:2 * n]])
    radii = np.maximum(z[2 * n:], 0.0)
    return np.clip(centers, 0.0, 1.0), radii


def _best_packing():
    """Multi-start over bimodal radius initializations; keep the best feasible packing."""
    n = 26
    rng = np.random.default_rng(12345)
    t0 = time.time()
    best_sum, best_c, best_r = -1.0, None, None
    # Mirror-symmetric phase: reduced SLSQP, then one unconstrained polish and repair.
    for _ in range(8):
        if time.time() - t0 > 40:
            break
        try:
            c, r = _sym_search(rng)
            c, r = _optimize(c, r)
            r = _repair(c, r)
        except Exception:
            continue
        s = float(np.sum(r))
        if s > best_sum:
            best_sum, best_c, best_r = s, c, r
    for start in range(12):
        k = (4, 6, 8, 10)[start % 4]
        if time.time() - t0 > 150:
            break
        if start >= 10:
            # Two-tier radial profile: large seeds on a ring, small seeds scattered outside
            kk = int(rng.integers(3, 6))
            ang = np.linspace(0.0, 2 * np.pi, kk, endpoint=False) + rng.uniform(0.0, 2 * np.pi)
            rr = rng.uniform(0.3, 0.35)
            inner = np.column_stack([0.5 + rr * np.cos(ang), 0.5 + rr * np.sin(ang)])
            outer = rng.uniform(0.05, 0.95, size=(n - kk, 2))
            c0 = np.clip(np.vstack([inner, outer]), 0.0, 1.0)
            r0 = np.concatenate([np.full(kk, 0.13), np.full(n - kk, 0.06)])
            r0 = r0 * rng.uniform(0.9, 1.1, size=n)
        else:
            c0 = rng.uniform(0.05, 0.95, size=(n, 2))
            r0 = np.full(n, 0.07)
            r0[:k] = 0.12
            r0 = r0 * rng.uniform(0.9, 1.1, size=n)
        try:
            c0, r0 = _relax(c0, r0)
            c, r = _optimize(c0, r0)
            r = _repair(c, r)
        except Exception:
            continue
        s = float(np.sum(r))
        if s > best_sum:
            best_sum, best_c, best_r = s, c, r
    # Basin hopping from the best packing: perturb 1-3 centers, re-solve, Metropolis accept
    if best_c is not None:
        cur_c, cur_r, cur_s = best_c, best_r, best_sum
        for _ in range(200):
            if time.time() - t0 > 200:
                break
            c0 = cur_c.copy()
            k = int(rng.integers(1, 4))
            pick = rng.choice(n, size=k, replace=False)
            c0[pick] = np.clip(c0[pick] + rng.normal(0.0, 0.1, size=(k, 2)), 0.0, 1.0)
            try:
                c, r = _optimize(c0, np.maximum(cur_r, 0.01))
                r = _repair(c, r)
            except Exception:
                continue
            s = float(np.sum(r))
            if s > best_sum:
                best_sum, best_c, best_r = s, c, r
            if s > cur_s or rng.random() < np.exp((s - cur_s) / 0.01):
                cur_c, cur_r, cur_s = c, r, s
    if best_c is None:
        c = rng.uniform(0.05, 0.95, size=(n, 2))
        best_c = np.clip(c, 0.0, 1.0)
        best_r = _repair(best_c, np.zeros(n))
    return np.clip(best_c, 0.0, 1.0), best_r


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
