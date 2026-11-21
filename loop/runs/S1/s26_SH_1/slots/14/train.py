# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""

import numpy as np


import time
from scipy.optimize import minimize, linprog

_N = 26
_I, _J = np.triu_indices(_N, 1)


def _cons_fun(z):
    n = _N
    x = z[:n]
    y = z[n:2 * n]
    r = z[2 * n:]
    I, J = _I, _J
    dx = x[I] - x[J]
    dy = y[I] - y[J]
    rs = r[I] + r[J]
    pair = dx * dx + dy * dy - rs * rs
    border = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
    return np.concatenate([border, pair])


def _cons_jac(z):
    n = _N
    I, J = _I, _J
    m = len(I)
    x = z[:n]
    y = z[n:2 * n]
    r = z[2 * n:]
    Jm = np.zeros((4 * n + m, 3 * n))
    k = np.arange(n)
    Jm[k, k] = 1.0
    Jm[k, 2 * n + k] = -1.0
    Jm[n + k, k] = -1.0
    Jm[n + k, 2 * n + k] = -1.0
    Jm[2 * n + k, n + k] = 1.0
    Jm[2 * n + k, 2 * n + k] = -1.0
    Jm[3 * n + k, n + k] = -1.0
    Jm[3 * n + k, 2 * n + k] = -1.0
    rows = 4 * n + np.arange(m)
    dx = x[I] - x[J]
    dy = y[I] - y[J]
    rs = r[I] + r[J]
    Jm[rows, I] = 2 * dx
    Jm[rows, J] = -2 * dx
    Jm[rows, n + I] = 2 * dy
    Jm[rows, n + J] = -2 * dy
    Jm[rows, 2 * n + I] = -2 * rs
    Jm[rows, 2 * n + J] = -2 * rs
    return Jm


def _obj(z):
    return -np.sum(z[2 * _N:])


def _obj_grad(z):
    g = np.zeros(3 * _N)
    g[2 * _N:] = -1.0
    return g


def _lp_radii(x, y):
    """Exact max-sum radii for fixed centers (LP), shrunk by a tiny margin."""
    n = _N
    I, J = _I, _J
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    d = np.sqrt((x[I] - x[J]) ** 2 + (y[I] - y[J]) ** 2)
    A = np.zeros((len(I), n))
    rows = np.arange(len(I))
    A[rows, I] = 1.0
    A[rows, J] = 1.0
    bmax = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    res = linprog(-np.ones(n), A_ub=A, b_ub=d,
                  bounds=[(0.0, float(b)) for b in bmax], method="highs")
    if res.status != 0:
        return None
    r = np.minimum(np.maximum(res.x - 1e-7, 0.0), bmax)
    for _ in range(20):
        viol = np.max(r[I] + r[J] - d)
        if viol <= 0:
            break
        r = np.maximum(r - viol, 0.0)
    return r


def _optimize(x0, y0, r0):
    z = np.concatenate([x0, y0, r0])
    cons = [{"type": "ineq", "fun": _cons_fun, "jac": _cons_jac}]
    # staged radius growth: cap radii at 0.08, 0.10, 0.12, then uncapped,
    # warm-starting each SLSQP stage from the previous solution
    for cap in (0.08, 0.10, 0.12, None):
        hi = None if cap is None else cap
        bnds = [(None, None)] * (2 * _N) + [(0.0, hi)] * _N
        z_prev = z
        try:
            res = minimize(_obj, z_prev, jac=_obj_grad, constraints=cons,
                           bounds=bnds, method="SLSQP",
                           options={"maxiter": 300, "ftol": 1e-12})
            z_new = res.x
        except Exception:
            z_new = z_prev
        if np.all(np.isfinite(z_new)):
            z = z_new
        else:
            z = z_prev
    if not np.all(np.isfinite(z)):
        z = np.concatenate([x0, y0, r0])
    return z[:_N], z[_N:2 * _N]


def _seed(rng):
    # 4 large corner circles, 8 medium mid-edge circles, small interior circles;
    # sizes are randomized per seed
    pts = [(0.13, 0.13), (0.87, 0.13), (0.13, 0.87), (0.87, 0.87)]
    rad = [rng.uniform(0.10, 0.15) for _ in range(4)]
    for t in (0.3, 0.7):
        pts += [(t, 0.08), (t, 0.92), (0.08, t), (0.92, t)]
        rad += [rng.uniform(0.06, 0.10) for _ in range(4)]
    pts = np.array(pts)
    n_int = _N - len(pts)
    interior = rng.uniform(0.2, 0.8, size=(n_int, 2))
    P = np.vstack([pts, interior])
    R = np.concatenate([np.array(rad), rng.uniform(0.03, 0.06, size=n_int)])
    return P[:, 0].copy(), P[:, 1].copy(), R


def _lloyd_seed(rng):
    # random points relaxed by 20 Lloyd iterations on a grid approximation
    n = _N
    g = np.linspace(0.0, 1.0, 60)
    GX, GY = np.meshgrid(g, g)
    G = np.column_stack([GX.ravel(), GY.ravel()])
    P = rng.uniform(0.05, 0.95, size=(n, 2))
    for _ in range(20):
        d2 = ((G[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        lab = np.argmin(d2, axis=1)
        for k in range(n):
            m = lab == k
            if np.any(m):
                P[k] = G[m].mean(axis=0)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    r = 0.5 * D.min(axis=1)
    r = np.minimum(r, np.min(np.column_stack([P, 1 - P]), axis=1))
    r = np.maximum(r, 0.01)
    return P[:, 0].copy(), P[:, 1].copy(), r


def _ring_seed(rng):
    # concentric rings from a random partition of 26; each ring gets a random rotation
    parts = [(1, 6, 12, 7), (1, 7, 9, 9), (4, 8, 14), (1, 6, 10, 9), (2, 8, 16)]
    part = parts[int(rng.integers(len(parts)))]
    k = len(part)
    pts = []
    for idx, cnt in enumerate(part):
        if cnt == 1:
            pts.append((0.5, 0.5))
            continue
        rho = 0.42 * (idx + 1) / k
        off = rng.uniform(0.0, 2 * np.pi)
        for j in range(cnt):
            a = off + 2 * np.pi * j / cnt
            pts.append((0.5 + rho * np.cos(a), 0.5 + rho * np.sin(a)))
    P = np.array(pts) + rng.normal(0.0, 0.01, size=(_N, 2))
    return np.clip(P, 0.05, 0.95)


def construct_packing():
    rng = np.random.default_rng(12345)
    best = None
    t0 = time.time()
    for _k in range(100000):
        if time.time() - t0 > 200.0 and best is not None:
            break
        if _k % 2 == 0:
            P = _ring_seed(rng)
        else:
            P = rng.uniform(0.05, 0.95, size=(_N, 2))
        x0 = P[:, 0].copy()
        y0 = P[:, 1].copy()
        r0 = np.full(_N, 0.05)
        x, y = _optimize(x0, y0, r0)
        r = _lp_radii(x, y)
        if r is None:
            continue
        s = float(np.sum(r))
        if best is None or s > best[0]:
            centers = np.column_stack([np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0)])
            best = (s, centers, r)
    if best is None:
        centers = np.column_stack(_seed(rng)[:2])
        radii = np.zeros(_N)
        return centers, radii
    return best[1], best[2]


def _legacy_construct_packing():
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
