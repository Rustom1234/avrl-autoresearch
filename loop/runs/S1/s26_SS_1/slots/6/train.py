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


import time
from scipy.optimize import linprog, minimize

_N = 26
_IU = np.triu_indices(_N, 1)


def _lp_radii(c):
    n = _N
    i, j = _IU
    d = np.sqrt(((c[i] - c[j]) ** 2).sum(1))
    m = len(i)
    A = np.zeros((m, n))
    A[np.arange(m), i] = 1
    A[np.arange(m), j] = 1
    wall = np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1 - c[:, 0], 1 - c[:, 1]))
    wall = np.maximum(wall, 0)
    res = linprog(-np.ones(n), A_ub=A, b_ub=d, bounds=[(0, w) for w in wall], method="highs")
    if res.status != 0:
        return np.zeros(n)
    return res.x


_G = 120
_gx = (np.arange(_G) + 0.5) / _G
_GRID = np.stack(np.meshgrid(_gx, _gx), -1).reshape(-1, 2)


def _physics(rng):
    n = _N
    c = rng.uniform(0.05, 0.95, (n, 2))
    for t in range(20):
        d2 = ((_GRID[:, None, :] - c[None, :, :]) ** 2).sum(2)
        lab = d2.argmin(1)
        cnt = np.bincount(lab, minlength=n)
        sx = np.bincount(lab, weights=_GRID[:, 0], minlength=n)
        sy = np.bincount(lab, weights=_GRID[:, 1], minlength=n)
        ok = cnt > 0
        c[ok, 0] = sx[ok] / cnt[ok]
        c[ok, 1] = sy[ok] / cnt[ok]
    c = c + rng.normal(0, 0.01, c.shape)
    return np.clip(c, 0.02, 0.98)


def _polish(c, r):
    n = _N
    i, j = _IU
    m = len(i)
    x0 = np.concatenate([c[:, 0], c[:, 1], r])

    def obj(x):
        return -x[2 * n:].sum()

    def objg(x):
        g = np.zeros(3 * n)
        g[2 * n:] = -1
        return g

    def cons(x):
        X, Y, R = x[:n], x[n:2 * n], x[2 * n:]
        dx = X[i] - X[j]
        dy = Y[i] - Y[j]
        pair = dx * dx + dy * dy - (R[i] + R[j]) ** 2
        return np.concatenate([pair, X - R, 1 - X - R, Y - R, 1 - Y - R])

    I = np.eye(n)
    Z = np.zeros((n, n))
    wj = np.vstack([
        np.hstack([I, Z, -I]),
        np.hstack([-I, Z, -I]),
        np.hstack([Z, I, -I]),
        np.hstack([Z, -I, -I]),
    ])

    def consj(x):
        X, Y, R = x[:n], x[n:2 * n], x[2 * n:]
        dx = X[i] - X[j]
        dy = Y[i] - Y[j]
        s = R[i] + R[j]
        J = np.zeros((m, 3 * n))
        k = np.arange(m)
        J[k, i] = 2 * dx
        J[k, j] = -2 * dx
        J[k, n + i] = 2 * dy
        J[k, n + j] = -2 * dy
        J[k, 2 * n + i] = -2 * s
        J[k, 2 * n + j] = -2 * s
        return np.vstack([J, wj])

    res = minimize(obj, x0, jac=objg, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": cons, "jac": consj}],
                   bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                   options={"maxiter": 300, "ftol": 1e-12})
    x = res.x
    return np.stack([x[:n], x[n:2 * n]], 1)


def optimized_packing():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    best_s, best = -1, None
    while time.time() - t0 < 150:
        try:
            c = _physics(rng)
            r = _lp_radii(c)
            c2 = _polish(c, r)
            c2 = np.clip(c2, 0, 1)
            r2 = _lp_radii(c2)
        except Exception:
            continue
        s = r2.sum()
        if s > best_s:
            best_s, best = s, (c2, r2)
    c, r = best
    r0 = np.maximum(r - 1e-10, 0)
    try:
        c1, r1 = _newton(c, r)
        if r1.sum() > r0.sum():
            return c1, r1
    except Exception:
        pass
    return c, r0


def _feasible_shrink(c, r, eps=1e-13):
    n = _N
    i, j = _IU
    r = np.minimum(r, np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1 - c[:, 0], 1 - c[:, 1])))
    r = np.maximum(r, 0)
    d = np.sqrt(((c[i] - c[j]) ** 2).sum(1))
    for _ in range(5):
        v = r[i] + r[j] - d
        bad = np.where(v > -eps)[0]
        if len(bad) == 0:
            break
        for k in bad:
            s = (v[k] + eps) / 2
            r[i[k]] -= s
            r[j[k]] -= s
        r = np.maximum(r, 0)
        v = r[i] + r[j] - d
    return r


def _newton(c, r):
    n = _N
    i, j = _IU
    X, Y, R = c[:, 0].copy(), c[:, 1].copy(), r.copy()
    d = np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
    pa = np.where(np.abs(d - R[i] - R[j]) < 1e-9)[0]
    wa = []
    for k in range(n):
        for t, val in enumerate((X[k], Y[k], 1 - X[k], 1 - Y[k])):
            if abs(val - R[k]) < 1e-9:
                wa.append((k, t))
    for _ in range(6):
        F = []
        J = []
        for k in pa:
            a, b = i[k], j[k]
            dx, dy = X[a] - X[b], Y[a] - Y[b]
            dd = np.hypot(dx, dy)
            F.append(dd - R[a] - R[b])
            row = np.zeros(3 * n)
            row[a], row[b] = dx / dd, -dx / dd
            row[n + a], row[n + b] = dy / dd, -dy / dd
            row[2 * n + a] = row[2 * n + b] = -1
            J.append(row)
        for k, t in wa:
            row = np.zeros(3 * n)
            if t == 0:
                f = X[k] - R[k]; row[k] = 1
            elif t == 1:
                f = Y[k] - R[k]; row[n + k] = 1
            elif t == 2:
                f = 1 - X[k] - R[k]; row[k] = -1
            else:
                f = 1 - Y[k] - R[k]; row[n + k] = -1
            row[2 * n + k] = -1
            F.append(f)
            J.append(row)
        if not F:
            break
        step = np.linalg.lstsq(np.array(J), -np.array(F), rcond=None)[0]
        X += step[:n]
        Y += step[n:2 * n]
        R += step[2 * n:]
    c1 = np.clip(np.stack([X, Y], 1), 0, 1)
    r1 = _feasible_shrink(c1, R, 1e-13)
    r1 = np.maximum(r1 - 1e-13, 0)
    return c1, r1
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = optimized_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
