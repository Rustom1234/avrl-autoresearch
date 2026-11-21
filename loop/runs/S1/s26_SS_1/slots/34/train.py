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


_PATTERNS = [
    [5, 6, 5, 6, 4], [4, 6, 5, 6, 5], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5],
    [5, 6, 5, 5, 5], [6, 5, 5, 5, 5], [4, 5, 4, 5, 4, 4], [5, 4, 5, 4, 5, 3],
    [4, 5, 5, 6, 6], [3, 5, 5, 5, 5, 3],
]


def _hexinit(rng):
    pat = _PATTERNS[rng.integers(len(_PATTERNS))]
    nr = len(pat)
    pts = []
    for k, m in enumerate(pat):
        y = (k + 0.5) / nr
        off = 0.25 / m * (1 if k % 2 else -1) * rng.uniform(0, 1.5)
        for j in range(m):
            pts.append([(j + 0.5) / m + off, y])
    c = np.array(pts, float)
    if len(c) > _N:
        c = c[rng.permutation(len(c))[:_N]]
    elif len(c) < _N:
        c = np.vstack([c, rng.uniform(0.1, 0.9, (_N - len(c), 2))])
    if rng.random() < 0.5:
        c = c[:, ::-1].copy()
    c = c + rng.normal(0, rng.uniform(0.005, 0.03), c.shape)
    return np.clip(c, 0.03, 0.97)


def _hexgrid(rng):
    nr = int(rng.choice([5, 6]))
    s = rng.uniform(0.17, 0.2)
    rs = s * np.sqrt(3) / 2
    ox, oy = rng.uniform(0, s), rng.uniform(0, rs * 2)
    pts = []
    for a in range(-3, 14):
        for b in range(-3, 14):
            x = ox + s * (b + 0.5 * (a % 2))
            y = oy + rs * a * (nr / 5.5)
            if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                pts.append([x, y])
    c = np.array(pts, float).reshape(-1, 2)
    if len(c) > _N:
        c = c[rng.permutation(len(c))[:_N]]
    elif len(c) < _N:
        c = np.vstack([c, rng.uniform(0.1, 0.9, (_N - len(c), 2))])
    if rng.random() < 0.5:
        c = c[:, ::-1].copy()
    c = c + rng.normal(0, 0.005, c.shape)
    return np.clip(c, 0.03, 0.97)


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
    while time.time() - t0 < 90:
        try:
            u = rng.random()
            c = _hexgrid(rng) if u < 0.3 else (_hexinit(rng) if u < 0.65 else _physics(rng))
            r = _lp_radii(c)
            c2 = _polish(c, r)
            c2 = np.clip(c2, 0, 1)
            r2 = _lp_radii(c2)
        except Exception:
            continue
        s = r2.sum()
        if s > best_s:
            best_s, best = s, (c2, r2)
    cur_s, cur = best_s, best
    while time.time() - t0 < 145:
        try:
            c, r = cur
            c = c.copy()
            k = int(rng.integers(1, 4))
            p = 1.0 / (r + 0.02) ** 2
            p = p / p.sum()
            idx = rng.choice(_N, size=k, replace=False, p=p)
            for q in idx:
                c[q] = rng.uniform(0.05, 0.95, 2)
            c = np.clip(c + rng.normal(0, 0.01, c.shape), 0.02, 0.98)
            r0 = _lp_radii(c)
            c2 = np.clip(_polish(c, r0), 0, 1)
            r2 = _lp_radii(c2)
        except Exception:
            continue
        s = r2.sum()
        if s > cur_s or rng.random() < np.exp((s - cur_s) / 0.002):
            cur_s, cur = s, (c2, r2)
        if s > best_s:
            best_s, best = s, (c2, r2)
    c, r = best
    r = np.maximum(r - 1e-10, 0)
    return c, r
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = optimized_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
