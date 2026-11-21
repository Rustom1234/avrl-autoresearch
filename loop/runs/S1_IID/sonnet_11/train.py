# EVOLVE-BLOCK-START
"""Multi-start SLSQP circle packing for n=26 with LP polish."""

import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
IU = np.triu_indices(N, 1)
I_IDX, J_IDX = IU
NP = len(I_IDX)


def _obj(v):
    return -np.sum(v[2 * N:])


def _obj_jac(v):
    g = np.zeros_like(v)
    g[2 * N:] = -1.0
    return g


def _cons(v):
    x = v[:N]
    y = v[N:2 * N]
    r = v[2 * N:]
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    s = r[I_IDX] + r[J_IDX]
    pair = dx * dx + dy * dy - s * s
    return np.concatenate([pair, x - r, 1 - x - r, y - r, 1 - y - r])


def _cons_jac(v):
    x = v[:N]
    y = v[N:2 * N]
    r = v[2 * N:]
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    s = r[I_IDX] + r[J_IDX]
    J = np.zeros((NP + 4 * N, 3 * N))
    k = np.arange(NP)
    J[k, I_IDX] = 2 * dx
    J[k, J_IDX] = -2 * dx
    J[k, N + I_IDX] = 2 * dy
    J[k, N + J_IDX] = -2 * dy
    J[k, 2 * N + I_IDX] = -2 * s
    J[k, 2 * N + J_IDX] = -2 * s
    a = np.arange(N)
    o = NP
    J[o + a, a] = 1
    J[o + a, 2 * N + a] = -1
    o += N
    J[o + a, a] = -1
    J[o + a, 2 * N + a] = -1
    o += N
    J[o + a, N + a] = 1
    J[o + a, 2 * N + a] = -1
    o += N
    J[o + a, N + a] = -1
    J[o + a, 2 * N + a] = -1
    return J


def _lp_radii(centers):
    """Optimal radii for fixed centers."""
    x, y = centers[:, 0], centers[:, 1]
    d = np.sqrt((x[I_IDX] - x[J_IDX]) ** 2 + (y[I_IDX] - y[J_IDX]) ** 2)
    A = np.zeros((NP, N))
    k = np.arange(NP)
    A[k, I_IDX] = 1
    A[k, J_IDX] = 1
    ub = np.minimum.reduce([x, y, 1 - x, 1 - y])
    ub = np.maximum(ub, 0.0)
    res = linprog(-np.ones(N), A_ub=A, b_ub=d,
                  bounds=[(0, u) for u in ub], method="highs")
    if res.status != 0:
        return None
    return np.maximum(res.x, 0.0)


def _valid(centers, radii):
    x, y = centers[:, 0], centers[:, 1]
    if np.any(radii < 0):
        return False
    if np.any(x - radii < 0) or np.any(x + radii > 1):
        return False
    if np.any(y - radii < 0) or np.any(y + radii > 1):
        return False
    d = np.sqrt((x[I_IDX] - x[J_IDX]) ** 2 + (y[I_IDX] - y[J_IDX]) ** 2)
    return bool(np.all(d - (radii[I_IDX] + radii[J_IDX]) >= 0))


def _finalize(centers, radii):
    centers = np.clip(centers, 0.0, 1.0)
    for eps in (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
        r = np.maximum(radii * (1 - eps) - eps, 0.0)
        if _valid(centers, r):
            return r
    return np.zeros(N)


def _init(rng, kind):
    if kind == 0:
        # perturbed hex-like rows 5,5,6,5,5
        rows = [5, 5, 6, 5, 5]
        pts = []
        for ri, c in enumerate(rows):
            yy = (ri + 0.5) / len(rows)
            for ci in range(c):
                xx = (ci + 0.5) / c
                pts.append([xx, yy])
        pts = np.array(pts) + rng.normal(0, 0.03, (N, 2))
    elif kind == 1:
        # perturbed 5x5 grid plus one extra
        pts = []
        for i in range(5):
            for j in range(5):
                pts.append([(i + 0.5) / 5, (j + 0.5) / 5])
        pts.append(rng.uniform(0.1, 0.9, 2))
        pts = np.array(pts) + rng.normal(0, 0.04, (N, 2))
    else:
        pts = rng.uniform(0.05, 0.95, (N, 2))
    pts = np.clip(pts, 0.02, 0.98)
    r = np.full(N, 0.05) * rng.uniform(0.6, 1.0, N)
    return np.concatenate([pts[:, 0], pts[:, 1], r])


def _solve(v0, maxiter=300):
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    res = minimize(_obj, v0, jac=_obj_jac, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": _cons, "jac": _cons_jac}],
                   options={"maxiter": maxiter, "ftol": 1e-12})
    return res.x


def construct_packing():
    t0 = time.time()
    budget = 170.0
    rng = np.random.default_rng(12345)
    best_sum = -1.0
    best = None
    it = 0
    last = 0.0
    while time.time() - t0 < budget - last:
        ts = time.time()
        v0 = _init(rng, it % 3 if it % 4 != 3 else 2)
        it += 1
        try:
            v = _solve(v0)
            centers = np.stack([v[:N], v[N:2 * N]], axis=1)
            centers = np.clip(centers, 0.0, 1.0)
            r = _lp_radii(centers)
            if r is None:
                continue
            s = r.sum()
            if s > best_sum:
                best_sum = s
                best = (centers.copy(), r.copy())
        except Exception:
            pass
        last = max(last * 0.5, (time.time() - ts) * 1.5)

    if best is None:
        # fallback: simple grid
        pts = []
        for i in range(5):
            for j in range(5):
                pts.append([(i + 0.5) / 5, (j + 0.5) / 5])
        pts.append([0.5, 0.5])
        centers = np.array(pts)
        r = _lp_radii(centers)
        if r is None:
            r = np.zeros(N)
        best = (centers, r)

    centers, radii = best
    radii = _finalize(centers, radii)
    return centers, radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
