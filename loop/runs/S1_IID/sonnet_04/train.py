# EVOLVE-BLOCK-START
"""Multi-start SLSQP circle packing for n=26 (maximize sum of radii)."""

import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
IU, JU = np.triu_indices(N, 1)
NP_ = len(IU)
ROWS = np.arange(NP_)


def _obj(v):
    return -np.sum(v[2 * N:])


def _obj_grad(v):
    g = np.zeros(3 * N)
    g[2 * N:] = -1.0
    return g


def _cons(v):
    x = v[:N]
    y = v[N:2 * N]
    r = v[2 * N:]
    dx = x[IU] - x[JU]
    dy = y[IU] - y[JU]
    sr = r[IU] + r[JU]
    pair = dx * dx + dy * dy - sr * sr
    return np.concatenate([pair, x - r, 1 - x - r, y - r, 1 - y - r])


def _jac(v):
    x = v[:N]
    y = v[N:2 * N]
    r = v[2 * N:]
    dx = x[IU] - x[JU]
    dy = y[IU] - y[JU]
    sr = r[IU] + r[JU]
    J = np.zeros((NP_ + 4 * N, 3 * N))
    J[ROWS, IU] = 2 * dx
    J[ROWS, JU] = -2 * dx
    J[ROWS, N + IU] = 2 * dy
    J[ROWS, N + JU] = -2 * dy
    J[ROWS, 2 * N + IU] = -2 * sr
    J[ROWS, 2 * N + JU] = -2 * sr
    k = np.arange(N)
    o = NP_
    J[o + k, k] = 1
    J[o + k, 2 * N + k] = -1
    o += N
    J[o + k, k] = -1
    J[o + k, 2 * N + k] = -1
    o += N
    J[o + k, N + k] = 1
    J[o + k, 2 * N + k] = -1
    o += N
    J[o + k, N + k] = -1
    J[o + k, 2 * N + k] = -1
    return J


def _initial(rng, mode):
    if mode == 0:
        pts = rng.uniform(0.05, 0.95, (N, 2))
    elif mode == 1:
        # perturbed 5x5 grid plus one extra
        g = (np.arange(5) + 0.5) / 5
        pts = np.array([(a, b) for a in g for b in g])
        pts = np.vstack([pts, rng.uniform(0.1, 0.9, (1, 2))])
        pts = pts + rng.normal(0, 0.03, pts.shape)
    else:
        # staggered rows 5,5,6,5,5 style / 6,5,5,5,5 permuted
        counts = [5, 5, 6, 5, 5]
        rng.shuffle(counts)
        pts = []
        for i, c in enumerate(counts):
            yy = (i + 0.5) / 5
            for k in range(c):
                pts.append(((k + 0.5) / c, yy))
        pts = np.array(pts) + rng.normal(0, 0.03, (N, 2))
    pts = np.clip(pts, 0.03, 0.97)
    r0 = np.full(N, 0.07) * rng.uniform(0.7, 1.1, N)
    return np.concatenate([pts[:, 0], pts[:, 1], r0])


def _lp_radii(c):
    d = np.sqrt(((c[IU] - c[JU]) ** 2).sum(1))
    A = np.zeros((NP_, N))
    A[ROWS, IU] = 1
    A[ROWS, JU] = 1
    border = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]),
                        np.minimum(c[:, 1], 1 - c[:, 1]))
    border = np.maximum(border, 0)
    res = linprog(-np.ones(N), A_ub=A, b_ub=d,
                  bounds=[(0, b) for b in border], method="highs")
    if res.status == 0:
        return np.maximum(res.x, 0)
    return None


def _valid(c, r):
    if np.any(r < 0):
        return False
    if np.any(c[:, 0] - r < 0) or np.any(c[:, 0] + r > 1):
        return False
    if np.any(c[:, 1] - r < 0) or np.any(c[:, 1] + r > 1):
        return False
    d = np.sqrt(((c[IU] - c[JU]) ** 2).sum(1))
    return not np.any(r[IU] + r[JU] > d)


def _finalize(c, r):
    c = np.clip(c, 0.0, 1.0)
    for eps in (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6):
        rr = np.maximum(r * (1 - eps) - eps, 0.0)
        if _valid(c, rr):
            return rr
    return np.zeros(N)


def construct_packing():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    best_v, best_s = None, -1.0
    it = 0
    while time.time() - t0 < 170:
        v0 = _initial(rng, it % 3)
        it += 1
        try:
            res = minimize(
                _obj, v0, jac=_obj_grad, method="SLSQP",
                bounds=[(0, 1)] * (2 * N) + [(0, 0.5)] * N,
                constraints=[{"type": "ineq", "fun": _cons, "jac": _jac}],
                options={"maxiter": 300, "ftol": 1e-12},
            )
        except Exception:
            continue
        v = res.x
        if np.min(_cons(v)) < -1e-6:
            continue
        s = v[2 * N:].sum()
        if s > best_s:
            best_s, best_v = s, v
    if best_v is None:
        best_v = _initial(rng, 1)
        best_v[2 * N:] = 0.02
    c = np.clip(np.stack([best_v[:N], best_v[N:2 * N]], axis=1), 0, 1)
    r = _lp_radii(c)
    if r is None:
        r = np.maximum(best_v[2 * N:], 0)
    r = _finalize(c, r)
    return c, r


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
