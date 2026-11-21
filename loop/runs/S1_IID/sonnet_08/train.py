# EVOLVE-BLOCK-START
"""Circle packing for n=26: multi-start SLSQP on (x, y, r) with analytic Jacobians, LP polish."""

import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
IU, JU = np.triu_indices(N, 1)
NP = len(IU)
TIME_BUDGET = 200.0


def _lp_radii(centers):
    d = np.sqrt(((centers[IU] - centers[JU]) ** 2).sum(1))
    wall = np.minimum(np.minimum(centers[:, 0], centers[:, 1]),
                      np.minimum(1 - centers[:, 0], 1 - centers[:, 1]))
    wall = np.maximum(wall, 0.0)
    A = np.zeros((NP, N))
    A[np.arange(NP), IU] = 1
    A[np.arange(NP), JU] = 1
    res = linprog(-np.ones(N), A_ub=A, b_ub=d, bounds=[(0, w) for w in wall],
                  method="highs")
    if res.status == 0:
        return np.maximum(res.x, 0.0)
    return np.zeros(N)


def _fun(v):
    return -v[2 * N:].sum()


def _jac(v):
    g = np.zeros(3 * N)
    g[2 * N:] = -1.0
    return g


def _cons(v):
    x = v[:N]; y = v[N:2 * N]; r = v[2 * N:]
    dx = x[IU] - x[JU]; dy = y[IU] - y[JU]; s = r[IU] + r[JU]
    pair = dx * dx + dy * dy - s * s
    walls = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
    return np.concatenate([pair, walls])


_ar = np.arange(NP)
_idx = np.arange(N)


def _cjac(v):
    x = v[:N]; y = v[N:2 * N]; r = v[2 * N:]
    dx = x[IU] - x[JU]; dy = y[IU] - y[JU]; s = r[IU] + r[JU]
    J = np.zeros((NP + 4 * N, 3 * N))
    J[_ar, IU] = 2 * dx
    J[_ar, JU] = -2 * dx
    J[_ar, N + IU] = 2 * dy
    J[_ar, N + JU] = -2 * dy
    J[_ar, 2 * N + IU] = -2 * s
    J[_ar, 2 * N + JU] = -2 * s
    o = NP
    J[o + _idx, _idx] = 1; J[o + _idx, 2 * N + _idx] = -1
    o += N
    J[o + _idx, _idx] = -1; J[o + _idx, 2 * N + _idx] = -1
    o += N
    J[o + _idx, N + _idx] = 1; J[o + _idx, 2 * N + _idx] = -1
    o += N
    J[o + _idx, N + _idx] = -1; J[o + _idx, 2 * N + _idx] = -1
    return J


def _init(rng, k):
    mode = k % 4
    if mode == 0:
        c = rng.uniform(0.05, 0.95, (N, 2))
    elif mode == 1:
        # jittered 5x5 grid plus one extra
        g = (np.arange(5) + 0.5) / 5
        pts = np.array([(a, b) for a in g for b in g])
        pts = np.vstack([pts, rng.uniform(0.1, 0.9, (1, 2))])
        c = pts + rng.normal(0, 0.04, pts.shape)
    elif mode == 2:
        # hex-like rows: 5,5,5,5,5 -> 26 with staggering
        rows = [5, 5, 6, 5, 5]
        pts = []
        for i, m in enumerate(rows):
            yy = (i + 0.5) / len(rows)
            off = 0.0 if m == 6 else 0.5 / 6 * (1 if i % 2 else -1) * 0
            for j in range(m):
                pts.append(((j + 0.5) / m + off, yy))
        c = np.array(pts) + rng.normal(0, 0.04, (N, 2))
    else:
        # rings
        c = np.zeros((N, 2))
        c[0] = 0.5
        for i in range(8):
            a = 2 * np.pi * i / 8 + rng.uniform(0, 0.3)
            c[i + 1] = 0.5 + 0.25 * np.array([np.cos(a), np.sin(a)])
        for i in range(17):
            a = 2 * np.pi * i / 17 + rng.uniform(0, 0.3)
            c[i + 9] = 0.5 + 0.43 * np.array([np.cos(a), np.sin(a)])
        c += rng.normal(0, 0.02, c.shape)
    return np.clip(c, 0.03, 0.97)


def _solve(c0):
    r0 = _lp_radii(c0)
    v0 = np.concatenate([c0[:, 0], c0[:, 1], r0])
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    res = minimize(_fun, v0, jac=_jac, bounds=bounds, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": _cons, "jac": _cjac}],
                   options={"maxiter": 250, "ftol": 1e-12})
    v = res.x
    c = np.clip(np.stack([v[:N], v[N:2 * N]], 1), 0, 1)
    r = _lp_radii(c)
    return c, r


def _valid(c, r):
    if np.any(r < 0):
        return False
    if np.any(c[:, 0] - r < 0) or np.any(c[:, 1] - r < 0):
        return False
    if np.any(c[:, 0] + r > 1) or np.any(c[:, 1] + r > 1):
        return False
    d = np.sqrt(((c[IU] - c[JU]) ** 2).sum(1))
    return not np.any(r[IU] + r[JU] > d)


def construct_packing():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    best = None
    best_s = -1.0
    longest = 0.0
    k = 0
    while True:
        el = time.time() - t0
        if k > 0 and el + 2.0 * longest + 5 > TIME_BUDGET:
            break
        ts = time.time()
        try:
            c, r = _solve(_init(rng, k))
            s = r.sum()
            if s > best_s:
                best_s, best = s, (c, r)
        except Exception:
            pass
        longest = max(longest, time.time() - ts)
        k += 1

    if best is None:
        c = _init(rng, 1)
        best = (c, _lp_radii(c))
    c, r = best
    # shrink until strictly valid in double precision
    eps = 1e-12
    for _ in range(40):
        rr = np.maximum(r * (1 - 1e-12) - eps, 0.0)
        if _valid(c, rr):
            return c, rr
        eps *= 4
    return c, np.maximum(r * 0.999 - 1e-6, 0.0)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
