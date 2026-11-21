# EVOLVE-BLOCK-START
"""Multi-start SLSQP circle packing for n=26 (maximize sum of radii)."""

import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
_I, _J = np.triu_indices(N, 1)
_P = len(_I)
_ar = np.arange(_P)


def _cons(v):
    x, y, r = v[:N], v[N:2 * N], v[2 * N:]
    dx = x[_I] - x[_J]
    dy = y[_I] - y[_J]
    s = r[_I] + r[_J]
    pair = dx * dx + dy * dy - s * s
    return np.concatenate([pair, x - r, 1 - x - r, y - r, 1 - y - r])


def _jac(v):
    x, y, r = v[:N], v[N:2 * N], v[2 * N:]
    dx = x[_I] - x[_J]
    dy = y[_I] - y[_J]
    s = r[_I] + r[_J]
    M = 4 * N + _P
    Jm = np.zeros((M, 3 * N))
    Jm[_ar, _I] = 2 * dx
    Jm[_ar, _J] = -2 * dx
    Jm[_ar, N + _I] = 2 * dy
    Jm[_ar, N + _J] = -2 * dy
    Jm[_ar, 2 * N + _I] = -2 * s
    Jm[_ar, 2 * N + _J] = -2 * s
    k = np.arange(N)
    o = _P
    Jm[o + k, k] = 1; Jm[o + k, 2 * N + k] = -1
    o += N
    Jm[o + k, k] = -1; Jm[o + k, 2 * N + k] = -1
    o += N
    Jm[o + k, N + k] = 1; Jm[o + k, 2 * N + k] = -1
    o += N
    Jm[o + k, N + k] = -1; Jm[o + k, 2 * N + k] = -1
    return Jm


_OBJ_G = np.concatenate([np.zeros(2 * N), -np.ones(N)])
_BOUNDS = [(0, 1)] * (2 * N) + [(0, 0.5)] * N


def _solve(c0, maxiter=250):
    v0 = np.concatenate([c0[:, 0], c0[:, 1], np.full(N, 0.05)])
    try:
        res = minimize(lambda v: -np.sum(v[2 * N:]), v0, jac=lambda v: _OBJ_G,
                       bounds=_BOUNDS, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": _cons, "jac": _jac}],
                       options={"maxiter": maxiter, "ftol": 1e-12})
    except Exception:
        return None, -1.0
    v = res.x
    if not np.all(np.isfinite(v)):
        return None, -1.0
    c = _cons(v)
    viol = max(0.0, -c.min())
    return v, np.sum(v[2 * N:]) - 10 * viol


def _init(rng, kind):
    if kind == 0:
        return rng.uniform(0.05, 0.95, (N, 2))
    if kind == 1:
        g = np.array([[(i + 0.5) / 5, (j + 0.5) / 5] for i in range(5) for j in range(5)])
        g = np.vstack([g, [[0.5, 0.5]]])
        return np.clip(g + rng.normal(0, 0.04, g.shape), 0.03, 0.97)
    # hex-like rows with jitter
    rows = [5, 6, 5, 5, 5]
    pts = []
    for k, m in enumerate(rows):
        y = (k + 0.5) / len(rows)
        off = 0.0 if m == 6 else 0.5 / 6 * (k % 2)
        for t in range(m):
            pts.append([(t + 0.5) / m + off * 0.3, y])
    pts = np.array(pts[:N])
    return np.clip(pts + rng.normal(0, 0.05, pts.shape), 0.03, 0.97)


def _lp_radii(centers):
    n = len(centers)
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    A = np.zeros((len(_I), n))
    A[_ar, _I] = 1
    A[_ar, _J] = 1
    b = d[_I, _J]
    ub = np.minimum.reduce([centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]])
    ub = np.maximum(ub, 0)
    res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=list(zip(np.zeros(n), ub)),
                  method="highs")
    if res.status != 0:
        return None
    return res.x


def _valid(centers, radii):
    if np.any(radii < 0):
        return False
    if np.any(centers[:, 0] - radii < 0) or np.any(centers[:, 0] + radii > 1):
        return False
    if np.any(centers[:, 1] - radii < 0) or np.any(centers[:, 1] + radii > 1):
        return False
    dx = centers[_I, 0] - centers[_J, 0]
    dy = centers[_I, 1] - centers[_J, 1]
    d = np.sqrt(dx * dx + dy * dy)
    return bool(np.all(d >= radii[_I] + radii[_J]))


def _finalize(centers):
    centers = np.clip(centers, 0.0, 1.0)
    r = _lp_radii(centers)
    if r is None:
        return None
    r = np.maximum(r - 1e-12, 0.0)
    for _ in range(200):
        if _valid(centers, r):
            return r
        r = np.maximum(r - 2e-11, 0.0)
    return None


def construct_packing():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    budget = 150.0
    best = None
    best_s = -1.0
    it = 0
    while time.time() - t0 < budget:
        kind = it % 3
        it += 1
        v, s = _solve(_init(rng, kind))
        if v is None:
            continue
        if s > best_s - 0.02:
            centers = np.stack([v[:N], v[N:2 * N]], axis=1)
            r = _finalize(centers)
            if r is not None:
                tot = r.sum()
                if tot > best_s:
                    best_s = tot
                    best = (centers, r)
    if best is None:
        # safe fallback: 5x5 grid + tiny circle
        g = [[(i + 0.5) / 5, (j + 0.5) / 5] for i in range(5) for j in range(5)]
        g.append([0.0, 0.0])
        centers = np.array(g)
        r = np.full(N, 0.1 - 1e-9)
        r[-1] = 0.0
        return centers, r
    return best


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
