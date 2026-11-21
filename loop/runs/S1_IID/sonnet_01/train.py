# EVOLVE-BLOCK-START
"""Multi-start SLSQP circle packing for n=26 (maximize sum of radii)."""

import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
II, JJ = np.triu_indices(N, 1)
NP = len(II)
ar = np.arange(N)
P = np.arange(NP)

# constant jacobian part for linear border constraints
_LIN = np.zeros((4 * N, 3 * N))
for k in range(N):
    _LIN[k, k] = 1; _LIN[k, 2 * N + k] = -1                 # x - r >= 0
    _LIN[N + k, k] = -1; _LIN[N + k, 2 * N + k] = -1        # 1 - x - r >= 0
    _LIN[2 * N + k, N + k] = 1; _LIN[2 * N + k, 2 * N + k] = -1
    _LIN[3 * N + k, N + k] = -1; _LIN[3 * N + k, 2 * N + k] = -1
_LINC = np.concatenate([np.zeros(N), np.ones(N), np.zeros(N), np.ones(N)])


def _cons(v):
    x, y, r = v[:N], v[N:2 * N], v[2 * N:]
    dx = x[II] - x[JJ]
    dy = y[II] - y[JJ]
    s = r[II] + r[JJ]
    pair = dx * dx + dy * dy - s * s
    return np.concatenate([pair, _LIN @ v + _LINC])


def _jac(v):
    x, y, r = v[:N], v[N:2 * N], v[2 * N:]
    dx = x[II] - x[JJ]
    dy = y[II] - y[JJ]
    s = r[II] + r[JJ]
    J = np.zeros((NP + 4 * N, 3 * N))
    J[P, II] = 2 * dx
    J[P, JJ] = -2 * dx
    J[P, N + II] = 2 * dy
    J[P, N + JJ] = -2 * dy
    J[P, 2 * N + II] = -2 * s
    J[P, 2 * N + JJ] = -2 * s
    J[NP:] = _LIN
    return J


def _obj(v):
    return -np.sum(v[2 * N:])


_g = np.concatenate([np.zeros(2 * N), -np.ones(N)])


def _grad(v):
    return _g


def _solve(x0, y0, r0, maxiter=300):
    v0 = np.concatenate([x0, y0, r0])
    bnds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    try:
        res = minimize(_obj, v0, jac=_grad, method="SLSQP", bounds=bnds,
                       constraints=[{"type": "ineq", "fun": _cons, "jac": _jac}],
                       options={"maxiter": maxiter, "ftol": 1e-12})
    except Exception:
        return None
    v = res.x
    if not np.all(np.isfinite(v)):
        return None
    return v


def _lp_radii(c):
    """Best radii for fixed centers via LP."""
    d = np.sqrt(((c[II] - c[JJ]) ** 2).sum(1))
    A = np.zeros((NP, N))
    A[P, II] = 1
    A[P, JJ] = 1
    ub = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
    ub = np.maximum(ub, 0)
    res = linprog(-np.ones(N), A_ub=A, b_ub=d, bounds=list(zip(np.zeros(N), ub)),
                  method="highs")
    if res.status != 0:
        return None
    return np.maximum(res.x, 0)


def _valid(c, r):
    if np.any(r < 0):
        return False
    if np.any(c[:, 0] - r < 0) or np.any(c[:, 0] + r > 1):
        return False
    if np.any(c[:, 1] - r < 0) or np.any(c[:, 1] + r > 1):
        return False
    d = np.sqrt(((c[II] - c[JJ]) ** 2).sum(1))
    return bool(np.all(r[II] + r[JJ] <= d))


def _starts(rng, k):
    if k % 3 == 0:
        c = rng.random((N, 2))
    elif k % 3 == 1:
        # jittered row layout, e.g. rows of 5,5,6,5,5 (shuffled variants)
        rows = [5, 5, 6, 5, 5]
        if k % 2 == 0:
            rows = [6, 5, 5, 5, 5]
            rng.shuffle(rows)
            rows = list(rows)
            rows[0] += 0
        tot = sum(rows)
        while tot < N:
            rows[rng.integers(len(rows))] += 1; tot += 1
        while tot > N:
            i = rng.integers(len(rows))
            if rows[i] > 1:
                rows[i] -= 1; tot -= 1
        pts = []
        m = len(rows)
        for i, cnt in enumerate(rows):
            yy = (i + 0.5) / m
            for j in range(cnt):
                pts.append([(j + 0.5 + 0.3 * (i % 2 - 0.5)) / cnt, yy])
        c = np.array(pts) + rng.normal(0, 0.03, (N, 2))
    else:
        # perturbed 5x5 grid plus one extra
        g = [[(i + 0.5) / 5, (j + 0.5) / 5] for i in range(5) for j in range(5)]
        g.append(rng.random(2))
        c = np.array(g) + rng.normal(0, 0.04, (N, 2))
    c = np.clip(c, 0.03, 0.97)
    r = np.full(N, 0.05) * (0.5 + rng.random(N))
    return c[:, 0].copy(), c[:, 1].copy(), r


def construct_packing():
    t0 = time.time()
    budget = 150.0
    rng = np.random.default_rng(12345)
    best_c, best_r, best_s = None, None, -1.0
    k = 0
    while time.time() - t0 < budget:
        x0, y0, r0 = _starts(rng, k)
        k += 1
        v = _solve(x0, y0, r0)
        if v is None:
            continue
        c = np.clip(np.stack([v[:N], v[N:2 * N]], 1), 0, 1)
        r = _lp_radii(c)
        if r is None:
            continue
        s = r.sum()
        if s > best_s:
            best_c, best_r, best_s = c, r, s

    if best_c is None:
        # fallback: simple grid
        g = np.array([[(i + 0.5) / 6, (j + 0.5) / 6] for i in range(5) for j in range(5)] + [[0.95, 0.95]])
        best_c = g
        best_r = np.full(N, 0.02)
    else:
        # final polish from best solution
        v = _solve(best_c[:, 0], best_c[:, 1], best_r, maxiter=500)
        if v is not None:
            c = np.clip(np.stack([v[:N], v[N:2 * N]], 1), 0, 1)
            r = _lp_radii(c)
            if r is not None and r.sum() >= best_s:
                best_c, best_r = c, r

    centers = best_c
    radii = best_r
    for m in (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6):
        rr = np.maximum(radii - m, 0.0)
        if _valid(centers, rr):
            return centers, rr
    return centers, np.maximum(radii * 0.99, 0.0)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
