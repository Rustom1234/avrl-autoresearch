# EVOLVE-BLOCK-START
"""Circle packing for n=26: multi-start SLSQP on (x, y, r), LP polish, safe shrink."""

import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
IU = np.triu_indices(N, 1)
I_IDX, J_IDX = IU
M = len(I_IDX)


def _cons(v):
    x = v[:N]
    y = v[N:2 * N]
    r = v[2 * N:]
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    s = r[I_IDX] + r[J_IDX]
    return dx * dx + dy * dy - s * s


def _cons_jac(v):
    x = v[:N]
    y = v[N:2 * N]
    r = v[2 * N:]
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    s = r[I_IDX] + r[J_IDX]
    J = np.zeros((M, 3 * N))
    k = np.arange(M)
    J[k, I_IDX] = 2 * dx
    J[k, J_IDX] = -2 * dx
    J[k, N + I_IDX] = 2 * dy
    J[k, N + J_IDX] = -2 * dy
    J[k, 2 * N + I_IDX] = -2 * s
    J[k, 2 * N + J_IDX] = -2 * s
    return J


_OBJ_G = np.concatenate([np.zeros(2 * N), -np.ones(N)])


def _obj(v):
    return -np.sum(v[2 * N:])


def _obj_g(v):
    return _OBJ_G


def _lp_radii(centers):
    """Optimal radii for fixed centers via LP."""
    x = centers[:, 0]
    y = centers[:, 1]
    ub = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    ub = np.maximum(ub, 0.0)
    d = np.sqrt((x[I_IDX] - x[J_IDX]) ** 2 + (y[I_IDX] - y[J_IDX]) ** 2)
    A = np.zeros((M, N))
    k = np.arange(M)
    A[k, I_IDX] = 1
    A[k, J_IDX] = 1
    try:
        res = linprog(-np.ones(N), A_ub=A, b_ub=d,
                      bounds=[(0, u) for u in ub], method="highs")
        if res.status == 0:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None


def _valid(centers, radii):
    if np.any(radii < 0):
        return False
    x = centers[:, 0]
    y = centers[:, 1]
    if np.any(x - radii < 0) or np.any(x + radii > 1):
        return False
    if np.any(y - radii < 0) or np.any(y + radii > 1):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if d < radii[i] + radii[j]:
                return False
            d2 = np.linalg.norm(centers[i] - centers[j])
            if d2 < radii[i] + radii[j]:
                return False
    return True


def _finalize(centers):
    centers = np.clip(centers, 0.0, 1.0)
    r = _lp_radii(centers)
    if r is None:
        return None, None
    r = r * (1 - 1e-11) - 1e-13
    r = np.maximum(r, 0.0)
    for _ in range(60):
        if _valid(centers, r):
            return centers, r
        r = r * (1 - 1e-9) - 1e-12
        r = np.maximum(r, 0.0)
    return None, None


def _solve(x0, maxiter=300):
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    cons = [{"type": "ineq", "fun": _cons, "jac": _cons_jac}]
    res = minimize(_obj, x0, jac=_obj_g, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-12})
    return res.x


def _init_hex(rng):
    # staggered rows with random row counts summing to 26
    patterns = [
        [5, 5, 6, 5, 5],
        [6, 5, 5, 5, 5],
        [5, 6, 5, 6, 4],
        [4, 5, 4, 5, 4, 4],
        [5, 4, 5, 4, 5, 3],
        [6, 5, 4, 5, 6],
    ]
    p = patterns[rng.integers(len(patterns))]
    pts = []
    nr = len(p)
    for ri, c in enumerate(p):
        yy = (ri + 0.5) / nr
        off = 0.5 / c if ri % 2 else 0.0
        for k in range(c):
            xx = (k + 0.5) / c
            pts.append([xx, yy])
    pts = np.array(pts)[:N]
    while len(pts) < N:
        pts = np.vstack([pts, rng.random((1, 2))])
    pts = pts + rng.normal(0, 0.03, pts.shape)
    return np.clip(pts, 0.03, 0.97)


def construct_packing():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    budget = 190.0
    best_sum = -1.0
    best = None
    it = 0
    while time.time() - t0 < budget:
        mode = it % 3
        if mode == 0:
            c = rng.random((N, 2)) * 0.9 + 0.05
        elif mode == 1:
            c = _init_hex(rng)
        else:
            g = np.stack(np.meshgrid(np.arange(5), np.arange(5)), -1).reshape(-1, 2)
            c = (g + 0.5) / 5
            c = np.vstack([c, rng.random((1, 2))])
            c = c + rng.normal(0, 0.04, c.shape)
            c = np.clip(c, 0.03, 0.97)
        it += 1
        r0 = np.full(N, 0.05) + rng.random(N) * 0.03
        v0 = np.concatenate([c[:, 0], c[:, 1], r0])
        try:
            v = _solve(v0)
        except Exception:
            continue
        cen = np.stack([v[:N], v[N:2 * N]], 1)
        cc, rr = _finalize(cen)
        if cc is None:
            continue
        s = float(np.sum(rr))
        if s > best_sum:
            best_sum = s
            best = (cc, rr)
    if best is None:
        # fallback: simple 5x5 grid plus one tiny circle, always valid
        g = np.stack(np.meshgrid(np.arange(5), np.arange(5)), -1).reshape(-1, 2)
        c = np.vstack([(g + 0.5) / 5, [[0.999, 0.999]]])
        r = np.full(N, 0.099)
        r[-1] = 1e-4
        return c, r
    return best


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
