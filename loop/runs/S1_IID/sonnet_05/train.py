# EVOLVE-BLOCK-START
"""Multi-start SLSQP circle packing for n=26 (maximise sum of radii)."""

import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
IU, JU = np.triu_indices(N, 1)
NP = len(IU)
ROWS = np.arange(NP)


def _cons(v):
    x = v[:N]
    y = v[N:2 * N]
    r = v[2 * N:]
    dx = x[IU] - x[JU]
    dy = y[IU] - y[JU]
    d = np.sqrt(dx * dx + dy * dy + 1e-18)
    g1 = d - r[IU] - r[JU]
    return np.concatenate([g1, x - r, 1 - x - r, y - r, 1 - y - r])


def _jac(v):
    x = v[:N]
    y = v[N:2 * N]
    dx = x[IU] - x[JU]
    dy = y[IU] - y[JU]
    d = np.sqrt(dx * dx + dy * dy + 1e-18)
    J = np.zeros((NP + 4 * N, 3 * N))
    J[ROWS, IU] = dx / d
    J[ROWS, JU] = -dx / d
    J[ROWS, N + IU] = dy / d
    J[ROWS, N + JU] = -dy / d
    J[ROWS, 2 * N + IU] = -1
    J[ROWS, 2 * N + JU] = -1
    k = np.arange(N)
    o = NP
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


_OBJ_G = np.concatenate([np.zeros(2 * N), -np.ones(N)])


def _obj(v):
    return -np.sum(v[2 * N:])


def _grad(v):
    return _OBJ_G


def _lp_radii(c):
    """Best radii for fixed centres via LP."""
    d = np.sqrt(((c[IU] - c[JU]) ** 2).sum(1))
    A = np.zeros((NP, N))
    A[ROWS, IU] = 1
    A[ROWS, JU] = 1
    wall = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]),
                      np.minimum(c[:, 1], 1 - c[:, 1]))
    wall = np.maximum(wall, 0.0)
    res = linprog(-np.ones(N), A_ub=A, b_ub=d,
                  bounds=[(0, w) for w in wall], method="highs")
    if res.status != 0:
        return None
    return res.x


def _valid(c, r):
    if np.any(r < 0):
        return False
    if np.any(c[:, 0] - r < 0) or np.any(c[:, 0] + r > 1):
        return False
    if np.any(c[:, 1] - r < 0) or np.any(c[:, 1] + r > 1):
        return False
    d = np.sqrt(((c[IU] - c[JU]) ** 2).sum(1))
    return bool(np.all(r[IU] + r[JU] <= d))


def _finalize(c, r):
    """Shrink radii slightly until strictly valid in double precision."""
    r = np.maximum(r, 0.0)
    for eps in (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
        rr = np.maximum(r - eps, 0.0)
        if _valid(c, rr):
            return rr
    return None


def _start(rng, k):
    if k % 3 == 0:
        c = rng.random((N, 2))
    elif k % 3 == 1:
        # perturbed 5x5 grid plus one extra
        g = (np.arange(5) + 0.5) / 5
        gx, gy = np.meshgrid(g, g)
        c = np.column_stack([gx.ravel(), gy.ravel()])
        c = np.vstack([c, rng.random((1, 2))])
        c = c + rng.normal(0, 0.04, c.shape)
    else:
        # staggered rows 5,5,6,5,5 style
        counts = [5, 5, 6, 5, 5]
        pts = []
        for i, m in enumerate(counts):
            yy = (i + 0.5) / 5
            for j in range(m):
                pts.append([(j + 0.5) / m, yy])
        c = np.array(pts) + rng.normal(0, 0.04, (N, 2))
    c = np.clip(c, 0.03, 0.97)
    r = np.full(N, 0.04) + rng.random(N) * 0.03
    return np.concatenate([c[:, 0], c[:, 1], r])


def construct_packing():
    t0 = time.time()
    budget = 170.0
    rng = np.random.default_rng(12345)
    best = (-1.0, None, None)
    k = 0
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    last = 0.0
    while True:
        el = time.time() - t0
        if el + last * 1.5 > budget and best[1] is not None:
            break
        if el > budget + 20:
            break
        ts = time.time()
        v0 = _start(rng, k)
        k += 1
        try:
            res = minimize(_obj, v0, jac=_grad, method="SLSQP",
                           bounds=bounds,
                           constraints=[{"type": "ineq", "fun": _cons,
                                         "jac": _jac}],
                           options={"maxiter": 300, "ftol": 1e-12})
            v = res.x
            c = np.column_stack([np.clip(v[:N], 0, 1),
                                 np.clip(v[N:2 * N], 0, 1)])
            r = _lp_radii(c)
            if r is not None:
                rr = _finalize(c, r)
                if rr is not None:
                    s = rr.sum()
                    if s > best[0]:
                        best = (s, c.copy(), rr.copy())
        except Exception:
            pass
        last = max(last * 0.5, time.time() - ts)

    if best[1] is None:
        # safe fallback: 5x5 grid + tiny circle
        g = (np.arange(5) + 0.5) / 5
        gx, gy = np.meshgrid(g, g)
        c = np.column_stack([gx.ravel(), gy.ravel()])
        c = np.vstack([c, [0.001, 0.001]])
        r = np.full(N, 0.099)
        r[-1] = 0.0005
        return c, r
    return best[1], best[2]


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
