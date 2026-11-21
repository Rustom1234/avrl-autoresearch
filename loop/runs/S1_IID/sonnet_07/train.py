# EVOLVE-BLOCK-START
"""Multi-start SLSQP circle packing for n=26 (maximize sum of radii)."""

import time
import numpy as np
from scipy.optimize import minimize

N = 26
IU, JU = np.triu_indices(N, 1)
NP = len(IU)


def _unpack(v):
    return v[:N], v[N:2 * N], v[2 * N:]


def _obj(v):
    return -np.sum(v[2 * N:])


def _obj_grad(v):
    g = np.zeros(3 * N)
    g[2 * N:] = -1.0
    return g


def _cons(v):
    x, y, r = _unpack(v)
    dx = x[IU] - x[JU]
    dy = y[IU] - y[JU]
    d = np.sqrt(dx * dx + dy * dy)
    pair = d - r[IU] - r[JU]
    return np.concatenate([pair, x - r, 1 - x - r, y - r, 1 - y - r])


_ar = np.arange(N)
_rows = np.arange(NP)
_BJ = np.zeros((4 * N, 3 * N))
_BJ[_ar, _ar] = 1
_BJ[_ar, 2 * N + _ar] = -1
_BJ[N + _ar, _ar] = -1
_BJ[N + _ar, 2 * N + _ar] = -1
_BJ[2 * N + _ar, N + _ar] = 1
_BJ[2 * N + _ar, 2 * N + _ar] = -1
_BJ[3 * N + _ar, N + _ar] = -1
_BJ[3 * N + _ar, 2 * N + _ar] = -1


def _cons_jac(v):
    x, y, r = _unpack(v)
    dx = x[IU] - x[JU]
    dy = y[IU] - y[JU]
    d = np.sqrt(dx * dx + dy * dy) + 1e-14
    J = np.zeros((NP + 4 * N, 3 * N))
    ux = dx / d
    uy = dy / d
    J[_rows, IU] = ux
    J[_rows, JU] = -ux
    J[_rows, N + IU] = uy
    J[_rows, N + JU] = -uy
    J[_rows, 2 * N + IU] = -1
    J[_rows, 2 * N + JU] = -1
    J[NP:] = _BJ
    return J


def _make_valid(x, y, r):
    """Shrink radii so the packing is valid with zero tolerance."""
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    r = np.maximum(r, 0.0) - 1e-11
    r = np.maximum(r, 0.0)
    r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
    r = np.maximum(r, 0.0)
    for _ in range(200):
        dx = x[IU] - x[JU]
        dy = y[IU] - y[JU]
        d = np.sqrt(dx * dx + dy * dy)
        ex = r[IU] + r[JU] - d
        bad = np.where(ex > -1e-13)[0]
        if len(bad) == 0:
            break
        for k in bad:
            i, j = IU[k], JU[k]
            e = r[i] + r[j] - d[k] + 2e-13
            if e > 0:
                r[i] = max(r[i] - e / 2, 0.0)
                r[j] = max(r[j] - e / 2, 0.0)
    r = r * (1 - 1e-12)
    return r


def _init_points(rng, k):
    mode = k % 4
    if mode == 0:
        x = rng.random(N)
        y = rng.random(N)
    elif mode == 1:
        # perturbed 5x5 grid plus one extra
        g = (np.arange(5) + 0.5) / 5
        gx, gy = np.meshgrid(g, g)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        pts = np.vstack([pts, rng.random((1, 2))])
        pts += rng.normal(0, 0.03, pts.shape)
        x, y = pts[:, 0], pts[:, 1]
    elif mode == 2:
        # hex-like rows with random row counts
        rows = [5, 5, 5, 5, 6] if rng.random() < 0.5 else [4, 5, 4, 5, 4, 4]
        rng.shuffle(rows)
        pts = []
        nr = len(rows)
        for ri, c in enumerate(rows):
            yy = (ri + 0.5) / nr
            for ci in range(c):
                pts.append([(ci + 0.5) / c, yy])
        pts = np.array(pts)
        while len(pts) < N:
            pts = np.vstack([pts, rng.random((1, 2))])
        pts = pts[:N] + rng.normal(0, 0.03, (N, 2))
        x, y = pts[:, 0], pts[:, 1]
    else:
        # large center, small edges
        a = rng.random(N) * 2 * np.pi
        rad = np.sqrt(rng.random(N)) * 0.45
        x = 0.5 + rad * np.cos(a)
        y = 0.5 + rad * np.sin(a)
    x = np.clip(x, 0.03, 0.97)
    y = np.clip(y, 0.03, 0.97)
    r = np.full(N, 0.07) * (0.7 + 0.6 * rng.random(N))
    return np.concatenate([x, y, r])


def _solve(v0, maxiter=300):
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    res = minimize(_obj, v0, jac=_obj_grad, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": _cons, "jac": _cons_jac}],
                   options={"maxiter": maxiter, "ftol": 1e-12})
    return res.x


def construct_packing():
    t0 = time.time()
    budget = 170.0
    rng = np.random.default_rng(12345)
    best_v = None
    best_s = -1.0
    k = 0
    while time.time() - t0 < budget:
        v0 = _init_points(rng, k)
        k += 1
        try:
            v = _solve(v0)
        except Exception:
            continue
        x, y, r = _unpack(v)
        viol = _cons(v).min()
        if viol < -1e-6:
            continue
        s = float(np.sum(r))
        if s > best_s:
            best_s = s
            best_v = v.copy()
        # occasional refinement of the best by perturbation
        if best_v is not None and k % 3 == 0 and time.time() - t0 < budget:
            vp = best_v.copy()
            vp[:2 * N] += rng.normal(0, 0.01, 2 * N)
            vp[:2 * N] = np.clip(vp[:2 * N], 0.01, 0.99)
            vp[2 * N:] *= 0.9
            try:
                v = _solve(vp)
                if _cons(v).min() > -1e-6 and np.sum(v[2 * N:]) > best_s:
                    best_s = float(np.sum(v[2 * N:]))
                    best_v = v.copy()
            except Exception:
                pass
    if best_v is None:
        g = (np.arange(5) + 0.5) / 5
        gx, gy = np.meshgrid(g, g)
        x = np.append(gx.ravel(), 0.5)
        y = np.append(gy.ravel(), 0.5)
        r = np.full(N, 0.02)
    else:
        x, y, r = _unpack(best_v)
        x, y, r = x.copy(), y.copy(), r.copy()
    r = _make_valid(x, y, r)
    centers = np.column_stack([x, y])
    return centers, r


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
