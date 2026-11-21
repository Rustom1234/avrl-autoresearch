# EVOLVE-BLOCK-START
"""Multistart penalty relaxation + SLSQP polish for packing 26 circles in a unit square.

Note: this file was written without being executed in the authoring session.
"""

import time

import numpy as np
from scipy.optimize import minimize

N = 26
_IU, _JU = np.triu_indices(N, k=1)
_P = _IU.size
_ARN = np.arange(N)
_SEED = 2718281
_TIME_BUDGET = 150.0  # seconds; no new start is begun after this (safety cap)
_MAX_STARTS = 150
_EPS = 1e-12


def _split(z):
    return z[:N], z[N:2 * N], z[2 * N:]


def _penalty(z, mu):
    """Objective -sum(r) + mu * squared overlap/containment violations, with gradient."""
    x, y, r = _split(z)
    dx = x[_IU] - x[_JU]
    dy = y[_IU] - y[_JU]
    d = np.sqrt(dx * dx + dy * dy) + _EPS
    v = np.maximum(r[_IU] + r[_JU] - d, 0.0)
    cp = 2.0 * v
    ux = dx / d
    uy = dy / d
    pen = np.dot(v, v)
    gx = np.bincount(_IU, -cp * ux, N) + np.bincount(_JU, cp * ux, N)
    gy = np.bincount(_IU, -cp * uy, N) + np.bincount(_JU, cp * uy, N)
    gr = np.bincount(_IU, cp, N) + np.bincount(_JU, cp, N)

    v1 = np.maximum(r - x, 0.0)
    v2 = np.maximum(r - 1.0 + x, 0.0)
    v3 = np.maximum(r - y, 0.0)
    v4 = np.maximum(r - 1.0 + y, 0.0)
    pen += np.dot(v1, v1) + np.dot(v2, v2) + np.dot(v3, v3) + np.dot(v4, v4)
    gx = gx + 2.0 * (v2 - v1)
    gy = gy + 2.0 * (v4 - v3)
    gr = gr + 2.0 * (v1 + v2 + v3 + v4)

    f = -np.sum(r) + mu * pen
    g = np.concatenate([mu * gx, mu * gy, mu * gr - 1.0])
    return f, g


def _relax(z):
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    for mu in (10.0, 1e2, 1e3, 1e4, 1e5):
        z = minimize(_penalty, z, args=(mu,), jac=True, method="L-BFGS-B",
                     bounds=bounds, options={"maxiter": 500}).x
    return z


def _cons(z):
    x, y, r = _split(z)
    dx = x[_IU] - x[_JU]
    dy = y[_IU] - y[_JU]
    s = r[_IU] + r[_JU]
    return np.concatenate([dx * dx + dy * dy - s * s, x - r, 1.0 - x - r, y - r, 1.0 - y - r])


def _cons_jac(z):
    x, y, r = _split(z)
    dx = x[_IU] - x[_JU]
    dy = y[_IU] - y[_JU]
    s = r[_IU] + r[_JU]
    J = np.zeros((_P + 4 * N, 3 * N))
    p = np.arange(_P)
    J[p, _IU] = 2.0 * dx
    J[p, _JU] = -2.0 * dx
    J[p, N + _IU] = 2.0 * dy
    J[p, N + _JU] = -2.0 * dy
    J[p, 2 * N + _IU] = -2.0 * s
    J[p, 2 * N + _JU] = -2.0 * s
    q = _P + _ARN
    J[q, _ARN] = 1.0
    J[q, 2 * N + _ARN] = -1.0
    q = _P + N + _ARN
    J[q, _ARN] = -1.0
    J[q, 2 * N + _ARN] = -1.0
    q = _P + 2 * N + _ARN
    J[q, N + _ARN] = 1.0
    J[q, 2 * N + _ARN] = -1.0
    q = _P + 3 * N + _ARN
    J[q, N + _ARN] = -1.0
    J[q, 2 * N + _ARN] = -1.0
    return J


def _obj(z):
    g = np.zeros(3 * N)
    g[2 * N:] = -1.0
    return -np.sum(z[2 * N:]), g


def _polish(z):
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(_obj, z, jac=True, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "ineq", "fun": _cons, "jac": _cons_jac}],
                       options={"maxiter": 400, "ftol": 1e-13})
        if np.all(np.isfinite(res.x)):
            return res.x
    except Exception:
        pass
    return z


def _repair(z):
    """Clip into the square and shrink radii until every pair is strictly separated."""
    x, y, r = _split(z)
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    r = np.clip(r, 0.0, None)
    r = np.minimum(r, np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y)))
    r = np.maximum(r, 0.0)
    for _ in range(1000):
        dx = x[_IU] - x[_JU]
        dy = y[_IU] - y[_JU]
        d = np.sqrt(dx * dx + dy * dy)
        s = r[_IU] + r[_JU]
        bad = np.nonzero(s > d * (1.0 - 1e-9))[0]
        if bad.size == 0:
            break
        for k in bad:
            i, j = _IU[k], _JU[k]
            s_ij = r[i] + r[j]
            if s_ij > 0.0:
                f = d[k] * (1.0 - 2e-9) / s_ij
                r[i] *= f
                r[j] *= f
    return x, y, r


def _search():
    rng = np.random.default_rng(_SEED)
    t0 = time.time()
    best_score = -1.0
    best_pack = None
    best_z = None
    for k in range(_MAX_STARTS):
        if k > 0 and time.time() - t0 > _TIME_BUDGET:
            break
        if best_z is None or k % 2 == 0:
            z0 = np.concatenate([rng.uniform(0.05, 0.95, 2 * N), rng.uniform(0.02, 0.10, N)])
        else:
            sigma = (0.02, 0.05, 0.10)[k % 3]
            z0 = best_z.copy()
            z0[:2 * N] += rng.normal(0.0, sigma, 2 * N)
            z0[2 * N:] *= rng.uniform(0.7, 1.0, N)
        try:
            z = _polish(_relax(z0))
            pack = _repair(z)
        except Exception:
            continue
        score = float(np.sum(pack[2]))
        if score > best_score:
            best_score = score
            best_pack = pack
            best_z = z
    if best_pack is None:
        g = np.linspace(0.05, 0.95, N)
        z = np.concatenate([g, g[::-1], np.full(N, 0.02)])
        best_pack = _repair(z)
    return best_pack


def construct_packing():
    """
    Construct a packing of 26 circles in the unit square maximising the sum of radii.

    Returns:
        Tuple of (centers, radii): centers shape (26, 2), radii shape (26,).
    """
    x, y, r = _search()
    centers = np.stack([x, y], axis=1)
    return centers, r


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
