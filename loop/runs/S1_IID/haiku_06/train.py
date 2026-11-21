# EVOLVE-BLOCK-START
"""Optimised circle packing for n=26 circles in the unit square.

Centres and radii are optimised jointly (maximise the sum of radii) with SLSQP
on smooth squared-distance constraints and analytic Jacobians. The search is a
seeded multi-start followed by basin hopping from the best packing found. Every
candidate is passed through an exact repair step so the returned packing is
strictly valid in double precision.
"""

import time

import numpy as np
from scipy.optimize import minimize

N = 26
SEED = 20260
N_STARTS = 30
N_HOPS = 200
TIME_BUDGET = 200.0  # safety backstop only; the harness kills jobs at 300 s
SHRINK = 1e-9

_PI, _PJ = np.triu_indices(N, k=1)
_NPAIR = _PI.size
_K = np.arange(N)
_BOUNDS = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N


def _obj(z):
    return -float(np.sum(z[2 * N:]))


def _obj_grad(z):
    g = np.zeros(3 * N)
    g[2 * N:] = -1.0
    return g


def _constraints(z):
    x = z[:N]
    y = z[N:2 * N]
    r = z[2 * N:]
    dx = x[_PI] - x[_PJ]
    dy = y[_PI] - y[_PJ]
    rs = r[_PI] + r[_PJ]
    pair = dx * dx + dy * dy - rs * rs
    bound = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    return np.concatenate([pair, bound])


def _constraints_jac(z):
    x = z[:N]
    y = z[N:2 * N]
    r = z[2 * N:]
    dx = x[_PI] - x[_PJ]
    dy = y[_PI] - y[_PJ]
    rs = r[_PI] + r[_PJ]
    m = _NPAIR
    rows = np.arange(m)
    J = np.zeros((m + 4 * N, 3 * N))
    J[rows, _PI] = 2.0 * dx
    J[rows, _PJ] = -2.0 * dx
    J[rows, N + _PI] = 2.0 * dy
    J[rows, N + _PJ] = -2.0 * dy
    J[rows, 2 * N + _PI] = -2.0 * rs
    J[rows, 2 * N + _PJ] = -2.0 * rs
    # Boundary blocks, in the same order as _constraints: x-r, 1-x-r, y-r, 1-y-r.
    for blk, (off, sign) in enumerate([(0, 1.0), (0, -1.0), (N, 1.0), (N, -1.0)]):
        brows = m + blk * N + _K
        J[brows, off + _K] = sign
        J[brows, 2 * N + _K] = -1.0
    return J


def _solve(z0):
    res = minimize(
        _obj,
        z0,
        jac=_obj_grad,
        method="SLSQP",
        bounds=_BOUNDS,
        constraints=[{"type": "ineq", "fun": _constraints, "jac": _constraints_jac}],
        options={"maxiter": 400, "ftol": 1e-13},
    )
    return res.x


def _repair(z):
    """Return (centers, radii) that are strictly valid, from any candidate z."""
    c = np.clip(np.stack([z[:N], z[N:2 * N]], axis=1), 0.0, 1.0)
    r = np.clip(z[2 * N:], 0.0, None).astype(float)
    x = c[:, 0]
    y = c[:, 1]
    r = np.minimum(r, np.minimum(np.minimum(x, y), np.minimum(1.0 - x, 1.0 - y)))
    r = np.maximum(r, 0.0)
    # Scaling a pair down never breaks a constraint already satisfied, so a few
    # passes are enough to absorb floating-point slop.
    for _ in range(3):
        for i in range(N):
            for j in range(i + 1, N):
                d = float(np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1]))
                s = r[i] + r[j]
                if s > d:
                    f = d / s if s > 0.0 else 0.0
                    r[i] *= f
                    r[j] *= f
    r = r * (1.0 - SHRINK)
    return c, r


def _evaluate(z):
    if not np.all(np.isfinite(z)):
        return -1.0, None, None
    c, r = _repair(z)
    return float(np.sum(r)), c, r


def _pack_z(c, r):
    return np.concatenate([c[:, 0], c[:, 1], r])


def construct_packing():
    """
    Construct 26 circles in the unit square with a large sum of radii.

    Returns:
        Tuple of (centers, radii): centers shape (26, 2), radii shape (26,).
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)

    best_val = -1.0
    best_c = None
    best_r = None

    # Phase 1: seeded multi-start from random interior centres.
    for _ in range(N_STARTS):
        if time.perf_counter() - t0 > TIME_BUDGET:
            break
        c0 = rng.uniform(0.1, 0.9, size=(N, 2))
        z0 = _pack_z(c0, np.full(N, 0.05))
        val, c, r = _evaluate(_solve(z0))
        if c is not None and val > best_val:
            best_val, best_c, best_r = val, c, r

    if best_c is None:
        # Fallback: a valid but poor packing, so the harness never sees a crash.
        best_c = np.full((N, 2), 0.5)
        best_r = np.zeros(N)
        best_val = 0.0

    # Phase 2: basin hopping from the best packing found so far.
    for _ in range(N_HOPS):
        if time.perf_counter() - t0 > TIME_BUDGET:
            break
        x = best_c[:, 0].copy()
        y = best_c[:, 1].copy()
        r = best_r.copy()
        sigma = rng.choice([0.01, 0.03, 0.06])
        x = x + rng.normal(0.0, sigma, N)
        y = y + rng.normal(0.0, sigma, N)
        if rng.random() < 0.5:
            idx = int(rng.integers(N))
            x[idx] = rng.uniform(0.05, 0.95)
            y[idx] = rng.uniform(0.05, 0.95)
            r[idx] = 0.02
        z0 = np.concatenate([np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0), r])
        val, c, rr = _evaluate(_solve(z0))
        if c is not None and val > best_val + 1e-12:
            best_val, best_c, best_r = val, c, rr

    # Final polish of the incumbent (cheap, can only help).
    if best_r.sum() > 0.0:
        val, c, rr = _evaluate(_solve(_pack_z(best_c, best_r)))
        if c is not None and val > best_val:
            best_val, best_c, best_r = val, c, rr

    return best_c, best_r


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
