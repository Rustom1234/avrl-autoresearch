# EVOLVE-BLOCK-START
"""Multi-start SLSQP packing of 26 circles in the unit square, maximising the sum of radii.

Variables: centres (x, y) for each circle and one radius per circle (78 in total).
Constraints: pairwise non-overlap (squared form, smooth) and containment in [0, 1]^2.
A seeded random search plus basin-hopping around the incumbent drives SLSQP; the best
result is repaired (pairwise radius scaling) and shrunk by a tiny margin so that it is
valid in double precision.
"""

import time

import numpy as np
from scipy.optimize import minimize

N = 26
SEED = 20240601
N_RANDOM_STARTS = 40
N_HOP_STARTS = 80
# Safety valve only: normally never reached, so results stay reproducible.
TIME_GUARD_S = 200.0
SHRINK = 1e-9

_IU, _JU = np.triu_indices(N, k=1)
_P = len(_IU)
_NVAR = 3 * N

_BOUNDS = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N


def _objective(z):
    return -np.sum(z[2 * N:])


def _objective_grad(z):
    g = np.zeros_like(z)
    g[2 * N:] = -1.0
    return g


def _ineq(z):
    c = z[:2 * N].reshape(N, 2)
    r = z[2 * N:]
    x, y = c[:, 0], c[:, 1]
    dx = c[_IU, 0] - c[_JU, 0]
    dy = c[_IU, 1] - c[_JU, 1]
    rs = r[_IU] + r[_JU]
    pair = dx * dx + dy * dy - rs * rs
    bnd = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    return np.concatenate([pair, bnd])


def _ineq_jac(z):
    c = z[:2 * N].reshape(N, 2)
    r = z[2 * N:]
    dx = c[_IU, 0] - c[_JU, 0]
    dy = c[_IU, 1] - c[_JU, 1]
    rs = r[_IU] + r[_JU]
    J = np.zeros((_P + 4 * N, _NVAR))
    rows = np.arange(_P)
    J[rows, 2 * _IU] = 2.0 * dx
    J[rows, 2 * _JU] = -2.0 * dx
    J[rows, 2 * _IU + 1] = 2.0 * dy
    J[rows, 2 * _JU + 1] = -2.0 * dy
    J[rows, 2 * N + _IU] = -2.0 * rs
    J[rows, 2 * N + _JU] = -2.0 * rs
    idx = np.arange(N)
    b = _P
    J[b + idx, 2 * idx] = 1.0            # x - r >= 0
    J[b + idx, 2 * N + idx] = -1.0
    b = _P + N
    J[b + idx, 2 * idx] = -1.0           # 1 - x - r >= 0
    J[b + idx, 2 * N + idx] = -1.0
    b = _P + 2 * N
    J[b + idx, 2 * idx + 1] = 1.0        # y - r >= 0
    J[b + idx, 2 * N + idx] = -1.0
    b = _P + 3 * N
    J[b + idx, 2 * idx + 1] = -1.0       # 1 - y - r >= 0
    J[b + idx, 2 * N + idx] = -1.0
    return J


def _solve(c0, r0):
    z0 = np.concatenate([np.clip(c0, 0.0, 1.0).reshape(-1), np.maximum(r0, 0.0)])
    try:
        res = minimize(
            _objective,
            z0,
            jac=_objective_grad,
            method="SLSQP",
            bounds=_BOUNDS,
            constraints=[{"type": "ineq", "fun": _ineq, "jac": _ineq_jac}],
            options={"maxiter": 400, "ftol": 1e-12},
        )
        z = res.x
    except Exception:
        z = z0
    c = np.clip(z[:2 * N].reshape(N, 2), 0.0, 1.0)
    return c, z[2 * N:]


def _repair(c, r):
    """Return a strictly valid radius vector for centres c (double precision safe)."""
    c = np.clip(c, 0.0, 1.0)
    bnd = np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1.0 - c[:, 0], 1.0 - c[:, 1]))
    r = np.maximum(np.minimum(np.maximum(r, 0.0), bnd), 0.0)
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1]))
            s = r[i] + r[j]
            if s > d:
                f = d / s if s > 0 else 0.0
                r[i] *= f
                r[j] *= f
    r = np.maximum(r * (1.0 - SHRINK), 0.0)
    return r


def construct_packing():
    """Return (centers, radii) for 26 circles in the unit square."""
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    best_c, best_r, best_s = None, None, -1.0

    def consider(c, r):
        nonlocal best_c, best_r, best_s
        r_ok = _repair(c, r)
        s = float(np.sum(r_ok))
        if s > best_s:
            best_c, best_r, best_s = c.copy(), r_ok, s

    # Phase 1: random starts.
    for _ in range(N_RANDOM_STARTS):
        if time.perf_counter() - t0 > TIME_GUARD_S:
            break
        c0 = rng.uniform(0.05, 0.95, size=(N, 2))
        r0 = np.full(N, 0.05)
        c, r = _solve(c0, r0)
        consider(c, r)

    # Phase 2: basin hopping around the incumbent with shrinking perturbations.
    for k in range(N_HOP_STARTS):
        if best_c is None or time.perf_counter() - t0 > TIME_GUARD_S:
            break
        sigma = 0.12 * (1.0 - k / N_HOP_STARTS) + 0.01
        c0 = best_c + rng.normal(0.0, sigma, size=(N, 2))
        r0 = best_r * 0.9
        c, r = _solve(c0, r0)
        consider(c, r)

    if best_c is None:
        # Fallback: a trivially valid packing (tiny circles on a grid).
        g = np.linspace(0.1, 0.9, 6)
        pts = np.array([(x, y) for x in g for y in g])[:N]
        best_c = pts
        best_r = _repair(pts, np.full(N, 0.1))
    return best_c, best_r


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing optimiser for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
