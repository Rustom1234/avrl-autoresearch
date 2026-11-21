# EVOLVE-BLOCK-START
"""Multi-start SLSQP optimisation of 26 circles in the unit square (maximise sum of radii)."""

import time

import numpy as np
from scipy.optimize import minimize

N = 26
SEED = 20240601
TIME_BUDGET = 180.0  # seconds of search wall time; harness limit is 300 s
SHRINK = 1e-10  # absolute radius margin for the final double-precision check

_IU, _JU = np.triu_indices(N, k=1)
_P = len(_IU)
_GOBJ = np.zeros(3 * N)
_GOBJ[2 * N:] = -1.0


def _constraints(z):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    dx = x[_IU] - x[_JU]
    dy = y[_IU] - y[_JU]
    rs = r[_IU] + r[_JU]
    pair = dx * dx + dy * dy - rs * rs
    border = np.concatenate([r - x, r - (1 - x), r - y, r - (1 - y)])
    return np.concatenate([pair, border])


def _jacobian(z):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    dx = x[_IU] - x[_JU]
    dy = y[_IU] - y[_JU]
    rs = r[_IU] + r[_JU]
    J = np.zeros((_P + 4 * N, 3 * N))
    rows = np.arange(_P)
    J[rows, _IU] = 2 * dx
    J[rows, _JU] = -2 * dx
    J[rows, N + _IU] = 2 * dy
    J[rows, N + _JU] = -2 * dy
    J[rows, 2 * N + _IU] = -2 * rs
    J[rows, 2 * N + _JU] = -2 * rs
    k = np.arange(N)
    b = _P
    J[b + k, k] = -1.0
    J[b + k, 2 * N + k] = 1.0
    J[b + N + k, k] = 1.0
    J[b + N + k, 2 * N + k] = 1.0
    J[b + 2 * N + k, N + k] = -1.0
    J[b + 2 * N + k, 2 * N + k] = 1.0
    J[b + 3 * N + k, N + k] = 1.0
    J[b + 3 * N + k, 2 * N + k] = 1.0
    return J


def _repair(centers, radii):
    """Return a strictly feasible (centers, radii) pair: circles inside the square, no overlaps."""
    c = np.clip(np.asarray(centers, dtype=float), 0.0, 1.0)
    x, y = c[:, 0].copy(), c[:, 1].copy()
    r = np.maximum(np.asarray(radii, dtype=float), 0.0)
    r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
    r = np.maximum(r, 0.0)
    for _ in range(500):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                s = r[i] + r[j]
                if s > d:
                    f = d / s * (1 - 1e-12) if s > 0 else 0.0
                    r[i] *= f
                    r[j] *= f
                    changed = True
        if not changed:
            break
    r = np.maximum(r - SHRINK, 0.0)
    return c, r


def _solve(z0):
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(
            lambda z: -np.sum(z[2 * N:]),
            z0,
            jac=lambda z: _GOBJ,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": _constraints, "jac": _jacobian}],
            options={"maxiter": 500, "ftol": 1e-13},
        )
        z = res.x
    except Exception:
        z = z0
    return z


def _score(centers, radii):
    c, r = _repair(centers, radii)
    return c, r, float(np.sum(r))


def construct_packing():
    """
    Search for a packing of 26 circles in the unit square maximising the sum of radii.

    Returns:
        Tuple of (centers, radii): centers shape (26, 2), radii shape (26,), feasible.
    """
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    best = None  # (sum, centers, radii)

    it = 0
    while it == 0 or time.perf_counter() - t0 < TIME_BUDGET:
        if best is None or it < 6 or rng.random() < 0.4:
            # random restart
            centers = rng.uniform(0.05, 0.95, size=(N, 2))
        else:
            # basin hopping: perturb the incumbent
            sigma = rng.choice([0.01, 0.03, 0.06, 0.1])
            centers = best[1] + rng.normal(0.0, sigma, size=(N, 2))
            centers = np.clip(centers, 0.01, 0.99)
        _, r0 = _repair(centers, np.full(N, 0.05))
        z0 = np.concatenate([centers[:, 0], centers[:, 1], r0])
        z = _solve(z0)
        c, r, s = _score(np.column_stack([z[:N], z[N:2 * N]]), z[2 * N:])
        if best is None or s > best[0]:
            best = (s, c, r)
        it += 1

    return best[1], best[2]


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing search for n=26"""
    centers, radii = construct_packing()
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
