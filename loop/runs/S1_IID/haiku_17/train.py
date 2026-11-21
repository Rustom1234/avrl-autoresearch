# EVOLVE-BLOCK-START
"""Optimized constructor for 26 circles in the unit square (maximize sum of radii).

Approach:
  1. Variables are all centers and radii jointly: z = [x (26), y (26), r (26)].
  2. Maximize sum(r) subject to r_i <= x_i, r_i <= 1 - x_i (same for y), and
     (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for every pair.
     Solved with SLSQP using analytic Jacobians.
  3. Multi-start from random centers, then basin hopping from the best solution.
  4. Repair pass: clip radii to the walls and shrink overlapping pairs so the
     packing is feasible in double precision, then shrink by a tiny margin.
Fixed seed, so the output is repeatable.
"""

import time

import numpy as np
from scipy.optimize import minimize

N = 26
_IU, _JU = np.triu_indices(N, 1)
_M = _IU.size
_K = np.arange(N)

_N_STARTS = 40
_N_HOPS = 120
_TIME_LIMIT = 200.0  # seconds, safety net well below the 5 minute harness limit
_T0 = time.perf_counter()


def _elapsed():
    return time.perf_counter() - _T0


def _cons(z):
    x = z[:N]
    y = z[N:2 * N]
    r = z[2 * N:]
    dx = x[_IU] - x[_JU]
    dy = y[_IU] - y[_JU]
    s = r[_IU] + r[_JU]
    pair = dx * dx + dy * dy - s * s
    bound = np.concatenate([x - r, y - r, 1.0 - x - r, 1.0 - y - r])
    return np.concatenate([pair, bound])


def _cons_jac(z):
    x = z[:N]
    y = z[N:2 * N]
    r = z[2 * N:]
    dx = x[_IU] - x[_JU]
    dy = y[_IU] - y[_JU]
    s = r[_IU] + r[_JU]
    J = np.zeros((_M + 4 * N, 3 * N))
    p = np.arange(_M)
    J[p, _IU] = 2.0 * dx
    J[p, _JU] = -2.0 * dx
    J[p, N + _IU] = 2.0 * dy
    J[p, N + _JU] = -2.0 * dy
    J[p, 2 * N + _IU] = -2.0 * s
    J[p, 2 * N + _JU] = -2.0 * s
    b = _M
    J[b + _K, _K] = 1.0
    J[b + _K, 2 * N + _K] = -1.0
    J[b + N + _K, N + _K] = 1.0
    J[b + N + _K, 2 * N + _K] = -1.0
    J[b + 2 * N + _K, _K] = -1.0
    J[b + 2 * N + _K, 2 * N + _K] = -1.0
    J[b + 3 * N + _K, N + _K] = -1.0
    J[b + 3 * N + _K, 2 * N + _K] = -1.0
    return J


def _obj(z):
    return -np.sum(z[2 * N:])


def _obj_grad(z):
    g = np.zeros_like(z)
    g[2 * N:] = -1.0
    return g


def _optimize(z0):
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(
            _obj,
            z0,
            jac=_obj_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": _cons, "jac": _cons_jac}],
            options={"maxiter": 400, "ftol": 1e-13},
        )
        return res.x
    except Exception:
        return z0


def _repair(centers, radii):
    """Return an exactly feasible (centers, radii) pair, with a tiny safety margin."""
    x = np.clip(np.asarray(centers[:, 0], dtype=float), 0.0, 1.0)
    y = np.clip(np.asarray(centers[:, 1], dtype=float), 0.0, 1.0)
    r = np.clip(np.asarray(radii, dtype=float), 0.0, None)
    r = np.minimum(r, np.minimum(np.minimum(x, y), np.minimum(1.0 - x, 1.0 - y)))
    dx = x[_IU] - x[_JU]
    dy = y[_IU] - y[_JU]
    d = np.sqrt(dx * dx + dy * dy)
    # Shrinking never increases a pair sum, so one pass fixes every pair.
    for p in range(_M):
        i = _IU[p]
        j = _JU[p]
        tot = r[i] + r[j]
        if tot > d[p]:
            f = d[p] / tot * (1.0 - 1e-12)
            r[i] *= f
            r[j] *= f
    r = r * (1.0 - 1e-10)
    return np.stack([x, y], axis=1), r


def _evaluate(z):
    return _repair(np.stack([z[:N], z[N:2 * N]], axis=1), z[2 * N:])


def _random_start(rng):
    xy = rng.uniform(0.05, 0.95, size=(N, 2))
    return np.concatenate([xy[:, 0], xy[:, 1], np.full(N, 0.05)])


def _search():
    rng = np.random.default_rng(2026)
    best = {"z": None, "c": None, "r": None, "val": -1.0}

    def offer(z):
        c, r = _evaluate(z)
        val = float(np.sum(r))
        if val > best["val"]:
            best.update(z=z.copy(), c=c, r=r, val=val)

    for _ in range(_N_STARTS):
        if best["z"] is not None and _elapsed() > _TIME_LIMIT:
            break
        offer(_optimize(_random_start(rng)))

    for _ in range(_N_HOPS):
        if _elapsed() > _TIME_LIMIT:
            break
        z = best["z"].copy()
        sigma = rng.choice([0.01, 0.03, 0.08])
        z[:N] = np.clip(z[:N] + rng.normal(0.0, sigma, N), 0.0, 1.0)
        z[N:2 * N] = np.clip(z[N:2 * N] + rng.normal(0.0, sigma, N), 0.0, 1.0)
        z[2 * N:] = np.clip(z[2 * N:] * 0.9, 0.0, 0.5)
        offer(_optimize(z))

    return best["c"], best["r"]


def construct_packing():
    """
    Construct an arrangement of 26 circles in the unit square maximizing the sum of radii.

    Returns:
        Tuple of (centers, radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
    """
    centers, radii = _search()
    return centers, radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
