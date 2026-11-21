import math
import time

import numpy as np
from scipy.optimize import minimize

N = 26
_SEED = 20260
_TIME_BUDGET = 180.0  # seconds of search; the normal run finishes well before this
_N_RANDOM = 80
_N_PERTURB = 80

_PI, _PJ = np.triu_indices(N, 1)
_M = _PI.size

# Variable vector z = [x (N), y (N), r (N)]
_GRAD = np.concatenate([np.zeros(2 * N), -np.ones(N)])


def _build_linear():
    """Boundary constraints: x-r>=0, y-r>=0, 1-x-r>=0, 1-y-r>=0."""
    A = np.zeros((4 * N, 3 * N))
    b = np.zeros(4 * N)
    idx = np.arange(N)
    A[idx, idx] = 1.0
    A[idx, 2 * N + idx] = -1.0
    A[N + idx, N + idx] = 1.0
    A[N + idx, 2 * N + idx] = -1.0
    A[2 * N + idx, idx] = -1.0
    A[2 * N + idx, 2 * N + idx] = -1.0
    b[2 * N + idx] = 1.0
    A[3 * N + idx, N + idx] = -1.0
    A[3 * N + idx, 2 * N + idx] = -1.0
    b[3 * N + idx] = 1.0
    return A, b


_A, _b = _build_linear()


def _pair_terms(z):
    x = z[:N]
    y = z[N:2 * N]
    r = z[2 * N:]
    dx = x[_PI] - x[_PJ]
    dy = y[_PI] - y[_PJ]
    s = r[_PI] + r[_PJ]
    return dx, dy, s


def _pair_fun(z):
    # Non-overlap as squared distance minus squared radius sum (smooth form).
    dx, dy, s = _pair_terms(z)
    return dx * dx + dy * dy - s * s


def _pair_jac(z):
    dx, dy, s = _pair_terms(z)
    J = np.zeros((_M, 3 * N))
    rows = np.arange(_M)
    J[rows, _PI] = 2.0 * dx
    J[rows, _PJ] = -2.0 * dx
    J[rows, N + _PI] = 2.0 * dy
    J[rows, N + _PJ] = -2.0 * dy
    J[rows, 2 * N + _PI] = -2.0 * s
    J[rows, 2 * N + _PJ] = -2.0 * s
    return J


def _optimize(c0, r0):
    """Jointly optimise centres and radii with SLSQP, maximising sum of radii."""
    c0 = np.clip(c0, 0.0, 1.0)
    z0 = np.concatenate([c0[:, 0], c0[:, 1], np.maximum(r0, 0.0)])
    cons = [
        {"type": "ineq", "fun": lambda z: _A @ z + _b, "jac": lambda z: _A},
        {"type": "ineq", "fun": _pair_fun, "jac": _pair_jac},
    ]
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    res = minimize(
        lambda z: -np.sum(z[2 * N:]),
        z0,
        jac=lambda z: _GRAD,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    z = res.x
    centers = np.column_stack([z[:N], z[N:2 * N]])
    return centers, z[2 * N:]


def _dists(c):
    return np.sqrt(((c[_PI] - c[_PJ]) ** 2).sum(axis=1))


def _repair(c, r):
    """Clip radii to the square and proportionally shrink overlapping pairs."""
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    border = np.minimum(
        np.minimum(c[:, 0], c[:, 1]), np.minimum(1.0 - c[:, 0], 1.0 - c[:, 1])
    )
    r = np.minimum(np.maximum(np.asarray(r, dtype=float), 0.0), border)
    d = _dists(c)
    for _ in range(1000):
        s = d / np.maximum(r[_PI] + r[_PJ], 1e-300)
        bad = s < 1.0 - 1e-13
        if not bad.any():
            break
        f = np.ones(N)
        np.minimum.at(f, _PI[bad], s[bad])
        np.minimum.at(f, _PJ[bad], s[bad])
        r = r * f
    return c, r


def _valid(c, r):
    if np.any(r < 0):
        return False
    if np.any(c[:, 0] - r < 0) or np.any(c[:, 1] - r < 0):
        return False
    if np.any(1.0 - c[:, 0] - r < 0) or np.any(1.0 - c[:, 1] - r < 0):
        return False
    return bool(np.all(_dists(c) >= r[_PI] + r[_PJ]))


def _finalize(c, r):
    c, r = _repair(c, r)
    r = r * (1.0 - 1e-9)
    for _ in range(50):
        if _valid(c, r):
            break
        r = r * (1.0 - 1e-7)
    r = np.maximum(r, 0.0)
    return c, r


def _search():
    rng = np.random.default_rng(_SEED)
    start = time.perf_counter()
    best = {"s": -1.0, "c": np.full((N, 2), 0.5), "r": np.zeros(N)}

    def consider(c, r):
        c, r = _repair(c, r)
        s = float(np.sum(r))
        if s > best["s"]:
            best["s"] = s
            best["c"] = c.copy()
            best["r"] = r.copy()

    # Phase 1: random multi-start.
    for _ in range(_N_RANDOM):
        if time.perf_counter() - start > _TIME_BUDGET:
            break
        c0 = rng.uniform(0.05, 0.95, size=(N, 2))
        try:
            c, r = _optimize(c0, np.full(N, 0.01))
            consider(c, r)
        except Exception:
            continue

    # Phase 2: perturb the incumbent and re-optimise (basin hopping).
    for _ in range(_N_PERTURB):
        if time.perf_counter() - start > _TIME_BUDGET:
            break
        sigma = float(rng.choice([0.01, 0.03, 0.06]))
        c0 = np.clip(best["c"] + rng.normal(0.0, sigma, size=(N, 2)), 0.0, 1.0)
        try:
            c, r = _optimize(c0, best["r"])
            consider(c, r)
        except Exception:
            continue

    return best["c"], best["r"]


def run_packing():
    """Pack 26 circles in the unit square; return (centers, radii, sum_radii)."""
    centers, radii = _search()
    centers, radii = _finalize(centers, radii)
    return centers, radii, float(np.sum(radii))
