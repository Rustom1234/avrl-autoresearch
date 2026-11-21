# EVOLVE-BLOCK-START
"""Optimised constructor for 26 circles in the unit square (maximise sum of radii)."""

import time

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
_SEED = 12345
_N_STARTS = 150
_TIME_BUDGET = 180.0  # seconds; harness stops a job at 5 minutes

_PI, _PJ = np.triu_indices(N, k=1)
_NP = len(_PI)
_ROWS = np.arange(_NP)
_OBJ_GRAD = np.concatenate([np.zeros(2 * N), -np.ones(N)])


def _make_bnd_jac():
    """Constant Jacobian of the 4N linear boundary constraints."""
    k = np.arange(N)
    b = np.zeros((4 * N, 3 * N))
    b[k, k] = 1.0                    # x - r >= 0
    b[k, 2 * N + k] = -1.0
    b[N + k, k] = -1.0               # 1 - x - r >= 0
    b[N + k, 2 * N + k] = -1.0
    b[2 * N + k, N + k] = 1.0        # y - r >= 0
    b[2 * N + k, 2 * N + k] = -1.0
    b[3 * N + k, N + k] = -1.0       # 1 - y - r >= 0
    b[3 * N + k, 2 * N + k] = -1.0
    return b


_BND_JAC = _make_bnd_jac()


def _cons(z):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    dx = x[_PI] - x[_PJ]
    dy = y[_PI] - y[_PJ]
    s = r[_PI] + r[_PJ]
    pair = dx * dx + dy * dy - s * s
    bnd = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    return np.concatenate([pair, bnd])


def _cons_jac(z):
    x, y = z[:N], z[N:2 * N]
    r = z[2 * N:]
    dx = x[_PI] - x[_PJ]
    dy = y[_PI] - y[_PJ]
    s = r[_PI] + r[_PJ]
    jp = np.zeros((_NP, 3 * N))
    jp[_ROWS, _PI] = 2.0 * dx
    jp[_ROWS, _PJ] = -2.0 * dx
    jp[_ROWS, N + _PI] = 2.0 * dy
    jp[_ROWS, N + _PJ] = -2.0 * dy
    jp[_ROWS, 2 * N + _PI] = -2.0 * s
    jp[_ROWS, 2 * N + _PJ] = -2.0 * s
    return np.vstack([jp, _BND_JAC])


def _optimise(z0):
    """Joint SLSQP over centres and radii. Result is re-checked by the LP below."""
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(
            lambda z: -np.sum(z[2 * N:]),
            z0,
            jac=lambda z: _OBJ_GRAD,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": _cons, "jac": _cons_jac}],
            options={"maxiter": 400, "ftol": 1e-12},
        )
        return res.x
    except Exception:
        return z0


def _radii_for_centers(c):
    """Exact optimal radii for fixed centres: an LP in r with linear constraints
    r_i + r_j <= dist_ij and 0 <= r_i <= distance to the square border."""
    x = np.clip(c[:, 0], 0.0, 1.0)
    y = np.clip(c[:, 1], 0.0, 1.0)
    d = np.hypot(x[_PI] - x[_PJ], y[_PI] - y[_PJ])
    a = np.zeros((_NP, N))
    a[_ROWS, _PI] = 1.0
    a[_ROWS, _PJ] = 1.0
    ub = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    res = linprog(
        -np.ones(N),
        A_ub=a,
        b_ub=d,
        bounds=[(0.0, float(u)) for u in ub],
        method="highs",
    )
    if not res.success:
        return None
    return np.maximum(res.x, 0.0)


def _is_valid(c, r):
    x, y = c[:, 0], c[:, 1]
    if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        return False
    if np.any(r < 0):
        return False
    if np.any(r > np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))):
        return False
    d = np.hypot(x[_PI] - x[_PJ], y[_PI] - y[_PJ])
    return bool(np.all(r[_PI] + r[_PJ] <= d))


def _search():
    rng = np.random.default_rng(_SEED)
    t0 = time.perf_counter()
    best_s, best_c, best_r = -np.inf, None, None
    for k in range(_N_STARTS):
        if k > 0 and time.perf_counter() - t0 > _TIME_BUDGET:
            break
        if best_c is None or k % 3 == 0:
            c0 = rng.uniform(0.05, 0.95, size=(N, 2))
        else:
            c0 = np.clip(best_c + rng.normal(0.0, 0.03, size=(N, 2)), 0.02, 0.98)
        z0 = np.concatenate([c0[:, 0], c0[:, 1], np.full(N, 0.05)])
        z = _optimise(z0)
        c = np.clip(np.column_stack([z[:N], z[N:2 * N]]), 0.0, 1.0)
        r = _radii_for_centers(c)
        if r is None:
            continue
        s = float(np.sum(r))
        if s > best_s:
            best_s, best_c, best_r = s, c, r
    if best_c is None:
        best_c = np.clip(rng.uniform(0.05, 0.95, size=(N, 2)), 0.0, 1.0)
        best_r = _radii_for_centers(best_c)
        if best_r is None:
            best_r = np.zeros(N)
    return best_c, best_r


def construct_packing():
    """
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26)
    """
    centers, radii = _search()
    # Shrink by a tiny margin so the packing is valid in double precision.
    r = radii * (1.0 - 1e-9)
    for _ in range(50):
        if _is_valid(centers, r):
            break
        r = r * (1.0 - 1e-6)
    radii = np.maximum(r, 0.0)
    return centers, radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
