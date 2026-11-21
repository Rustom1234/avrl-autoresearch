# EVOLVE-BLOCK-START
"""Multi-start SLSQP plus LP radius polish for 26 circles in the unit square.

Maximises the sum of radii. Centers are optimised with SLSQP over (x, y, r) with
analytic Jacobians. For any set of centers, the radii are the exact LP optimum,
then rescaled so the packing is strictly feasible in double precision.
Seeded RNG; the wall-clock guard only limits run time.
Note: this file was written without being executed. Run it once to confirm.
"""

import time

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
SEED = 20240601
TIME_BUDGET = 200.0   # seconds of search wall time (harness limit is 300 s)
N_STARTS = 400        # upper bound on local optimisations

II, JJ = np.triu_indices(N, 1)
P = II.size
A_UB = np.zeros((P, N))
A_UB[np.arange(P), II] = 1.0
A_UB[np.arange(P), JJ] = 1.0
BOUNDS_Z = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
K = np.arange(N)


def _valid(c, r):
    """Exact feasibility check, no tolerance."""
    if np.any(r < 0.0) or np.any(c < 0.0) or np.any(c > 1.0):
        return False
    if np.any(c - r[:, None] < 0.0) or np.any(c + r[:, None] > 1.0):
        return False
    d = np.sqrt(np.sum((c[II] - c[JJ]) ** 2, axis=1))
    return bool(np.all(d >= r[II] + r[JJ]))


def _finalize(c):
    """Given centers, return (centers, radii) with a strictly valid packing, or None."""
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    if not np.all(np.isfinite(c)):
        return None
    b = np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1.0 - c[:, 0], 1.0 - c[:, 1]))
    b = np.maximum(b, 0.0)
    D = np.sqrt(np.sum((c[II] - c[JJ]) ** 2, axis=1))
    try:
        res = linprog(
            -np.ones(N),
            A_ub=A_UB,
            b_ub=D,
            bounds=[(0.0, float(bi)) for bi in b],
            method="highs",
        )
    except Exception:
        return None
    if res.status != 0 or res.x is None:
        return None
    r = np.maximum(res.x, 0.0) * (1.0 - 1e-12)
    shrink = 1e-12
    for _ in range(40):
        if _valid(c, r):
            return c, r
        r = r * (1.0 - shrink)
        shrink = min(shrink * 4.0, 0.5)
    return None


def _cons(z):
    x = z[:N]
    y = z[N:2 * N]
    r = z[2 * N:]
    dx = x[II] - x[JJ]
    dy = y[II] - y[JJ]
    rs = r[II] + r[JJ]
    return np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r, dx * dx + dy * dy - rs * rs])


def _cons_jac(z):
    x = z[:N]
    y = z[N:2 * N]
    r = z[2 * N:]
    dx = x[II] - x[JJ]
    dy = y[II] - y[JJ]
    rs = r[II] + r[JJ]
    J = np.zeros((4 * N + P, 3 * N))
    # boundary rows
    J[K, K] = 1.0
    J[K, 2 * N + K] = -1.0
    J[N + K, K] = -1.0
    J[N + K, 2 * N + K] = -1.0
    J[2 * N + K, N + K] = 1.0
    J[2 * N + K, 2 * N + K] = -1.0
    J[3 * N + K, N + K] = -1.0
    J[3 * N + K, 2 * N + K] = -1.0
    # pair rows
    p = 4 * N + np.arange(P)
    J[p, II] = 2.0 * dx
    J[p, JJ] = -2.0 * dx
    J[p, N + II] = 2.0 * dy
    J[p, N + JJ] = -2.0 * dy
    J[p, 2 * N + II] = -2.0 * rs
    J[p, 2 * N + JJ] = -2.0 * rs
    return J


def _obj(z):
    return -float(np.sum(z[2 * N:]))


def _obj_grad(z):
    g = np.zeros(3 * N)
    g[2 * N:] = -1.0
    return g


def _run_slsqp(z0):
    try:
        res = minimize(
            _obj,
            z0,
            jac=_obj_grad,
            method="SLSQP",
            bounds=BOUNDS_Z,
            constraints=[{"type": "ineq", "fun": _cons, "jac": _cons_jac}],
            options={"maxiter": 400, "ftol": 1e-12},
        )
    except Exception:
        return None
    z = res.x
    if z is None or not np.all(np.isfinite(z)):
        return None
    return z


def _random_start(rng):
    c = rng.uniform(0.05, 0.95, size=(N, 2))
    r = np.full(N, 0.01)
    return np.concatenate([c[:, 0], c[:, 1], r])


def _perturb(best_c, rng):
    sigma = rng.uniform(0.01, 0.10)
    c = best_c + rng.normal(0.0, sigma, size=best_c.shape)
    k = int(rng.integers(0, 4))
    if k:
        idx = rng.choice(N, size=k, replace=False)
        c[idx] = rng.uniform(0.05, 0.95, size=(k, 2))
    return np.clip(c, 0.0, 1.0)


def _search(deadline):
    rng = np.random.default_rng(SEED)
    best = None
    best_s = -1.0
    for it in range(N_STARTS):
        if time.perf_counter() >= deadline:
            break
        z0 = None
        if best is not None and it % 3 != 0:
            fin = _finalize(_perturb(best[0], rng))
            if fin is not None:
                z0 = np.concatenate([fin[0][:, 0], fin[0][:, 1], fin[1]])
        if z0 is None:
            z0 = _random_start(rng)
        z = _run_slsqp(z0)
        if z is None:
            continue
        c = np.clip(np.column_stack([z[:N], z[N:2 * N]]), 0.0, 1.0)
        fin = _finalize(c)
        if fin is None:
            continue
        s = float(np.sum(fin[1]))
        if s > best_s:
            best_s = s
            best = fin
    return best


def _fallback():
    xs = np.linspace(0.05, 0.95, 6)
    ys = np.linspace(0.05, 0.95, 5)
    grid = np.array([(x, y) for x in xs for y in ys])[:N]
    fin = _finalize(grid)
    if fin is None:
        return grid, np.zeros(N)
    return fin


def construct_packing():
    """
    Construct 26 circles in the unit square maximising the sum of radii.

    Returns:
        Tuple of (centers, radii): centers shape (26, 2), radii shape (26,).
    """
    start = time.perf_counter()
    best = None
    try:
        best = _search(start + TIME_BUDGET)
    except Exception:
        best = None
    if best is None:
        best = _fallback()
    return best


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    # Calculate the sum of radii
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
