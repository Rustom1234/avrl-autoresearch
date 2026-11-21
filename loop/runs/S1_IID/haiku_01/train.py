# EVOLVE-BLOCK-START
"""Optimised circle packing for n=26 circles in the unit square (maximise sum of radii)."""

import time

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
I, J = np.triu_indices(N, k=1)
M = len(I)
SEARCH_BUDGET_S = 150.0


def _radii_lp(c):
    """Optimal radii for fixed centres (LP over r)."""
    x, y = c[:, 0], c[:, 1]
    bnd = np.maximum(np.minimum.reduce([x, y, 1 - x, 1 - y]), 0.0)
    d = np.sqrt(((c[I] - c[J]) ** 2).sum(axis=1))
    A = np.zeros((M, N))
    A[np.arange(M), I] = 1.0
    A[np.arange(M), J] = 1.0
    res = linprog(-np.ones(N), A_ub=A, b_ub=d,
                  bounds=list(zip(np.zeros(N), bnd)), method="highs")
    if res.status != 0 or res.x is None:
        return np.zeros(N)
    return res.x


def _make_valid(c, r):
    """Shrink radii until every circle is inside the square and no pair overlaps (exact check)."""
    x, y = c[:, 0], c[:, 1]
    r = np.maximum(np.asarray(r, dtype=float), 0.0)
    for _ in range(10000):
        r = np.maximum(np.minimum(r, np.minimum.reduce([x, y, 1 - x, 1 - y])), 0.0)
        d = np.sqrt(((c[I] - c[J]) ** 2).sum(axis=1))
        s = r[I] + r[J]
        bad = s > d
        if not bad.any():
            break
        ratio = np.where(bad, d / np.maximum(s, 1e-300), 1.0)
        f = np.ones(N)
        np.minimum.at(f, I, ratio)
        np.minimum.at(f, J, ratio)
        r = r * f * (1 - 1e-12)
    return r


def _finalise(c):
    c = np.clip(c, 0.0, 1.0)
    r = _make_valid(c, _radii_lp(c))
    r = r * (1 - 1e-10)
    return r, c


def _slsqp(c0, r0):
    """Jointly optimise centres and radii: maximise sum(r) subject to containment and non-overlap."""
    z0 = np.concatenate([c0[:, 0], c0[:, 1], r0])
    ar = np.arange(N)
    rows = 4 * N + np.arange(M)

    def obj(z):
        return -np.sum(z[2 * N:])

    def obj_jac(z):
        g = np.zeros(3 * N)
        g[2 * N:] = -1.0
        return g

    def cons(z):
        x, y, r = z[:N], z[N:2 * N], z[2 * N:]
        dx = x[I] - x[J]
        dy = y[I] - y[J]
        s = r[I] + r[J]
        return np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r, dx * dx + dy * dy - s * s])

    def cons_jac(z):
        x, y, r = z[:N], z[N:2 * N], z[2 * N:]
        dx = x[I] - x[J]
        dy = y[I] - y[J]
        s = r[I] + r[J]
        Jm = np.zeros((4 * N + M, 3 * N))
        Jm[ar, ar] = 1.0
        Jm[ar, 2 * N + ar] = -1.0
        Jm[N + ar, ar] = -1.0
        Jm[N + ar, 2 * N + ar] = -1.0
        Jm[2 * N + ar, N + ar] = 1.0
        Jm[2 * N + ar, 2 * N + ar] = -1.0
        Jm[3 * N + ar, N + ar] = -1.0
        Jm[3 * N + ar, 2 * N + ar] = -1.0
        Jm[rows, I] = 2 * dx
        Jm[rows, J] = -2 * dx
        Jm[rows, N + I] = 2 * dy
        Jm[rows, N + J] = -2 * dy
        Jm[rows, 2 * N + I] = -2 * s
        Jm[rows, 2 * N + J] = -2 * s
        return Jm

    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    res = minimize(obj, z0, jac=obj_jac, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
                   options={"maxiter": 400, "ftol": 1e-12})
    if not np.all(np.isfinite(res.x)):
        return c0
    return np.column_stack([res.x[:N], res.x[N:2 * N]])


def _search():
    """Multistart SLSQP plus basin hopping from the best packing so far."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(2026)
    best_sum, best_c, best_r = -1.0, None, None
    it = 0
    while it == 0 or time.perf_counter() - t0 < SEARCH_BUDGET_S:
        if best_c is None or rng.random() < 0.35:
            c0 = rng.uniform(0.05, 0.95, size=(N, 2))
            r0 = np.full(N, 0.01)
        else:
            sigma = rng.choice([0.02, 0.05, 0.1])
            c0 = np.clip(best_c + rng.normal(0.0, sigma, size=(N, 2)), 0.0, 1.0)
            r0 = best_r.copy()
        it += 1
        try:
            c = _slsqp(c0, r0)
        except Exception:
            c = c0
        r, c = _finalise(c)
        s = float(np.sum(r))
        if s > best_sum:
            best_sum, best_c, best_r = s, c, r
    return best_c, best_r


# EVOLVE-BLOCK-END


def run_packing():
    """Run the circle packing optimiser for n=26. Returns (centers, radii, sum_radii)."""
    centers, radii = _search()
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
