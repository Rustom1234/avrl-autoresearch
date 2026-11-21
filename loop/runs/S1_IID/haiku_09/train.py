# EVOLVE-BLOCK-START
"""Multistart + basin-hopping SLSQP for packing 26 circles in the unit square.

Maximises the sum of radii. Centres and radii are optimised jointly with containment
and pairwise non-overlap constraints. Every candidate is repaired by an LP over the
radii for its centres, then shrunk by a tiny margin, so the returned packing is valid.
"""

import time

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
SEED = 20240601
TIME_BUDGET = 170.0  # seconds of search; the harness kills jobs at 300 s
MARGIN = 1e-9

_I, _J = np.triu_indices(N, 1)
_P = len(_I)
_ROWS = np.arange(_P)
_IDX = np.arange(N)


def _repair(centers):
    """Return (centers, radii) where radii are strictly feasible for these centers."""
    c = np.clip(np.asarray(centers, dtype=float), 0.0, 1.0)
    x, y = c[:, 0], c[:, 1]
    border = np.maximum(np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y)), 0.0)
    d = np.sqrt(((c[_I] - c[_J]) ** 2).sum(axis=1))

    # LP: maximise sum r subject to r_i + r_j <= d_ij and 0 <= r_i <= border_i.
    A = np.zeros((_P, N))
    A[_ROWS, _I] = 1.0
    A[_ROWS, _J] = 1.0
    r = np.zeros(N)
    try:
        res = linprog(-np.ones(N), A_ub=A, b_ub=d,
                      bounds=list(zip(np.zeros(N), border)), method="highs")
        if res.status == 0:
            r = np.asarray(res.x, dtype=float)
    except Exception:
        pass

    # Remove solver tolerance violations: scale each circle by the smallest
    # factor demanded by its violated pairs. One pass is enough, since every
    # violated pair ends up with sum <= d_ij.
    r = np.clip(r, 0.0, border)
    s = r[_I] + r[_J]
    f = np.ones(_P)
    bad = s > d
    f[bad] = d[bad] / s[bad]
    fac = np.ones(N)
    np.minimum.at(fac, _I, f)
    np.minimum.at(fac, _J, f)
    r = r * fac * (1.0 - MARGIN)
    return c, np.maximum(r, 0.0)


def _slsqp(z0):
    """Locally maximise the sum of radii from z0 = [x (N), y (N), r (N)]."""

    def cons(z):
        x, y, r = z[:N], z[N:2 * N], z[2 * N:]
        dx = x[_I] - x[_J]
        dy = y[_I] - y[_J]
        s = r[_I] + r[_J]
        pair = dx * dx + dy * dy - s * s
        border = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
        return np.concatenate([pair, border])

    def cons_jac(z):
        x, y, r = z[:N], z[N:2 * N], z[2 * N:]
        dx = x[_I] - x[_J]
        dy = y[_I] - y[_J]
        s = r[_I] + r[_J]
        Jm = np.zeros((_P + 4 * N, 3 * N))
        Jm[_ROWS, _I] = 2.0 * dx
        Jm[_ROWS, _J] = -2.0 * dx
        Jm[_ROWS, N + _I] = 2.0 * dy
        Jm[_ROWS, N + _J] = -2.0 * dy
        Jm[_ROWS, 2 * N + _I] = -2.0 * s
        Jm[_ROWS, 2 * N + _J] = -2.0 * s
        k = _P
        Jm[k + _IDX, _IDX] = 1.0
        Jm[k + _IDX, 2 * N + _IDX] = -1.0
        k += N
        Jm[k + _IDX, _IDX] = -1.0
        Jm[k + _IDX, 2 * N + _IDX] = -1.0
        k += N
        Jm[k + _IDX, N + _IDX] = 1.0
        Jm[k + _IDX, 2 * N + _IDX] = -1.0
        k += N
        Jm[k + _IDX, N + _IDX] = -1.0
        Jm[k + _IDX, 2 * N + _IDX] = -1.0
        return Jm

    def obj(z):
        return -np.sum(z[2 * N:])

    def obj_jac(z):
        g = np.zeros(3 * N)
        g[2 * N:] = -1.0
        return g

    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    res = minimize(obj, z0, jac=obj_jac, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
                   options={"maxiter": 300, "ftol": 1e-12})
    return res.x


def run_packing():
    """Run the search and return (centers, radii, sum_radii) for n=26."""
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()

    best_sum, best_c, best_r, best_z = -1.0, None, None, None

    def elapsed():
        return time.perf_counter() - t0

    def evaluate(z):
        nonlocal best_sum, best_c, best_r, best_z
        c, r = _repair(np.column_stack([z[:N], z[N:2 * N]]))
        s = float(r.sum())
        if s > best_sum:
            best_sum, best_c, best_r, best_z = s, c, r, z
        return s

    def start_from(c0):
        _, r0 = _repair(c0)
        z0 = np.concatenate([c0[:, 0], c0[:, 1], r0])
        try:
            return _slsqp(z0)
        except Exception:
            return None

    # Deterministic fallback so that a valid incumbent always exists.
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 6), np.linspace(0.1, 0.9, 5))
    grid = np.column_stack([gx.ravel(), gy.ravel()])[:N]
    if len(grid) < N:
        grid = np.vstack([grid, np.full((N - len(grid), 2), 0.5)])
    evaluate(np.concatenate([grid[:, 0], grid[:, 1], np.zeros(N)]))

    # Phase 1: random multistart.
    while elapsed() < 0.5 * TIME_BUDGET:
        z = start_from(rng.uniform(0.05, 0.95, size=(N, 2)))
        if z is not None:
            evaluate(z)

    # Phase 2: basin hopping around the incumbent (greedy acceptance).
    while elapsed() < TIME_BUDGET:
        c = np.column_stack([best_z[:N], best_z[N:2 * N]])
        if rng.random() < 0.1:
            c = rng.uniform(0.05, 0.95, size=(N, 2))
        elif rng.random() < 0.5:
            c = c + rng.normal(0.0, rng.uniform(0.01, 0.08), size=c.shape)
        else:
            k = int(rng.integers(1, 4))
            idx = rng.choice(N, size=k, replace=False)
            c[idx] = rng.uniform(0.05, 0.95, size=(k, 2))
        c = np.clip(c, 0.01, 0.99)
        z = start_from(c)
        if z is not None:
            evaluate(z)

    centers = np.asarray(best_c, dtype=float)
    radii = np.asarray(best_r, dtype=float)
    return centers, radii, float(np.sum(radii))


# EVOLVE-BLOCK-END
