# EVOLVE-BLOCK-START
"""Multi-start SLSQP packing of 26 circles in the unit square, maximising the sum of radii.

Pipeline: seeded structured and random starts, then SLSQP on (x, y, r) with exact
containment and non-overlap constraints, then basin-hopping perturbations. Every
candidate is repaired in double precision and shrunk by a tiny factor, so the
returned packing is valid.
"""

import time

import numpy as np
from scipy.optimize import minimize

N = 26
_I, _J = np.triu_indices(N, k=1)
_NP = _I.size
_K = np.arange(N)
_TIME_GUARD = 200.0  # seconds; safety net only, well above the expected run time
_N_STARTS_RANDOM = 30
_N_HOPS = 150


def _constraint_fun(z):
    x = z[:N]
    y = z[N:2 * N]
    r = z[2 * N:]
    dx = x[_I] - x[_J]
    dy = y[_I] - y[_J]
    rs = r[_I] + r[_J]
    pair = dx * dx + dy * dy - rs * rs
    bound = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    return np.concatenate([pair, bound])


def _constraint_jac(z):
    x = z[:N]
    y = z[N:2 * N]
    r = z[2 * N:]
    dx = x[_I] - x[_J]
    dy = y[_I] - y[_J]
    rs = r[_I] + r[_J]
    jac = np.zeros((_NP + 4 * N, 3 * N))
    p = np.arange(_NP)
    jac[p, _I] = 2.0 * dx
    jac[p, _J] = -2.0 * dx
    jac[p, N + _I] = 2.0 * dy
    jac[p, N + _J] = -2.0 * dy
    jac[p, 2 * N + _I] = -2.0 * rs
    jac[p, 2 * N + _J] = -2.0 * rs
    b = _NP
    jac[b + _K, _K] = 1.0
    jac[b + _K, 2 * N + _K] = -1.0
    jac[b + N + _K, _K] = -1.0
    jac[b + N + _K, 2 * N + _K] = -1.0
    jac[b + 2 * N + _K, N + _K] = 1.0
    jac[b + 2 * N + _K, 2 * N + _K] = -1.0
    jac[b + 3 * N + _K, N + _K] = -1.0
    jac[b + 3 * N + _K, 2 * N + _K] = -1.0
    return jac


def _objective(z):
    return -np.sum(z[2 * N:])


def _objective_grad(z):
    g = np.zeros_like(z)
    g[2 * N:] = -1.0
    return g


def _slsqp(c, r):
    z0 = np.concatenate([c[:, 0], c[:, 1], r])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    try:
        res = minimize(
            _objective,
            z0,
            jac=_objective_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": _constraint_fun, "jac": _constraint_jac}],
            options={"maxiter": 300, "ftol": 1e-12},
        )
        z = res.x
    except Exception:
        z = z0
    if not np.all(np.isfinite(z)):
        z = z0
    c2 = np.stack([z[:N], z[N:2 * N]], axis=1)
    return c2, z[2 * N:]


def _repair(c, r):
    """Return a provably valid packing: clip into the square, fix overlaps, shrink slightly."""
    c = np.clip(c, 0.0, 1.0)
    r = np.clip(r, 0.0, 0.5)
    wall = np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1.0 - c[:, 0], 1.0 - c[:, 1]))
    r = np.minimum(r, wall).copy()
    for _ in range(1000):
        changed = False
        for i, j in zip(_I, _J):
            d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
            s = r[i] + r[j]
            if s > d:
                f = d / s * (1.0 - 1e-12) if s > 0.0 else 0.0
                r[i] *= f
                r[j] *= f
                changed = True
        if not changed:
            break
    return c, r * (1.0 - 1e-9)


def _solve(c, r):
    c = np.clip(c, 0.0, 1.0)
    r = np.clip(r, 0.0, 0.5)
    c2, r2 = _slsqp(c, r)
    c2, r2 = _repair(c2, r2)
    return c2, r2, float(np.sum(r2))


def _structured_starts():
    # Hexagonal-like rows with 6, 5, 6, 5, 4 circles (sums to 26).
    pts = []
    for y, m in zip(np.linspace(0.1, 0.9, 5), [6, 5, 6, 5, 4]):
        off = 0.0 if m == 6 else 0.1
        for x in np.linspace(0.08 + off, 0.92 - off, m):
            pts.append((x, y))
    hexc = np.array(pts)
    # 5x5 grid plus one centre circle.
    g = np.linspace(0.1, 0.9, 5)
    gridc = np.vstack([np.array([(a, b) for a in g for b in g]), [[0.5, 0.5]]])
    return [hexc, gridc]


def construct_packing():
    t0 = time.perf_counter()
    rng = np.random.default_rng(2026)

    starts = [(c, np.full(N, 0.03)) for c in _structured_starts()]
    for _ in range(_N_STARTS_RANDOM):
        starts.append((rng.uniform(0.05, 0.95, size=(N, 2)), np.full(N, 0.03)))

    best = None
    for c, r in starts:
        if best is not None and time.perf_counter() - t0 > _TIME_GUARD:
            break
        res = _solve(c, r)
        if best is None or res[2] > best[2]:
            best = res

    # Basin hopping around the incumbent.
    sigmas = [0.005, 0.02, 0.05]
    for it in range(_N_HOPS):
        if time.perf_counter() - t0 > _TIME_GUARD:
            break
        sigma = sigmas[it % len(sigmas)]
        c = best[0] + rng.normal(0.0, sigma, size=(N, 2))
        if rng.random() < 0.3:
            k = rng.integers(N)
            c[k] = rng.uniform(0.05, 0.95, size=2)
        r = best[1].copy()
        res = _solve(c, r)
        if res[2] > best[2] + 1e-12:
            best = res

    return best[0], best[1]


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
