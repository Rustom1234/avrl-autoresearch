"""Multi-start SLSQP optimisation of 26 circles in the unit square (maximise sum of radii).

Decision variables: 26 centres (x, y) and 26 radii. Constraints: every circle inside the
square and every pair non-overlapping (squared-distance form, smooth for SLSQP). A repair
step then shrinks radii so the packing is valid in double precision with zero tolerance.
"""

import time

import numpy as np
from scipy.optimize import minimize

N = 26
TIME_BUDGET = 150.0  # seconds of search before returning (harness limit is 5 min)
HARD_LIMIT = 200.0   # absolute cutoff, only used if no valid packing was found yet
SEED = 2024

_PI, _PJ = np.triu_indices(N, 1)
_NP = _PI.size
_IDX = np.arange(N)


def _unpack(z):
    return z[: 2 * N].reshape(N, 2), z[2 * N:]


def _objective(z):
    return -np.sum(z[2 * N:])


def _objective_grad(z):
    g = np.zeros_like(z)
    g[2 * N:] = -1.0
    return g


def _constraints(z):
    c, r = _unpack(z)
    dx = c[_PI, 0] - c[_PJ, 0]
    dy = c[_PI, 1] - c[_PJ, 1]
    s = r[_PI] + r[_PJ]
    pair = dx * dx + dy * dy - s * s
    x, y = c[:, 0], c[:, 1]
    bnd = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    return np.concatenate([pair, bnd])


def _constraints_jac(z):
    c, r = _unpack(z)
    dx = c[_PI, 0] - c[_PJ, 0]
    dy = c[_PI, 1] - c[_PJ, 1]
    s = r[_PI] + r[_PJ]
    J = np.zeros((_NP + 4 * N, 3 * N))
    p = np.arange(_NP)
    J[p, 2 * _PI] = 2.0 * dx
    J[p, 2 * _PJ] = -2.0 * dx
    J[p, 2 * _PI + 1] = 2.0 * dy
    J[p, 2 * _PJ + 1] = -2.0 * dy
    J[p, 2 * N + _PI] = -2.0 * s
    J[p, 2 * N + _PJ] = -2.0 * s
    k = _NP
    J[k + _IDX, 2 * _IDX] = 1.0
    J[k + _IDX, 2 * N + _IDX] = -1.0
    k += N
    J[k + _IDX, 2 * _IDX] = -1.0
    J[k + _IDX, 2 * N + _IDX] = -1.0
    k += N
    J[k + _IDX, 2 * _IDX + 1] = 1.0
    J[k + _IDX, 2 * N + _IDX] = -1.0
    k += N
    J[k + _IDX, 2 * _IDX + 1] = -1.0
    J[k + _IDX, 2 * N + _IDX] = -1.0
    return J


def _optimise(z0):
    z0 = np.asarray(z0, dtype=float).copy()
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    z0 = np.clip(z0, [b[0] for b in bounds], [b[1] for b in bounds])
    res = minimize(
        _objective,
        z0,
        jac=_objective_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": _constraints, "jac": _constraints_jac}],
        options={"maxiter": 400, "ftol": 1e-12},
    )
    return res.x


def _repair(c, r):
    """Return a strictly valid packing: centres in the square, radii shrunk to fit."""
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    border = np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1.0 - c[:, 0], 1.0 - c[:, 1]))
    r = np.minimum(np.maximum(np.asarray(r, dtype=float), 0.0), border)
    for _ in range(2000):
        d = np.sqrt(np.sum((c[_PI] - c[_PJ]) ** 2, axis=1))
        s = r[_PI] + r[_PJ]
        bad = s > d
        if not np.any(bad):
            break
        ratio = np.ones(_NP)
        ratio[bad] = d[bad] / s[bad]
        factor = np.ones(N)
        np.minimum.at(factor, _PI, ratio)
        np.minimum.at(factor, _PJ, ratio)
        r = r * factor * (1.0 - 1e-12)
    r = np.maximum(r * (1.0 - 1e-10), 0.0)  # safety margin for float rounding in the checker
    return c, r


def _random_start(rng):
    c = rng.uniform(0.05, 0.95, size=(N, 2))
    r = np.full(N, 0.02)
    return np.concatenate([c.ravel(), r])


def _perturb(z, rng):
    c, r = _unpack(z)
    c = c.copy()
    if rng.random() < 0.5:
        c = c + rng.normal(0.0, 0.03, size=c.shape)
    else:
        # relocate a couple of circles to random spots (basin hopping move)
        for i in rng.choice(N, size=2, replace=False):
            c[i] = rng.uniform(0.05, 0.95, size=2)
    c = np.clip(c, 0.0, 1.0)
    r = np.maximum(r * 0.9, 0.0)
    return np.concatenate([c.ravel(), r])


def _search():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    best = None  # (sum, centres, radii)
    best_z = None
    while True:
        if best_z is None or rng.random() < 0.35:
            z0 = _random_start(rng)
        else:
            z0 = _perturb(best_z, rng)
        try:
            z = _optimise(z0)
            if np.all(np.isfinite(z)):
                c, r = _repair(*_unpack(z))
                val = float(np.sum(r))
                if best is None or val > best[0]:
                    best = (val, c, r)
                    best_z = np.concatenate([c.ravel(), r])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass
        elapsed = time.time() - t0
        if best is not None and elapsed > TIME_BUDGET:
            break
        if best is None and elapsed > HARD_LIMIT:
            c, r = _repair(*_unpack(_random_start(rng)))
            best = (float(np.sum(r)), c, r)
            break
    return best


def construct_packing():
    _, centers, radii = _search()
    return centers, radii


def run_packing():
    """Run the optimised packing for n=26 and return (centers, radii, sum_radii)."""
    centers, radii = construct_packing()
    centers = np.asarray(centers, dtype=float).reshape(N, 2)
    radii = np.asarray(radii, dtype=float).reshape(N)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
