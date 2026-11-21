# EVOLVE-BLOCK-START
"""Multistart SLSQP + monotone basin hopping for 26 circles in the unit square (max sum of radii)"""

import time

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
PI_I, PI_J = np.triu_indices(N, k=1)
NP = PI_I.size  # 325 pair constraints
_ROWS = np.arange(NP)
_SEED = 20240601
_TIME_BUDGET = 190.0  # seconds; harness stops at 300 s, target is about 4 min
_N_STARTS = 8
_N_HOPS = 600
_OBJ_GRAD = np.concatenate([np.zeros(2 * N), -np.ones(N)])
_BOUNDS = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N


def _unpack(z):
    return z[:N], z[N:2 * N], z[2 * N:]


def _pack_z(c, r):
    return np.concatenate([c[:, 0], c[:, 1], r])


def _cons(z):
    x, y, r = _unpack(z)
    dx = x[PI_I] - x[PI_J]
    dy = y[PI_I] - y[PI_J]
    rs = r[PI_I] + r[PI_J]
    pair = dx * dx + dy * dy - rs * rs
    bnd = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    return np.concatenate([pair, bnd])


def _cons_jac(z):
    x, y, r = _unpack(z)
    dx = x[PI_I] - x[PI_J]
    dy = y[PI_I] - y[PI_J]
    rs = r[PI_I] + r[PI_J]
    jac = np.zeros((NP + 4 * N, 3 * N))
    jac[_ROWS, PI_I] = 2 * dx
    jac[_ROWS, PI_J] = -2 * dx
    jac[_ROWS, N + PI_I] = 2 * dy
    jac[_ROWS, N + PI_J] = -2 * dy
    jac[_ROWS, 2 * N + PI_I] = -2 * rs
    jac[_ROWS, 2 * N + PI_J] = -2 * rs
    eye = np.eye(N)
    b = NP
    jac[b:b + N, :N] = eye
    jac[b:b + N, 2 * N:] = -eye
    jac[b + N:b + 2 * N, :N] = -eye
    jac[b + N:b + 2 * N, 2 * N:] = -eye
    jac[b + 2 * N:b + 3 * N, N:2 * N] = eye
    jac[b + 2 * N:b + 3 * N, 2 * N:] = -eye
    jac[b + 3 * N:b + 4 * N, N:2 * N] = -eye
    jac[b + 3 * N:b + 4 * N, 2 * N:] = -eye
    return jac


def _slsqp(z0):
    res = minimize(
        lambda z: -np.sum(z[2 * N:]),
        z0,
        jac=lambda z: _OBJ_GRAD,
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": _cons, "jac": _cons_jac}],
        bounds=_BOUNDS,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    return res.x


def _radii_for(c):
    """Optimal radii for fixed centers via LP (maximise sum of radii)."""
    x, y = c[:, 0], c[:, 1]
    bnd = np.maximum(np.minimum(np.minimum(x, y), np.minimum(1.0 - x, 1.0 - y)), 0.0)
    d = np.hypot(x[PI_I] - x[PI_J], y[PI_I] - y[PI_J])
    a_ub = np.zeros((NP, N))
    a_ub[_ROWS, PI_I] = 1.0
    a_ub[_ROWS, PI_J] = 1.0
    res = linprog(
        -np.ones(N),
        A_ub=a_ub,
        b_ub=d,
        bounds=[(0.0, float(v)) for v in bnd],
        method="highs",
    )
    if res.status != 0:
        return np.zeros(N)
    return np.maximum(res.x, 0.0)


def _score_z(z):
    c = np.clip(np.column_stack([z[:N], z[N:2 * N]]), 0.0, 1.0)
    r = _radii_for(c)
    return float(r.sum()), c, r


def _is_valid(c, r):
    x, y = c[:, 0], c[:, 1]
    if np.any(r < 0.0):
        return False
    if np.any(x - r < 0.0) or np.any(y - r < 0.0):
        return False
    if np.any(x + r > 1.0) or np.any(y + r > 1.0):
        return False
    d = np.hypot(x[PI_I] - x[PI_J], y[PI_I] - y[PI_J])
    return bool(np.all(d >= r[PI_I] + r[PI_J]))


def _shrink(c, r):
    r = np.maximum(r * (1.0 - 1e-9) - 1e-12, 0.0)
    for _ in range(60):
        if _is_valid(c, r):
            return r
        r = np.maximum(r * (1.0 - 1e-7), 0.0)
    return r


def _perturb(z, rng):
    x, y, r = _unpack(z)
    x = x.copy()
    y = y.copy()
    sigma = rng.choice([0.005, 0.02, 0.05])
    x += rng.normal(0.0, sigma, N)
    y += rng.normal(0.0, sigma, N)
    for k in rng.choice(N, size=int(rng.integers(0, 3)), replace=False):
        x[k] = rng.uniform(0.05, 0.95)
        y[k] = rng.uniform(0.05, 0.95)
    r = np.maximum(r * 0.9, 0.01)
    return np.concatenate([np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0), r])


def construct_packing():
    """
    Find a packing of 26 circles in the unit square maximising the sum of radii.

    Returns:
        Tuple of (centers, radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(_SEED)

    # Multistart: random centers, joint SLSQP over (x, y, r), repaired radii via LP.
    cur_val, cur_c, cur_r = -1.0, None, None
    for _ in range(_N_STARTS):
        c0 = rng.uniform(0.08, 0.92, size=(N, 2))
        z0 = _pack_z(c0, np.full(N, 0.03))
        val, c, r = _score_z(_slsqp(z0))
        if val > cur_val:
            cur_val, cur_c, cur_r = val, c, r

    # Monotone basin hopping around the incumbent.
    for _ in range(_N_HOPS):
        if time.perf_counter() - t0 > _TIME_BUDGET:
            break
        z = _perturb(_pack_z(cur_c, cur_r), rng)
        val, c, r = _score_z(_slsqp(z))
        if val > cur_val + 1e-10:
            cur_val, cur_c, cur_r = val, c, r

    radii = _shrink(cur_c, cur_r)
    return cur_c, radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
