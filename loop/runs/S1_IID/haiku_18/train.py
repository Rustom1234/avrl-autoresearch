# EVOLVE-BLOCK-START
"""Multi-start SLSQP circle packing for n=26 circles in the unit square.

Joint variables (x, y, r) are optimised with SLSQP to maximise sum(r) under
containment and pairwise non-overlap constraints, with analytic Jacobians.
Starts are seeded random layouts or perturbations of the best layout so far
(jitter, or relocating the smallest circle). Each optimised layout keeps its
centers; the radii are then recomputed by an LP with the exact pairwise
constraints r_i + r_j <= d_ij, shrunk by a tiny margin and verified.
"""

import time

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
TIME_BUDGET = 170.0  # seconds of search; the harness stops jobs at 300 s
_PI, _PJ = np.triu_indices(N, 1)
NP = len(_PI)
_IDX = np.arange(N)
_PAIR = np.arange(NP)


def _split(v):
    return v[:N], v[N:2 * N], v[2 * N:]


def _cons(v):
    x, y, r = _split(v)
    dx = x[_PI] - x[_PJ]
    dy = y[_PI] - y[_PJ]
    rs = r[_PI] + r[_PJ]
    return np.concatenate([
        dx * dx + dy * dy - rs * rs,  # non-overlap
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r,
    ])


def _cons_jac(v):
    x, y, r = _split(v)
    dx = x[_PI] - x[_PJ]
    dy = y[_PI] - y[_PJ]
    rs = r[_PI] + r[_PJ]
    J = np.zeros((NP + 4 * N, 3 * N))
    J[_PAIR, _PI] = 2 * dx
    J[_PAIR, _PJ] = -2 * dx
    J[_PAIR, N + _PI] = 2 * dy
    J[_PAIR, N + _PJ] = -2 * dy
    J[_PAIR, 2 * N + _PI] = -2 * rs
    J[_PAIR, 2 * N + _PJ] = -2 * rs
    b = NP
    J[b + _IDX, _IDX] = 1.0  # x - r
    J[b + _IDX, 2 * N + _IDX] = -1.0
    b += N
    J[b + _IDX, _IDX] = -1.0  # 1 - x - r
    J[b + _IDX, 2 * N + _IDX] = -1.0
    b += N
    J[b + _IDX, N + _IDX] = 1.0  # y - r
    J[b + _IDX, 2 * N + _IDX] = -1.0
    b += N
    J[b + _IDX, N + _IDX] = -1.0  # 1 - y - r
    J[b + _IDX, 2 * N + _IDX] = -1.0
    return J


def _obj(v):
    return -np.sum(v[2 * N:])


def _obj_grad(v):
    g = np.zeros(3 * N)
    g[2 * N:] = -1.0
    return g


def _optimize(v0):
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    res = minimize(_obj, v0, jac=_obj_grad, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": _cons, "jac": _cons_jac}],
                   options={"maxiter": 400, "ftol": 1e-10})
    return res.x


def _wall(c):
    return np.min(np.stack([c[:, 0], c[:, 1], 1.0 - c[:, 0], 1.0 - c[:, 1]]), axis=0)


def _radii_lp(c, wall):
    A = np.zeros((NP, N))
    A[_PAIR, _PI] = 1.0
    A[_PAIR, _PJ] = 1.0
    d = np.linalg.norm(c[_PI] - c[_PJ], axis=1)
    res = linprog(-np.ones(N), A_ub=A, b_ub=d,
                  bounds=[(0.0, w) for w in wall], method="highs")
    if res.status != 0 or res.x is None:
        return None
    return np.maximum(res.x, 0.0)


def _valid(c, r):
    if np.any(r < 0.0):
        return False
    if np.any(c - r[:, None] < 0.0) or np.any(c + r[:, None] > 1.0):
        return False
    d = np.linalg.norm(c[_PI] - c[_PJ], axis=1)
    return bool(np.all(d >= r[_PI] + r[_PJ]))


def _safe_radii(c):
    wall = _wall(c)
    r = _radii_lp(c, wall)
    if r is not None:
        r = np.minimum(r, wall) * (1.0 - 1e-6)
        for _ in range(20):
            if _valid(c, r):
                return r
            r = r * (1.0 - 1e-6)
    # Guaranteed-valid fallback: one uniform radius.
    d = np.linalg.norm(c[_PI] - c[_PJ], axis=1)
    r0 = min(float(wall.min()), float(d.min()) / 2.0) * (1.0 - 1e-9)
    return np.full(N, max(r0, 0.0))


def _layout(c):
    c = np.clip(c, 0.0, 1.0)
    r = _safe_radii(c)
    return float(np.sum(r)), c, r


def _search():
    rng = np.random.default_rng(20260117)
    t0 = time.perf_counter()
    best = (-1.0, None, None, None)  # score, centers, radii, SLSQP vector
    while time.perf_counter() - t0 < TIME_BUDGET:
        if best[3] is None or rng.random() < 0.35:
            c0 = rng.uniform(0.0, 1.0, (N, 2))
            r0 = np.full(N, 0.02)
        else:
            v = best[3]
            c0 = np.stack([v[:N], v[N:2 * N]], axis=1)
            r0 = v[2 * N:].copy()
            if rng.random() < 0.3:
                k = int(np.argmin(r0))  # relocate the smallest circle
                c0[k] = rng.uniform(0.0, 1.0, 2)
                r0[k] = 0.02
            else:
                c0 = c0 + rng.normal(0.0, rng.choice([0.01, 0.03, 0.08]), (N, 2))
        c0 = np.clip(c0, 0.0, 1.0)
        v0 = np.concatenate([c0[:, 0], c0[:, 1], np.clip(r0, 0.0, 0.5)])
        try:
            v = _optimize(v0)
        except Exception:
            continue
        if not np.all(np.isfinite(v)):
            continue
        s, c, r = _layout(np.stack([v[:N], v[N:2 * N]], axis=1))
        if s > best[0]:
            best = (s, c, r, v)
    return best


def construct_packing():
    """Return (centers, radii) for 26 non-overlapping circles in the unit square."""
    score, centers, radii, _ = _search()
    if centers is None:
        score, centers, radii = _layout(np.random.default_rng(0).uniform(0.05, 0.95, (N, 2)))
    return centers, radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
