# EVOLVE-BLOCK-START
"""Circle packing for n=26: seeded multi-start SLSQP plus basin hopping."""

import time
import numpy as np
from scipy.optimize import minimize

N = 26
IU = np.triu_indices(N, 1)
I_IDX, J_IDX = IU
NP = len(I_IDX)
ROWS = np.arange(NP)


def _cons_fun(v):
    x = v[:N]
    y = v[N:2 * N]
    r = v[2 * N:]
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    s = r[I_IDX] + r[J_IDX]
    pair = dx * dx + dy * dy - s * s
    return np.concatenate([pair, x - r, 1 - x - r, y - r, 1 - y - r])


def _cons_jac(v):
    x = v[:N]
    y = v[N:2 * N]
    r = v[2 * N:]
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    s = r[I_IDX] + r[J_IDX]
    J = np.zeros((NP + 4 * N, 3 * N))
    J[ROWS, I_IDX] = 2 * dx
    J[ROWS, J_IDX] = -2 * dx
    J[ROWS, N + I_IDX] = 2 * dy
    J[ROWS, N + J_IDX] = -2 * dy
    J[ROWS, 2 * N + I_IDX] = -2 * s
    J[ROWS, 2 * N + J_IDX] = -2 * s
    k = np.arange(N)
    b = NP
    J[b + k, k] = 1
    J[b + k, 2 * N + k] = -1
    b += N
    J[b + k, k] = -1
    J[b + k, 2 * N + k] = -1
    b += N
    J[b + k, N + k] = 1
    J[b + k, 2 * N + k] = -1
    b += N
    J[b + k, N + k] = -1
    J[b + k, 2 * N + k] = -1
    return J


_OBJ_G = np.concatenate([np.zeros(2 * N), -np.ones(N)])


def _solve(v0, maxiter=300):
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    try:
        res = minimize(
            lambda v: -np.sum(v[2 * N:]),
            v0,
            jac=lambda v: _OBJ_G,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": _cons_fun, "jac": _cons_jac}],
            options={"maxiter": maxiter, "ftol": 1e-12},
        )
        return res.x
    except Exception:
        return None


def _feasible_radii(c):
    """Make radii valid for given centers (exact repair), maximal-ish."""
    c = np.clip(c, 0.0, 1.0)
    r = np.minimum.reduce([c[:, 0], 1 - c[:, 0], c[:, 1], 1 - c[:, 1]])
    r = np.maximum(r, 0.0)
    return c, r


def _repair(v):
    c = np.stack([v[:N], v[N:2 * N]], axis=1)
    c = np.clip(c, 0.0, 1.0)
    r = np.maximum(v[2 * N:], 0.0)
    wall = np.minimum.reduce([c[:, 0], 1 - c[:, 0], c[:, 1], 1 - c[:, 1]])
    r = np.minimum(r, wall) - 1e-11
    r = np.maximum(r, 0.0)
    for _ in range(200):
        d = np.sqrt(np.sum((c[I_IDX] - c[J_IDX]) ** 2, axis=1))
        over = r[I_IDX] + r[J_IDX] - d + 1e-12
        bad = np.where(over > 0)[0]
        if len(bad) == 0:
            break
        for k in bad:
            i, j = I_IDX[k], J_IDX[k]
            cut = over[k] / 2 + 1e-12
            r[i] = max(r[i] - cut, 0.0)
            r[j] = max(r[j] - cut, 0.0)
    return c, r


def _score(v):
    c, r = _repair(v)
    return float(np.sum(r))


def _init_random(rng):
    c = rng.random((N, 2)) * 0.9 + 0.05
    r = np.full(N, 0.05) + rng.random(N) * 0.03
    return np.concatenate([c[:, 0], c[:, 1], r])


def _init_grid(rng):
    # rows pattern with perturbation, e.g. 5,5,6,5,5 or 6,5,5,5,5 etc
    patterns = [
        [5, 5, 6, 5, 5], [6, 5, 5, 5, 5], [5, 6, 5, 6, 4],
        [4, 5, 4, 5, 4, 4], [5, 5, 5, 5, 6], [6, 5, 6, 5, 4],
    ]
    p = patterns[rng.integers(len(patterns))]
    pts = []
    m = len(p)
    for ri, cnt in enumerate(p):
        yy = (ri + 0.5) / m
        for k in range(cnt):
            pts.append([(k + 0.5) / cnt, yy])
    pts = np.array(pts)[:N]
    if len(pts) < N:
        pts = np.vstack([pts, rng.random((N - len(pts), 2))])
    pts = pts + rng.normal(0, 0.03, pts.shape)
    pts = np.clip(pts, 0.03, 0.97)
    r = np.full(N, 0.08) + rng.normal(0, 0.01, N)
    return np.concatenate([pts[:, 0], pts[:, 1], np.clip(r, 0.02, 0.15)])


def construct_packing():
    t0 = time.time()
    budget = 170.0
    rng = np.random.default_rng(12345)
    best_v, best_s = None, -1.0
    pool = []

    # Phase 1: multistart
    phase1 = budget * 0.55
    it = 0
    while time.time() - t0 < phase1:
        v0 = _init_grid(rng) if it % 2 == 0 else _init_random(rng)
        it += 1
        v = _solve(v0)
        if v is None:
            continue
        s = _score(v)
        pool.append((s, v))
        if s > best_s:
            best_s, best_v = s, v
    if best_v is None:
        best_v = _init_grid(rng)
        best_s = _score(best_v)

    # Phase 2: basin hopping from the best
    cur_v, cur_s = best_v.copy(), best_s
    while time.time() - t0 < budget:
        v0 = cur_v.copy()
        mode = rng.integers(3)
        if mode == 0:
            v0[:2 * N] += rng.normal(0, 0.02, 2 * N)
        elif mode == 1:
            k = rng.integers(1, 4)
            idx = rng.choice(N, k, replace=False)
            v0[idx] = rng.random(k)
            v0[N + idx] = rng.random(k)
            v0[2 * N + idx] = 0.03
        else:
            i, j = rng.choice(N, 2, replace=False)
            for a in (0, N):
                v0[a + i], v0[a + j] = v0[a + j], v0[a + i]
            v0[:2 * N] += rng.normal(0, 0.005, 2 * N)
        v0[:2 * N] = np.clip(v0[:2 * N], 0.0, 1.0)
        v = _solve(v0)
        if v is None:
            continue
        s = _score(v)
        if s > cur_s - 0.002 * rng.random():
            cur_v, cur_s = v, s
        if s > best_s:
            best_s, best_v = s, v
            cur_v, cur_s = v.copy(), s

    c, r = _repair(best_v)
    return c, r


def compute_max_radii(centers):
    c, r = _feasible_radii(np.asarray(centers, dtype=float))
    return r


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
