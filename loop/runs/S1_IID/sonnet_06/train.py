# EVOLVE-BLOCK-START
"""Multi-start SLSQP circle packing for n=26 with LP polish and safe repair."""

import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_, J_ = np.triu_indices(N, 1)
M = len(I_)


def _cons_factory():
    rows = np.arange(M)
    n = N

    def f(v):
        x = v[:n]
        y = v[n:2 * n]
        r = v[2 * n:]
        pair = (x[I_] - x[J_]) ** 2 + (y[I_] - y[J_]) ** 2 - (r[I_] + r[J_]) ** 2
        return np.concatenate([pair, x - r, 1 - x - r, y - r, 1 - y - r])

    def jac(v):
        x = v[:n]
        y = v[n:2 * n]
        r = v[2 * n:]
        Jm = np.zeros((M + 4 * n, 3 * n))
        dx = x[I_] - x[J_]
        dy = y[I_] - y[J_]
        sr = r[I_] + r[J_]
        Jm[rows, I_] = 2 * dx
        Jm[rows, J_] = -2 * dx
        Jm[rows, n + I_] = 2 * dy
        Jm[rows, n + J_] = -2 * dy
        Jm[rows, 2 * n + I_] = -2 * sr
        Jm[rows, 2 * n + J_] = -2 * sr
        k = np.arange(n)
        b = M
        Jm[b + k, k] = 1
        Jm[b + k, 2 * n + k] = -1
        b += n
        Jm[b + k, k] = -1
        Jm[b + k, 2 * n + k] = -1
        b += n
        Jm[b + k, n + k] = 1
        Jm[b + k, 2 * n + k] = -1
        b += n
        Jm[b + k, n + k] = -1
        Jm[b + k, 2 * n + k] = -1
        return Jm

    return f, jac


_F, _JAC = _cons_factory()
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
            constraints=[{"type": "ineq", "fun": _F, "jac": _JAC}],
            options={"maxiter": maxiter, "ftol": 1e-12},
        )
        return res.x
    except Exception:
        return v0


def _lp_radii(c):
    d = np.sqrt(np.sum((c[I_] - c[J_]) ** 2, axis=1))
    A = np.zeros((M, N))
    A[np.arange(M), I_] = 1
    A[np.arange(M), J_] = 1
    ub = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
    ub = np.maximum(ub, 0)
    try:
        res = linprog(-np.ones(N), A_ub=A, b_ub=d,
                      bounds=[(0, float(u)) for u in ub], method="highs")
        if res.status == 0:
            return res.x
    except Exception:
        pass
    return None


def _violations(c, r):
    d = np.sqrt(np.sum((c[I_] - c[J_]) ** 2, axis=1))
    d2 = np.sum((c[I_] - c[J_]) ** 2, axis=1)
    return d - (r[I_] + r[J_]), d2 - (r[I_] + r[J_]) ** 2


def _repair(c, r):
    c = np.clip(c, 0.0, 1.0)
    r = np.maximum(r, 0.0)
    r = np.minimum(r, np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]]))
    r = r * (1 - 1e-10) - 1e-13
    r = np.maximum(r, 0.0)
    for _ in range(200):
        g1, g2 = _violations(c, r)
        bad = np.where((g1 < 1e-13) | (g2 < 0))[0]
        if len(bad) == 0:
            break
        for k in bad:
            i, j = I_[k], J_[k]
            ex = (r[i] + r[j]) - np.sqrt(np.sum((c[i] - c[j]) ** 2)) + 1e-12
            ex = max(ex, 1e-12)
            r[i] = max(r[i] - ex, 0.0)
            r[j] = max(r[j] - ex, 0.0)
    b = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
    r = np.minimum(r, b)
    return r


def _finalize(v):
    c = np.column_stack([v[:N], v[N:2 * N]])
    c = np.clip(c, 0.0, 1.0)
    r0 = _lp_radii(c)
    if r0 is None:
        r0 = np.maximum(v[2 * N:], 0.0)
    r = _repair(c, r0)
    g1, g2 = _violations(c, r)
    ok = (g1.min() >= 0) and (g2.min() >= 0) and np.all(r >= 0)
    return c, r, ok


def _start(rng, kind):
    if kind == 0:
        c = rng.random((N, 2)) * 0.9 + 0.05
    elif kind == 1:
        # perturbed grid with rows of 5/6 style layouts
        rows = rng.choice([5, 6])
        pts = []
        per = [5, 5, 5, 5, 6] if rows == 5 else [4, 5, 4, 5, 4, 4]
        per = list(per)
        while sum(per) < N:
            per[rng.integers(len(per))] += 1
        while sum(per) > N:
            k = rng.integers(len(per))
            if per[k] > 1:
                per[k] -= 1
        nr = len(per)
        for ri, cnt in enumerate(per):
            yy = (ri + 0.5) / nr
            for ci in range(cnt):
                pts.append([(ci + 0.5 + 0.25 * (ri % 2)) / (cnt + 0.25), yy])
        c = np.array(pts) + rng.normal(0, 0.02, (N, 2))
        c = np.clip(c, 0.03, 0.97)
    else:
        # corners/edges plus random interior
        c = rng.random((N, 2))
        k = rng.integers(4, 12)
        c[:4] = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        for i in range(4, k):
            c[i] = [rng.choice([0.08, 0.92]) if rng.random() < 0.5 else rng.random(),
                    rng.random()]
        c = np.clip(c + rng.normal(0, 0.01, c.shape), 0.03, 0.97)
    r = np.full(N, 0.04) + rng.random(N) * 0.03
    return np.concatenate([c[:, 0], c[:, 1], r])


def construct_packing():
    t0 = time.time()
    budget = 150.0
    rng = np.random.default_rng(12345)
    best = None
    best_val = -1.0
    it = 0
    last = 0.0
    while True:
        el = time.time() - t0
        if el > budget or (best is not None and last > 0 and el + 1.5 * last > budget + 30):
            break
        ts = time.time()
        v0 = _start(rng, it % 3)
        v = _solve(v0, 250)
        # one refinement round
        v = _solve(v, 150)
        c, r, ok = _finalize(v)
        if ok and r.sum() > best_val:
            best_val = r.sum()
            best = (c.copy(), r.copy())
        last = time.time() - ts
        it += 1
    if best is None:
        # fallback: tiny safe grid
        g = np.linspace(0.1, 0.9, 6)
        pts = np.array([[a, b] for a in g for b in g])[:N]
        r = np.full(N, 0.07)
        return pts, r
    return best


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
