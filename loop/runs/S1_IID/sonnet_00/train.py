import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
IU, JU = np.triu_indices(N, 1)
NP = len(IU)
ROWS = np.arange(NP)


def _cons(z):
    x = z[:N]; y = z[N:2 * N]; r = z[2 * N:]
    dx = x[IU] - x[JU]; dy = y[IU] - y[JU]
    d = np.sqrt(dx * dx + dy * dy + 1e-18)
    return np.concatenate([d - r[IU] - r[JU], x - r, 1 - x - r, y - r, 1 - y - r])


def _jac(z):
    x = z[:N]; y = z[N:2 * N]
    dx = x[IU] - x[JU]; dy = y[IU] - y[JU]
    d = np.sqrt(dx * dx + dy * dy + 1e-18)
    J = np.zeros((NP + 4 * N, 3 * N))
    ux = dx / d; uy = dy / d
    J[ROWS, IU] = ux; J[ROWS, JU] = -ux
    J[ROWS, N + IU] = uy; J[ROWS, N + JU] = -uy
    J[ROWS, 2 * N + IU] = -1; J[ROWS, 2 * N + JU] = -1
    k = np.arange(N)
    o = NP
    J[o + k, k] = 1; J[o + k, 2 * N + k] = -1
    o += N
    J[o + k, k] = -1; J[o + k, 2 * N + k] = -1
    o += N
    J[o + k, N + k] = 1; J[o + k, 2 * N + k] = -1
    o += N
    J[o + k, N + k] = -1; J[o + k, 2 * N + k] = -1
    return J


def _obj(z):
    return -np.sum(z[2 * N:])


def _grad(z):
    g = np.zeros(3 * N)
    g[2 * N:] = -1
    return g


def _init(rng, mode):
    if mode == 0:
        # jittered rows layout with 5 rows (5,5,6,5,5)
        counts = [5, 5, 6, 5, 5]
        pts = []
        for ri, c in enumerate(counts):
            yy = (ri + 0.5) / len(counts)
            for ci in range(c):
                pts.append([(ci + 0.5) / c, yy])
        p = np.array(pts) + rng.normal(0, 0.03, (N, 2))
    elif mode == 1:
        g = np.array([[i, j] for i in range(5) for j in range(5)], float)
        g = (g + 0.5) / 5
        g = np.vstack([g, [rng.uniform(0.1, 0.9, 2)]])
        p = g + rng.normal(0, 0.04, (N, 2))
    else:
        p = rng.uniform(0.05, 0.95, (N, 2))
    p = np.clip(p, 0.03, 0.97)
    r = np.full(N, rng.uniform(0.04, 0.09))
    return np.concatenate([p[:, 0], p[:, 1], r])


def _solve(z0, maxiter):
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    res = minimize(_obj, z0, jac=_grad, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": _cons, "jac": _jac}],
                   options={"maxiter": maxiter, "ftol": 1e-12})
    return res.x


def _lp_radii(c):
    d = np.sqrt(((c[IU] - c[JU]) ** 2).sum(1))
    A = np.zeros((NP, N))
    A[ROWS, IU] = 1; A[ROWS, JU] = 1
    ub = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
    ub = np.maximum(ub, 0)
    res = linprog(-np.ones(N), A_ub=A, b_ub=d, bounds=[(0, u) for u in ub],
                  method="highs")
    if res.status == 0:
        return np.maximum(res.x, 0)
    return None


def _make_valid(c, r):
    c = np.clip(c, 0.0, 1.0)
    r = np.maximum(r - 1e-11, 0)
    r = np.minimum(r, np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]]))
    r = np.maximum(r, 0)
    for _ in range(50):
        d = np.sqrt(((c[IU] - c[JU]) ** 2).sum(1))
        bad = r[IU] + r[JU] > d
        if not bad.any():
            break
        for k in np.where(bad)[0]:
            i, j = IU[k], JU[k]
            s = r[i] + r[j]
            if s > d[k]:
                f = d[k] / s * (1 - 1e-12)
                r[i] *= f; r[j] *= f
    return c, r


def construct_packing():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    best_s = -1.0
    best_c = None
    it = 0
    while time.time() - t0 < 150:
        z0 = _init(rng, it % 3)
        it += 1
        try:
            z = _solve(z0, 400)
        except Exception:
            continue
        c = np.stack([z[:N], z[N:2 * N]], 1)
        r = z[2 * N:]
        viol = -min(_cons(z).min(), 0)
        s = r.sum() - 26 * viol
        if viol < 1e-7 and s > best_s:
            best_s = s
            best_c = c.copy()
    if best_c is None:
        best_c = _init(rng, 1)[:2 * N].reshape(2, N).T
    # polish: a few extra SLSQP passes from the best, then LP radii
    z = np.concatenate([best_c[:, 0], best_c[:, 1], np.full(N, 0.05)])
    r0 = _lp_radii(best_c)
    if r0 is not None:
        z[2 * N:] = r0
        try:
            z2 = _solve(z, 500)
            c2 = np.stack([z2[:N], z2[N:2 * N]], 1)
            r2 = _lp_radii(np.clip(c2, 0, 1))
            if r2 is not None and r2.sum() >= r0.sum():
                best_c = np.clip(c2, 0, 1)
        except Exception:
            pass
    r = _lp_radii(best_c)
    if r is None:
        r = np.full(N, 0.01)
    c, r = _make_valid(best_c, r)
    return c, r


# EVOLVE-BLOCK-END


def run_packing():
    centers, radii = construct_packing()
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
