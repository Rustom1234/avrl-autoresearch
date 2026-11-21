# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""

import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Place circles in a structured pattern
    # This is a simple pattern - evolution will improve this

    # First, place a large circle in the center
    centers[0] = [0.5, 0.5]

    # Place 8 circles around it in a ring
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    # Place 16 more circles in an outer ring
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    # Additional positioning adjustment to make sure all circles
    # are inside the square and don't overlap
    # Clip to ensure everything is inside the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)
    return centers, radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # Then, limit by distance to other circles
            # Each pair of circles with centers at distance d can have
            # sum of radii at most d to avoid overlap
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

                    # If current radii would cause overlap
                    if radii[i] + radii[j] > dist:
                        # Scale both radii proportionally
                        scale = dist / (radii[i] + radii[j])
                        radii[i] *= scale
                        radii[j] *= scale

    # Ensure no negative radii due to numerical issues
    radii = np.maximum(radii, 0.0)
    return radii


def construct_packing():
    import time
    from scipy.optimize import minimize
    n = 26
    t0 = time.time()
    I, J = np.triu_indices(n, 1)
    m = len(I)
    rng = np.random.default_rng(0)

    def pen(v, mu):
        x = v[:n]; y = v[n:2 * n]; r = v[2 * n:]
        dx = x[I] - x[J]; dy = y[I] - y[J]
        d = np.sqrt(dx * dx + dy * dy) + 1e-12
        ov = np.maximum(r[I] + r[J] - d, 0)
        f = -r.sum() + mu * np.sum(ov ** 2)
        gx = np.zeros(n); gy = np.zeros(n); gr = -np.ones(n)
        c = 2 * mu * ov
        np.add.at(gr, I, c); np.add.at(gr, J, c)
        np.add.at(gx, I, -c * dx / d); np.add.at(gx, J, c * dx / d)
        np.add.at(gy, I, -c * dy / d); np.add.at(gy, J, c * dy / d)
        for p, g in ((x, gx), (y, gy)):
            w1 = np.maximum(r - p, 0)
            w2 = np.maximum(p + r - 1, 0)
            f += mu * np.sum(w1 ** 2 + w2 ** 2)
            g += -2 * mu * w1 + 2 * mu * w2
            gr += 2 * mu * (w1 + w2)
        return f, np.concatenate([gx, gy, gr])

    def cons(v):
        x = v[:n]; y = v[n:2 * n]; r = v[2 * n:]
        d = np.sqrt((x[I] - x[J]) ** 2 + (y[I] - y[J]) ** 2)
        return np.concatenate([d - r[I] - r[J], x - r, 1 - x - r, y - r, 1 - y - r])

    def cjac(v):
        x = v[:n]; y = v[n:2 * n]
        dx = x[I] - x[J]; dy = y[I] - y[J]
        d = np.sqrt(dx * dx + dy * dy) + 1e-12
        Jm = np.zeros((m + 4 * n, 3 * n))
        k = np.arange(m)
        Jm[k, I] = dx / d; Jm[k, J] = -dx / d
        Jm[k, n + I] = dy / d; Jm[k, n + J] = -dy / d
        Jm[k, 2 * n + I] = -1; Jm[k, 2 * n + J] = -1
        a = np.arange(n)
        Jm[m + a, a] = 1; Jm[m + a, 2 * n + a] = -1
        Jm[m + n + a, a] = -1; Jm[m + n + a, 2 * n + a] = -1
        Jm[m + 2 * n + a, n + a] = 1; Jm[m + 2 * n + a, 2 * n + a] = -1
        Jm[m + 3 * n + a, n + a] = -1; Jm[m + 3 * n + a, 2 * n + a] = -1
        return Jm

    def finalize(v):
        c = np.clip(np.stack([v[:n], v[n:2 * n]], 1), 0, 1)
        r = np.maximum(v[2 * n:] - 1e-10, 0)
        r = np.minimum(r, np.minimum(np.minimum(c[:, 0], c[:, 1]),
                                     np.minimum(1 - c[:, 0], 1 - c[:, 1])))
        for _ in range(5):
            ok = True
            for i, j in zip(I, J):
                d = np.hypot(*(c[i] - c[j]))
                if r[i] + r[j] > d:
                    s = d / (r[i] + r[j]) * (1 - 1e-12)
                    r[i] *= s; r[j] *= s
                    ok = False
            if ok:
                break
        return c, r

    best = None
    bs = -1
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    bv = None
    while time.time() - t0 < 60:
        v = np.concatenate([rng.uniform(0.05, 0.95, 2 * n), np.full(n, 0.05)])
        for mu in (10, 100, 1e3, 1e4, 1e5):
            v = minimize(pen, v, args=(mu,), jac=True, method="L-BFGS-B",
                         bounds=bounds, options={"maxiter": 400}).x
        try:
            res = minimize(lambda z: -z[2 * n:].sum(), v, jac=lambda z: np.concatenate([np.zeros(2 * n), -np.ones(n)]),
                           method="SLSQP", bounds=bounds,
                           constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                           options={"maxiter": 200, "ftol": 1e-12})
            v = res.x
        except Exception:
            pass
        c, r = finalize(v)
        s = r.sum()
        if s > bs:
            bs = s; best = (c, r); bv = v.copy()

    cur_v = bv.copy(); cur_s = bs
    try:
        n2 = n + 1
        I2, J2 = np.triu_indices(n2, 1)
        m2 = len(I2)

        def cons2(v):
            x = v[:n2]; y = v[n2:2 * n2]; r = v[2 * n2:]
            d = np.sqrt((x[I2] - x[J2]) ** 2 + (y[I2] - y[J2]) ** 2)
            return np.concatenate([d - r[I2] - r[J2], x - r, 1 - x - r, y - r, 1 - y - r])

        def cjac2(v):
            x = v[:n2]; y = v[n2:2 * n2]
            dx = x[I2] - x[J2]; dy = y[I2] - y[J2]
            d = np.sqrt(dx * dx + dy * dy) + 1e-12
            Jm = np.zeros((m2 + 4 * n2, 3 * n2))
            k = np.arange(m2)
            Jm[k, I2] = dx / d; Jm[k, J2] = -dx / d
            Jm[k, n2 + I2] = dy / d; Jm[k, n2 + J2] = -dy / d
            Jm[k, 2 * n2 + I2] = -1; Jm[k, 2 * n2 + J2] = -1
            a = np.arange(n2)
            Jm[m2 + a, a] = 1; Jm[m2 + a, 2 * n2 + a] = -1
            Jm[m2 + n2 + a, a] = -1; Jm[m2 + n2 + a, 2 * n2 + a] = -1
            Jm[m2 + 2 * n2 + a, n2 + a] = 1; Jm[m2 + 2 * n2 + a, 2 * n2 + a] = -1
            Jm[m2 + 3 * n2 + a, n2 + a] = -1; Jm[m2 + 3 * n2 + a, 2 * n2 + a] = -1
            return Jm

        b2 = [(0, 1)] * (2 * n2) + [(0, 0.5)] * n2
        best27 = None; s27 = -1
        while time.time() - t0 < 85:
            v = np.concatenate([rng.uniform(0.05, 0.95, 2 * n2), np.full(n2, 0.05)])
            try:
                res = minimize(lambda z: -z[2 * n2:].sum(), v,
                               jac=lambda z: np.concatenate([np.zeros(2 * n2), -np.ones(n2)]),
                               method="SLSQP", bounds=b2,
                               constraints=[{"type": "ineq", "fun": cons2, "jac": cjac2}],
                               options={"maxiter": 200, "ftol": 1e-10})
            except Exception:
                continue
            z = res.x
            if cons2(z).min() > -1e-6 and z[2 * n2:].sum() > s27:
                s27 = z[2 * n2:].sum(); best27 = z.copy()
        if best27 is not None:
            x27 = best27[:n2]; y27 = best27[n2:2 * n2]; r27 = best27[2 * n2:]
            bd = None; bds = -1
            for kdel in range(n2):
                keep = np.array([q for q in range(n2) if q != kdel])
                w = np.concatenate([x27[keep], y27[keep], r27[keep]])
                try:
                    res = minimize(lambda z: -z[2 * n:].sum(), w,
                                   jac=lambda z: np.concatenate([np.zeros(2 * n), -np.ones(n)]),
                                   method="SLSQP", bounds=bounds,
                                   constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                                   options={"maxiter": 150, "ftol": 1e-12})
                except Exception:
                    continue
                c, r = finalize(res.x)
                s = r.sum()
                if s > bds:
                    bds = s; bd = res.x.copy()
                if s > bs:
                    bs = s; best = (c, r)
            if bd is not None:
                cur_v = bd; cur_s = bds
    except Exception:
        pass
    while time.time() - t0 < 200:
        w = cur_v.copy()
        k = rng.integers(1, 5)
        idx = rng.choice(n, k, replace=False)
        w[idx] += rng.normal(0, 0.05, k)
        w[n + idx] += rng.normal(0, 0.05, k)
        w[:2 * n] = np.clip(w[:2 * n], 0, 1)
        w[2 * n:] = np.minimum(w[2 * n:], 0.5) * 0.9
        try:
            res = minimize(lambda z: -z[2 * n:].sum(), w, jac=lambda z: np.concatenate([np.zeros(2 * n), -np.ones(n)]),
                           method="SLSQP", bounds=bounds,
                           constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                           options={"maxiter": 100, "ftol": 1e-12})
            w = res.x
        except Exception:
            continue
        c, r = finalize(w)
        s = r.sum()
        if s > cur_s or rng.random() < np.exp((s - cur_s) / 0.002):
            cur_v = w; cur_s = s
        if s > bs:
            bs = s; best = (c, r)
    return best


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
