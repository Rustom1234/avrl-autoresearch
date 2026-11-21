#!/usr/bin/env python3
"""Score a PART of a law: the candidate plus one free smooth curve of each input named in --free.

    fit_part.py --code cand.py --train train.csv --free t P

The free curves are cubic splines with inner knots at quantiles of the input. For any constants of the candidate
the best curves are a linear least-squares fit, so the curves are projected out and only the candidate's constants
are searched, with the optimizer of fit.py and its starting points plus 16 more (no randomness, no dependence on
run time).

    share = squares left by candidate + curves  /  squares left by the curves alone

A candidate that adds nothing gives 1. A candidate that holds exactly the terms the curves cannot express gives a
share near 0, but not 0, because splines are approximate. A fixed cutoff cannot tell such a part from a near miss
(offline check, SR6_PREREG.md), so the fit is done twice, with COARSE and with FINE curves: for a right part the
share keeps falling as the curves get finer, for a wrong part it stalls. Prints one JSON object.
"""
import argparse, importlib.util, json, os, sys, warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

COARSE, FINE = 20, 40      # inner knots of the two stand-ins
_spec = importlib.util.spec_from_file_location("fit", Path(__file__).resolve().parent.parent / "fit.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)


def starts():
    """The 8 starting points of fit.py plus 16 with magnitudes spread from 0.03 to 10 (fixed seed). The offline check
    showed that 8 points of order 1 can miss the constants of a right part (a tanh or sine with a small rate)."""
    r = np.random.default_rng(54321)
    return F.starts() + [np.exp(r.uniform(np.log(0.03), np.log(10.0), F.MAXP)) for _ in range(16)]


def basis(x, k):
    inner = np.quantile(x, np.linspace(0, 1, k + 2)[1:-1])
    t = np.r_[[x.min()] * 4, inner, [x.max()] * 4]
    return BSpline.design_matrix(x, t, 3).toarray()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--free", nargs="+", required=True)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    code = open(a.code).read()
    if len(code) > F.MAX_CODE:
        return {"ok": False, "error": "the equation code is longer than %d characters" % F.MAX_CODE}
    names = open(a.train).readline().strip().split(",")[:-1]
    bad = [v for v in a.free if v not in names]
    if bad:
        return {"ok": False, "error": "free must name inputs out of %s; got %s" % (names, bad)}
    ns = {"np": np, "numpy": np}
    try:
        exec(compile(code, "candidate", "exec"), ns)
        f = ns["equation"]
    except Exception as e:
        return {"ok": False, "error": "could not define equation(): %s: %s" % (type(e).__name__, str(e)[:300])}
    X, y = F.load(a.train)
    total = float(np.sum((y - y.mean()) ** 2))
    try:
        with np.errstate(all="ignore"):
            np.broadcast_to(np.asarray(f(*X, np.ones(F.MAXP)), dtype=float), y.shape)
    except Exception as e:
        return {"ok": False, "error": "equation() failed on the data: %s: %s" % (type(e).__name__, str(e)[:300])}

    def fit_at(k, points, budget):
        """Best share over the starting points. `budget` caps the total number of model evaluations, so the cost of
        one call is bounded without any clock (the result never depends on run time)."""
        B = np.hstack([np.ones((len(y), 1))] + [basis(X[names.index(v)], k) for v in dict.fromkeys(a.free)])
        U, sv, _ = np.linalg.svd(B, full_matrices=False)
        Q = U[:, sv > 1e-10 * sv[0]]
        out_of = lambda r: r - Q @ (Q.T @ r)          # the part of r that the free curves cannot express
        den = float(np.sum(out_of(y) ** 2))
        if den < 1e-12 * total:
            return None, den / total, []

        def res(p):
            with np.errstate(all="ignore"):
                out = np.broadcast_to(np.asarray(f(*X, p), dtype=float), y.shape)
            if not np.all(np.isfinite(out)):
                return np.full(y.shape, 1e6)
            return out_of(out - y) / np.sqrt(den)

        best, spent = (float("inf"), np.ones(F.MAXP)), 0
        for p0 in points:
            if spent >= budget:
                break
            for method in ("lm", "trf"):
                try:
                    sol = least_squares(res, p0, method=method, max_nfev=min(3000, budget - spent), xtol=1e-15,
                                        ftol=1e-15, gtol=1e-15)
                except Exception:
                    continue
                spent += int(sol.nfev) * (1 if method == "lm" else F.MAXP + 1)   # trf counts Jacobians apart
                v = float(np.sum(res(sol.x) ** 2))
                if v < best[0]:
                    best = (v, sol.x)
                break
            if best[0] < 1e-20:
                break
        return best[0], den / total, [float(x) for x in best[1]]

    fine, alone, params = fit_at(FINE, starts(), 30000)
    if fine is None:
        return {"ok": True, "share": None, "share_coarse": None, "right": False, "curves_alone": alone, "params": []}
    coarse, _, _ = fit_at(COARSE, [np.array(params)] + F.starts(), 12000)
    if coarse is None:
        return {"ok": True, "share": None, "share_coarse": None, "right": False, "curves_alone": alone, "params": []}
    return {"ok": True, "share": fine, "share_coarse": coarse, "right": is_right(coarse, fine), "curves_alone": alone,
            "params": params}


def is_right(coarse, fine):
    """A right part is limited only by the curves, so its share keeps falling when the curves get finer."""
    return bool(fine < 1e-9 or (fine < 1e-4 and fine < coarse / 3))


if __name__ == "__main__":
    try:
        r = main()
    except Exception as e:                      # never a traceback on stdout: the caller parses JSON
        r = {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:300])}
    json.dump(r, sys.stdout)
