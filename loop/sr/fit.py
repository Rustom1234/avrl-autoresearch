#!/usr/bin/env python3
"""Score one candidate law: fit its constants on the training data, report the error on any split.

    fit.py --code cand.py --train train.csv [--eval val=val.csv test_id=... test_ood=...]

cand.py defines   def equation(<inputs in the order of the csv columns>, params): return <numpy expression>
with at most 10 constants params[0..9]. Constants are fitted by least squares (Levenberg-Marquardt) from 8 fixed
starting points, so the same candidate always gets the same score: no randomness and no dependence on run time.
Error = NMSE = sum (prediction - y)^2 / sum (y - mean y)^2. Prints one JSON object.
Run it in a subprocess under prlimit: the candidate is code written by a model.
"""
import argparse, json, sys, warnings

import numpy as np
from scipy.optimize import least_squares

MAXP = 10
MAX_CODE = 2500


def starts():
    r = np.random.default_rng(12345)
    S = [np.ones(MAXP), np.full(MAXP, 0.5)]
    S += [r.uniform(0.1, 2.0, MAXP) for _ in range(3)] + [r.normal(0.0, 1.0, MAXP) for _ in range(3)]
    return S


def load(path):
    M = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return [M[:, j] for j in range(M.shape[1] - 1)], M[:, -1]


def nmse(f, p, X, y):
    with np.errstate(all="ignore"):
        out = np.broadcast_to(np.asarray(f(*X, p), dtype=float), y.shape)
    v = float(np.sum((out - y) ** 2) / np.sum((y - y.mean()) ** 2))
    return v if np.isfinite(v) else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", nargs="*", default=[])
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    code = open(a.code).read()
    if len(code) > MAX_CODE:
        return {"ok": False, "error": "the equation code is longer than %d characters" % MAX_CODE}
    ns = {"np": np, "numpy": np}
    try:
        exec(compile(code, "candidate", "exec"), ns)
        f = ns["equation"]
    except Exception as e:
        return {"ok": False, "error": "could not define equation(): %s: %s" % (type(e).__name__, str(e)[:300])}
    X, y = load(a.train)
    sd = y.std()

    def res(p):
        with np.errstate(all="ignore"):
            out = np.broadcast_to(np.asarray(f(*X, p), dtype=float), y.shape)
        r = (out - y) / sd
        return np.where(np.isfinite(r), r, 1e6)

    try:
        res(np.ones(MAXP))
    except Exception as e:
        return {"ok": False, "error": "equation() failed on the data: %s: %s" % (type(e).__name__, str(e)[:300])}
    best = (float("inf"), np.ones(MAXP))
    for p0 in starts():
        for method in ("lm", "trf"):
            try:
                sol = least_squares(res, p0, method=method, max_nfev=3000, xtol=1e-15, ftol=1e-15, gtol=1e-15)
            except Exception:
                continue
            v = nmse(f, sol.x, X, y)
            if v < best[0]:
                best = (v, sol.x)
            break
        if best[0] < 1e-24:
            break
    out = {"ok": True, "train_nmse": best[0], "params": [float(x) for x in best[1]], "evals": {}}
    for spec in a.eval:
        name, path = spec.split("=", 1)
        Xe, ye = load(path)
        out["evals"][name] = nmse(f, best[1], Xe, ye)
    return out


if __name__ == "__main__":
    try:
        r = main()
    except Exception as e:                      # never a traceback on stdout: the caller parses JSON
        r = {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:300])}
    json.dump(r, sys.stdout)
