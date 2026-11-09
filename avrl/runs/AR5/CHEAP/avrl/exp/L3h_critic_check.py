#!/usr/bin/env python3
"""L3h critic check: independent re-derivation of the L3h arms, bootstrap and verdict.

Does not import allocators.py. The arms are re-implemented from the ledger text: threshold order (agreement ascending,
n_distinct descending, ties in chunk order, implemented with np.lexsort), cumulative rounding at 25 percent, the tiebreak
refit (StandardScaler + LogisticRegression(max_iter=1000) when the verified count crosses a multiple of 20 with both classes
present), and the tiebreak rank (agreement asc, n_distinct desc, p desc, ties in chunk order). Then:
  (1) stored per-seed caught counts for both arms (seeds 0-7) and the diagnostic changed-step counts;
  (2) design A (full-pool resamples with replacement, B = 100) and design B (1500-group subsamples without replacement,
      B = 200), seeds 0-3, compared draw by draw with runs/L3h/result.json;
  (3) the pre-registered verdict, recomputed from the draws;
  (4) Monte Carlo check of the 97.5th percentile and basic upper bound over 5000 resamples of the stored draws;
  (5) gold-freeness of the features, and a leakage test on the ORIGINAL exp/l3h_tiebreak_threshold.py: flip the
      proxy_error label of every group that the baseline tiebreak run never verified; caught counts must not change;
  (6) reproduction of the main 8-seed caught counts with the original script's run_arm;
  (7) every warning category raised during the arm runs (the original ConvCounter counts ConvergenceWarning only).

CPU only. Read-only on the pool and on runs/L3h/result.json. Writes runs/L3h/critic_check.txt.
"""
import importlib.util
import json
import math
import sys
import warnings
from collections import Counter

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

AVRL = "/homes/rdubash/avrl_live/CHEAP/avrl"
POOL = f"{AVRL}/runs/p0/feats.parquet"
RES = f"{AVRL}/runs/L3h/result.json"
L3S = f"{AVRL}/runs/L3s/result.json"
ORIG = f"{AVRL}/exp/l3h_tiebreak_threshold.py"
OUT = f"{AVRL}/runs/L3h/critic_check.txt"
FEATS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
STEP = 16
BUDGET = 0.25
REFIT = 20
SPEC_CAUGHT = [340, 341, 344, 332, 331, 328, 325, 333]
SEEDS_MAIN = list(range(8))
SEEDS_BOOT = list(range(4))
B_A, B_B = 100, 200
TOL = 0.003

lines = []


def say(msg=""):
    print(msg, flush=True)
    lines.append(msg)


def arm_run(X, agr, nd, y, arm, seed):
    """Independent arm implementation. X, agr, nd, y are the (sub)pool arrays. Returns a dict."""
    n = len(y)
    perm = np.random.default_rng(1000 + seed).permutation(n)
    Xv, yv = [], []
    model = None
    last_fit = 0
    caught = calls = changed = fitted = 0
    chosen_seq = []
    for t in range(math.ceil(n / STEP)):
        chunk = perm[t * STEP:(t + 1) * STEP]
        m = len(chunk)
        q = BUDGET * m
        k = int(math.floor(q * (t + 1)) - math.floor(q * t))
        if k > 0:
            a_c, n_c = agr[chunk], nd[chunk]
            thr_pos = np.lexsort((np.arange(m), -n_c, a_c))[:k]
            if arm == "tiebreak" and model is not None:
                p = model.predict_proba(X[chunk])[:, 1]
                pos = np.lexsort((np.arange(m), -p, -n_c, a_c))[:k]
                fitted += 1
                changed += int(set(pos.tolist()) != set(thr_pos.tolist()))
            else:
                pos = thr_pos
            chosen = chunk[pos]
            chosen_seq.append(tuple(sorted(int(g) for g in chosen)))
            for g in chosen:
                caught += int(y[g])
                if arm == "tiebreak":
                    Xv.append(X[g])
                    yv.append(int(y[g]))
            calls += k
        if arm == "tiebreak":
            nn = len(yv)
            due = nn >= REFIT and nn // REFIT > last_fit // REFIT
            if due and len(set(yv)) > 1:
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(
                    np.array(Xv), np.array(yv))
                last_fit = nn
    assert calls == round(BUDGET * n), (arm, seed, calls)
    errors = int(np.sum(y))
    return {"caught": caught, "calls": calls, "changed": changed, "fitted": fitted,
            "admitted": (errors - caught) / errors, "errors": errors, "chosen": chosen_seq}


def main():
    res = json.load(open(RES))
    l3s = json.load(open(L3S))
    orig = None
    spec = importlib.util.spec_from_file_location("l3h_orig", ORIG)
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)

    df = pd.read_parquet(POOL)
    X = df[FEATS].to_numpy(dtype=np.float64)
    agr = df["agreement"].to_numpy(dtype=np.float64)
    nd = df["n_distinct"].to_numpy(dtype=np.int64)
    y = df["proxy_error"].to_numpy(dtype=np.int64)
    n = len(y)
    say(f"pool groups={n} proxy_errors={int(y.sum())} steps={math.ceil(n / STEP)} calls_per_run={round(BUDGET * n)}")

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")

        # (7) and (1): main 8 seeds, both arms, independent implementation
        main_thr = [arm_run(X, agr, nd, y, "threshold", s) for s in SEEDS_MAIN]
        main_tie = [arm_run(X, agr, nd, y, "tiebreak", s) for s in SEEDS_MAIN]
        caught_thr = [r["caught"] for r in main_thr]
        caught_tie = [r["caught"] for r in main_tie]
        say("(1) per-seed caught, independent re-derivation")
        say(f"  threshold {caught_thr} mean {np.mean(caught_thr):.3f}")
        say(f"  tiebreak  {caught_tie} mean {np.mean(caught_tie):.3f}")
        say(f"  threshold equals spec list: {caught_thr == SPEC_CAUGHT}")
        say(f"  threshold equals runs/L3s per seed: {caught_thr == l3s['main_25']['threshold']['caught_per_seed']}")
        say(f"  tiebreak equals stored result.json per seed: {caught_tie == res['main_25']['tiebreak']['caught_per_seed']}")
        changed = [r["changed"] for r in main_tie]
        fitted = [r["fitted"] for r in main_tie]
        say(f"  fitted steps {fitted} (total {sum(fitted)}); changed steps {changed} (total {sum(changed)}, "
            f"{100 * sum(changed) / sum(fitted):.1f} percent)")
        say(f"  stored changed steps {res['tiebreak_diagnostic']['changed_steps_per_seed']}: "
            f"{changed == res['tiebreak_diagnostic']['changed_steps_per_seed']}")
        paired_main = np.array([t["admitted"] - h["admitted"] for t, h in zip(main_tie, main_thr)])
        point = float(np.mean(paired_main[:4]))
        say(f"  paired admitted diff seeds 0-7 mean {paired_main.mean():+.6f} (stored "
            f"{res['metric']['main_mean_admitted_diff_seeds_0_to_7']:+.6f}); seeds 0-3 point {point:+.6f} "
            f"(stored {res['metric']['point_estimate_seeds_0_to_3']:+.6f})")
        say(f"  caught mean seeds 0-7: tiebreak {np.mean(caught_tie):.3f} (run.log prints 335.12, ledger table 335.13)")

        # (2): design A and design B draws, seeds 0-3
        dA, dB = [], []
        for b in range(B_A):
            ids = np.random.default_rng(10000 + b).choice(n, size=n, replace=True)
            Xs, as_, ns, ys = X[ids], agr[ids], nd[ids], y[ids]
            d = [arm_run(Xs, as_, ns, ys, "tiebreak", s)["admitted"] - arm_run(Xs, as_, ns, ys, "threshold", s)["admitted"]
                 for s in SEEDS_BOOT]
            dA.append(float(np.mean(d)))
        for b in range(B_B):
            ids = np.random.default_rng(10000 + b).choice(n, size=n // 2, replace=False)
            assert len(np.unique(ids)) == n // 2
            Xs, as_, ns, ys = X[ids], agr[ids], nd[ids], y[ids]
            d = [arm_run(Xs, as_, ns, ys, "tiebreak", s)["admitted"] - arm_run(Xs, as_, ns, ys, "threshold", s)["admitted"]
                 for s in SEEDS_BOOT]
            dB.append(float(np.mean(d)))
        dA, dB = np.array(dA), np.array(dB)
        stA = res["metric"]["design_A"]["per_resample_diff"]
        stB = res["metric"]["design_B"]["per_subsample_diff"]
        say("(2) bootstrap draws, independent vs stored")
        say(f"  design A max |diff| {np.max(np.abs(dA - np.array(stA))):.3e} over {len(dA)} draws")
        say(f"  design B max |diff| {np.max(np.abs(dB - np.array(stB))):.3e} over {len(dB)} draws")

        # (3): verdict from the independent draws
        U_A = float(np.percentile(dA, 97.5))
        pctB = float(np.percentile(dB, 97.5))
        basicB = float(2 * point - np.percentile(dB, 2.5))
        biasB = float(dB.mean() - point)
        if pctB >= 0 or basicB >= 0:
            vB = "KILL"
        elif abs(biasB) < TOL:
            vB = "PASS"
        else:
            vB = "INCONCLUSIVE"
        if U_A >= 0 or vB == "KILL":
            verdict = "KILL"
        elif vB == "PASS":
            verdict = "PASS"
        else:
            verdict = "INCONCLUSIVE"
        say("(3) verdict from independent draws")
        say(f"  U_A {U_A:+.6f} (stored {res['metric']['U_A']:+.6f}); resamples below 0 {int((dA < 0).sum())}/{B_A} "
            f"(stored {res['metric']['design_A']['n_below_zero']})")
        say(f"  design B percentile upper {pctB:+.6f} (stored {res['metric']['design_B']['ci95_percentile'][1]:+.6f}); "
            f"basic upper {basicB:+.6f} (stored {res['metric']['design_B']['ci95_basic'][1]:+.6f}); "
            f"bias {biasB:+.6f} (stored {res['metric']['design_B']['bias']:+.6f}); design B verdict {vB}")
        say(f"  design B SD {dB.std(ddof=1):.6f} (stored {res['metric']['design_B']['sd']:.6f}); "
            f"design B below 0 {int((dB < 0).sum())}/{B_B} (stored {res['metric']['design_B']['n_below_zero']})")
        say(f"  overall verdict {verdict}; stored verdict {res['verdict']}; match {verdict == res['verdict']}")

        # (4): Monte Carlo over the stored draws
        rng = np.random.default_rng(20261019)
        mcA = np.array([np.percentile(rng.choice(dA, size=B_A, replace=True), 97.5) for _ in range(5000)])
        mcB = []
        mcBasic = []
        for _ in range(5000):
            r = rng.choice(dB, size=B_B, replace=True)
            mcB.append(np.percentile(r, 97.5))
            mcBasic.append(2 * point - np.percentile(r, 2.5))
        mcB, mcBasic = np.array(mcB), np.array(mcBasic)
        say("(4) Monte Carlo over stored draws (5000 resamples each; measures B noise only)")
        say(f"  design A 97.5th pct: mean {mcA.mean():+.6f} SD {mcA.std(ddof=1):.6f} min {mcA.min():+.6f} "
            f"max {mcA.max():+.6f}; all >= 0: {bool((mcA >= 0).all())}")
        say(f"  design B 97.5th pct: mean {mcB.mean():+.6f} SD {mcB.std(ddof=1):.6f} min {mcB.min():+.6f}; "
            f"all >= 0: {bool((mcB >= 0).all())}")
        say(f"  design B basic upper: mean {mcBasic.mean():+.6f} SD {mcBasic.std(ddof=1):.6f} min {mcBasic.min():+.6f}; "
            f"all >= 0: {bool((mcBasic >= 0).all())}; resamples below 0: {int((mcBasic < 0).sum())}/5000")
        say(f"  design B percentile upper resamples below 0: {int((mcB < 0).sum())}/5000; "
            f"design A percentile upper resamples below 0: {int((mcA < 0).sum())}/5000")
        say(f"  SD of draws: design A {dA.std(ddof=1):.6f}, design B {dB.std(ddof=1):.6f}")

        # (5): gold-freeness and leakage test on the original script
        say("(5) gold-freeness and leakage")
        say(f"  log_qlen equals log1p(q_len): {bool(np.allclose(df['log_qlen'], np.log1p(df['q_len'])))}")
        for c in FEATS:
            say(f"  feature {c} equals proxy_error: {bool(np.array_equal(df[c].to_numpy(), y))}")
        verified_union = [set() for _ in SEEDS_MAIN]
        for s, r in zip(SEEDS_MAIN, main_tie):
            for seq in r["chosen"]:
                verified_union[s].update(seq)
        flip_ok = []
        for s in SEEDS_MAIN:
            never = np.ones(n, dtype=bool)
            never[list(verified_union[s])] = False
            y2 = y.copy()
            y2[never] = 1 - y2[never]
            pool2 = {"X": X, "agreement": agr, "n_distinct": nd, "y": y2}
            pool1 = {"X": X, "agreement": agr, "n_distinct": nd, "y": y}
            base = orig.run_arm(pool1, "tiebreak", BUDGET, s)
            flipped = orig.run_arm(pool2, "tiebreak", BUDGET, s)
            flip_ok.append(base["caught"] == flipped["caught"] and base["calls"] == flipped["calls"])
        say(f"  original script, tiebreak: flipping labels of never-verified groups leaves caught and calls unchanged "
            f"on all 8 seeds: {all(flip_ok)} (never-verified groups per seed: "
            f"{[int(n - len(v)) for v in verified_union]})")

        # (6): reproduction of the stored main run with the original script
        orig_thr = [orig.run_arm({"X": X, "agreement": agr, "n_distinct": nd, "y": y}, "threshold", BUDGET, s)
                    for s in SEEDS_MAIN]
        orig_tie = [orig.run_arm({"X": X, "agreement": agr, "n_distinct": nd, "y": y}, "tiebreak", BUDGET, s)
                    for s in SEEDS_MAIN]
        say("(6) original script reproduces the stored main run")
        say(f"  threshold caught {[r['caught'] for r in orig_thr]} equals independent: "
            f"{[r['caught'] for r in orig_thr] == caught_thr}")
        say(f"  tiebreak caught {[r['caught'] for r in orig_tie]} equals independent: "
            f"{[r['caught'] for r in orig_tie] == caught_tie}")

    counts = Counter(w.category.__name__ for w in wrec)
    say("(7) warnings raised during this check, by category:")
    say(f"  {dict(counts) if counts else 'none'}")
    say(f"  ConvergenceWarning count: {counts.get('ConvergenceWarning', 0)}")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    say(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
