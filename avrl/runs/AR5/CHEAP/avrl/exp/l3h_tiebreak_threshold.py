#!/usr/bin/env python3
"""L3h: does breaking exact (agreement, n_distinct) ties by the regression's predicted corrupted-reward probability p
admit fewer corrupted rewards than the plain threshold order at 25%?

Derived from avrl/exp/l3p_scaled_greedy_loop.py (the L3p file is not edited), with the L3s subsample bootstrap (design B)
and its verdict_from rule taken from avrl/exp/l3s_subsample_centring.py. Same pool, seed permutations, 16-group steps,
selection RNG, control structure and metric definitions as L3p and L3s. Budget 0.25 only.

Arms:
  threshold  ThresholdAllocator (allocators.py, unchanged; the control). Per seed checked against the spec list and
             runs/L3s/result.json main_25.threshold before the tiebreak arm runs.
  tiebreak   TieBreakThreshold (tested arm). Refit every 20 verified groups on the six non-embedding features inside
             Pipeline(StandardScaler, LogisticRegression(max_iter=1000)), with the L3p greedy6s update cadence. Before the
             first fit: plain threshold order. After it: each 16-group step sorts by (agreement ascending, n_distinct
             descending, p descending), stable, and verifies the first k. Selection uses no RNG and no cells.

The tie-break reorders only groups that share (agreement, n_distinct) with each other, so the verified set changes only
when the cut falls inside such a block.

Design A (primary, sets U_A): full pool resampled WITH replacement, B = 100, rng default_rng(10000 + b), seeds 0-3,
750 calls per run, each resample's own error count as the denominator. U_A is the 97.5th percentile of
(tiebreak admitted minus threshold admitted).
Design B (the L3s design): half-pool subsamples WITHOUT replacement, size n // 2 = 1500, B = 200, rng default_rng(10000 + b),
seeds 0-3, 375 calls per run. Percentile and basic 95% upper bounds and bias (bootstrap mean minus the seeds 0-3 full-pool
point) go through verdict_from, identical to L3s.

Verdict: see VERDICT_RULE and overall_verdict(). KILL is checked first. INVALID_CONTROL if the threshold control fails;
no bootstrap is run then.

Features: agreement, n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen (six non-embedding features).
Pool: runs/p0/feats.parquet (3000 groups, 8 rollouts each), read only. Gold (proxy_error) is used only for scoring and as
the update label of verified groups, never as a selection input.

CPU only. No GPU, no pip, no ssh. Writes result.json into --out; the runner redirects stdout to run.log.
"""
import argparse
import json
import math
import os
import sys
import time
import warnings

sys.dont_write_bytecode = True
AVRL_DIR = "/homes/rdubash/avrl_live/CHEAP/avrl"
L3S_RESULT = os.path.join(AVRL_DIR, "runs", "L3s", "result.json")
sys.path.insert(0, AVRL_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import allocators as A  # noqa: E402

STEP = 16
FEAT_COLS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
SPEC_THRESHOLD_CAUGHT_025 = [340, 341, 344, 332, 331, 328, 325, 333]
SPEC_THRESHOLD_CAUGHT_MEAN_025 = 334.25
BIAS_TOL = 0.003
B_A = 100
B_B = 200
VERDICT_RULE = (
    "Design A: U_A = 97.5th percentile of (tiebreak admitted minus threshold admitted) at 25%, over B=100 full-pool "
    "resamples WITH replacement (default_rng(10000+b)), seeds 0-3, 750 calls, per-resample error denominator. "
    "Design B (L3s design): B=200 half-pool subsamples WITHOUT replacement (1500 groups, default_rng(10000+b)), seeds 0-3, "
    "375 calls; its percentile and basic 95% upper bounds and bias go through the L3s verdict_from rule. "
    "KILL if U_A >= 0, or if design B returns KILL (either design B upper bound at or above 0). "
    "PASS if design B returns PASS (both design B upper bounds below 0 and |bias| < 0.003) and U_A < 0. "
    "Otherwise INCONCLUSIVE, which is the pre-registered default. "
    "INVALID_CONTROL if the threshold arm does not reproduce the per-seed caught counts 340, 341, 344, 332, 331, 328, 325, 333 "
    "(mean 334.25) and runs/L3s/result.json; no bootstrap is run and the tiebreak arm gives no verdict. "
    "A PASS is a candidate re-specification only: claim-2 credit needs a second-pool test (L3w, gated)."
)
ASSUMPTIONS = [
    "steps of 16 groups over a seed-permuted pool (last step 8 groups); n_verify(step_groups, 0.25, step=t) as in E03, L3p and L3s",
    "feat_vec = [agreement, n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen] (6 dims), as in L3p and L3s",
    "tiebreak: refit when the cumulative verified count crosses a multiple of 20 (both classes present), on the cumulative "
    "verified set, as Pipeline(StandardScaler, LogisticRegression(max_iter=1000)); the update is L3p's greedy6s update",
    "tiebreak select: before the first fit, the threshold order; after it, sort by (agreement asc, n_distinct desc, p desc) with a "
    "stable sort, where p is the unclipped predict_proba of the fitted pipeline; verify the first k",
    "the tiebreak top-k differs from the threshold top-k only when the cut falls inside a block of groups sharing (agreement, n_distinct)",
    "cell key is a constant 0; selection uses no RNG and no cells",
    "admitted = (errors - caught) / errors. Main table: errors = 436. Design A denominator: errors of each resample. "
    "Design B denominator: errors of each subsample",
    "permutation rng default_rng(1000+seed); selection rng default_rng(2000+seed) per run (unused by both arms)",
    "design B draws are default_rng(10000+b).choice(n, size=n//2, replace=False), the same draws as L3s; a cross-check of the "
    "threshold subsample means against runs/L3s is recorded in checks and does not gate the verdict",
    "gold (proxy_error) used only for scoring and as the update label of verified groups",
    "gating: tiebreak runs only if threshold reproduces the spec list and runs/L3s/result.json per seed",
    "diagnostic (no verdict effect): on the main run, count the steps with a fitted model where the tiebreak verified set differs from "
    "the plain threshold top-k in the same state",
]


class TieBreakThreshold(A.GreedyOnlineAllocator):
    """Threshold order with exact (agreement, n_distinct) ties broken by the regression's p. update() is L3p's GreedyScaled
    update (same cadence, same Pipeline). select() is the only new code."""
    name = "tiebreak"

    def __init__(self, refit_every=20, prior_strength=2.0):
        super().__init__(refit_every=refit_every, prior_strength=prior_strength)
        self.n_model_steps = 0  # steps with a fitted model (diagnostic only)
        self.n_changed = 0      # of those, steps where the verified set differs from the plain threshold top-k

    def update(self, verified):
        for feat_vec, label, _cell in verified:
            self.X.append(feat_vec)
            self.y.append(label)
        n = len(self.y)
        due = n >= self.refit_every and n // self.refit_every > self.last_fit_n // self.refit_every
        if due and len(set(self.y)) > 1:
            self.model = Pipeline([("scale", StandardScaler()),
                                   ("lr", LogisticRegression(max_iter=1000))]).fit(np.array(self.X), np.array(self.y))
            self.last_fit_n = n

    def select(self, group_records, k, rng):
        if k == 0:
            return []
        threshold_top = [r["idx"] for r in A.ThresholdAllocator._order(group_records)[:k]]
        if self.model is None:
            return threshold_top
        X = np.array([r["feat_vec"] for r in group_records])
        p = self.model.predict_proba(X)[:, 1]
        order = sorted(range(len(group_records)),
                       key=lambda i: (group_records[i]["agreement"], -group_records[i]["n_distinct"], -p[i]))
        chosen = [group_records[i]["idx"] for i in order[:k]]
        self.n_model_steps += 1
        self.n_changed += int(set(chosen) != set(threshold_top))
        return chosen


ARMS = {"threshold": A.ThresholdAllocator, "tiebreak": TieBreakThreshold}


def load_pool(path):
    assert len(FEAT_COLS) == 6 and not any(c.startswith("pc") for c in FEAT_COLS), FEAT_COLS
    df = pd.read_parquet(path)
    assert len(df) == 3000, f"expected 3000 groups, got {len(df)}"
    return {
        "X": df[FEAT_COLS].to_numpy(dtype=np.float64),
        "agreement": df["agreement"].to_numpy(dtype=np.float64),
        "n_distinct": df["n_distinct"].to_numpy(dtype=np.int64),
        "y": df["proxy_error"].to_numpy(dtype=np.int64),
    }


def subset(pool, ids):
    return {k: v[ids] for k, v in pool.items()}


def run_arm(pool, arm, budget, seed):
    n = len(pool["y"])
    perm = np.random.default_rng(1000 + seed).permutation(n)
    alloc = ARMS[arm]()
    sel_rng = np.random.default_rng(2000 + seed)
    caught = 0
    calls = 0
    for t in range(math.ceil(n / STEP)):
        chunk = perm[t * STEP:(t + 1) * STEP]
        k = alloc.n_verify(len(chunk), budget, step=t)
        # cheap, gold-free fields only. cell is a constant: neither arm reads cells in selection.
        records = [
            {
                "idx": j,
                "feat_vec": pool["X"][g],
                "cell": 0,
                "agreement": float(pool["agreement"][g]),
                "n_distinct": int(pool["n_distinct"][g]),
            }
            for j, g in enumerate(chunk)
        ]
        chosen = alloc.select(records, k, sel_rng)
        assert len(chosen) == k and len(set(chosen)) == k, (arm, t, k, chosen)
        verified = []
        for j in chosen:
            g = int(chunk[j])
            verified.append((pool["X"][g], int(pool["y"][g]), 0))
            caught += int(pool["y"][g])
        alloc.update(verified)
        calls += k
    assert calls == round(budget * n), (arm, seed, calls)
    errors = int(pool["y"].sum())
    return {"arm": arm, "budget": budget, "seed": seed, "errors": errors, "caught": caught,
            "admitted": (errors - caught) / errors, "calls": calls,
            "model_steps": getattr(alloc, "n_model_steps", 0), "changed_steps": getattr(alloc, "n_changed", 0)}


def arm_stats(rows):
    return {
        "caught_per_seed": [r["caught"] for r in rows],
        "caught_mean": float(np.mean([r["caught"] for r in rows])),
        "admitted_per_seed": [round(r["admitted"], 6) for r in rows],
        "admitted_mean": float(np.mean([r["admitted"] for r in rows])),
        "calls_per_seed": [r["calls"] for r in rows],
    }


def summarize(x, point):
    """Bootstrap summary. point is the seeds 0-3 full-pool paired mean. The percentile CI comes from the draws; the basic CI
    is the reflected interval [2 * point - q97.5, 2 * point - q2.5]."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {"n_resamples": int(len(x)), "mean": float(x.mean()), "bias": float(x.mean() - point),
            "sd": float(x.std(ddof=1)), "ci95_percentile": [float(lo), float(hi)],
            "ci95_basic": [float(2 * point - hi), float(2 * point - lo)],
            "n_below_zero": int((x < 0).sum())}


def verdict_from(pct_upper, basic_upper, bias):
    """L3s rule, unchanged. KILL is checked first: either upper bound at or above 0."""
    if pct_upper >= 0 or basic_upper >= 0:
        return "KILL"
    if abs(bias) < BIAS_TOL:
        return "PASS"
    return "INCONCLUSIVE"


def overall_verdict(u_a, verdict_b):
    """L3h rule (VERDICT_RULE). KILL is checked first."""
    if u_a >= 0 or verdict_b == "KILL":
        return "KILL"
    if verdict_b == "PASS":
        return "PASS"
    return "INCONCLUSIVE"


def load_json(path):
    with open(path) as f:
        return json.load(f)


class ConvCounter:
    """Counts ConvergenceWarnings per arm and phase. The warnings record is drained after each arm run,
    so every count is attributed to the arm that raised it."""

    def __init__(self, record):
        self.record = record
        self.counts = {}

    def run(self, arm, phase, fn):
        out = fn()
        key = f"{arm}/{phase}"
        self.counts[key] = self.counts.get(key, 0) + sum(
            issubclass(w.category, ConvergenceWarning) for w in self.record)
        del self.record[:]
        return out

    def get(self, arm, phase):
        return self.counts.get(f"{arm}/{phase}", 0)


def check_threshold_vs_l3s(rows, l3s):
    """Threshold must reproduce the spec list and runs/L3s/result.json per seed at 25%. Checked before the tiebreak arm runs."""
    ref = l3s["main_25"]["threshold"]
    got = [r["caught"] for r in rows]
    checks = {
        "caught_per_seed_equals_spec": got == SPEC_THRESHOLD_CAUGHT_025,
        "caught_mean_equals_spec": float(np.mean(got)) == SPEC_THRESHOLD_CAUGHT_MEAN_025,
        "caught_per_seed_equals_l3s": got == ref["caught_per_seed"],
        "admitted_per_seed_equals_l3s": [round(r["admitted"], 6) for r in rows] == ref["admitted_per_seed"],
    }
    return all(checks.values()), checks


def bootstrap(pool, budget, seeds, n_boot, replace, size, phase, log, cc):
    """Paired bootstrap over pool groups, threshold and tiebreak only. replace=True: full-pool resamples (design A, size n).
    replace=False: subsamples without replacement (design B, size n // 2). Each draw's own error count is the denominator."""
    n = len(pool["y"])
    diffs, thr_boot = [], []
    for b in range(n_boot):
        ids = np.random.default_rng(10000 + b).choice(n, size=size, replace=replace)
        if not replace:
            assert len(np.unique(ids)) == size, b
        sub = subset(pool, ids)
        thr = [cc.run("threshold", phase, lambda s=s: run_arm(sub, "threshold", budget, s)) for s in seeds]
        tie = [cc.run("tiebreak", phase, lambda s=s: run_arm(sub, "tiebreak", budget, s)) for s in seeds]
        thr_boot.append(float(np.mean([r["admitted"] for r in thr])))
        diffs.append(float(np.mean([g["admitted"] - t["admitted"] for g, t in zip(tie, thr)])))
        if (b + 1) % 20 == 0:
            log(f"{phase} {b + 1}/{n_boot} tiebreak-threshold {diffs[-1]:+.4f}")
    return diffs, thr_boot


def write_result(out, result):
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    return os.path.join(out, "result.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--boot_a", type=int, default=B_A)
    ap.add_argument("--boot_b", type=int, default=B_B)
    ap.add_argument("--boot_seeds", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.seeds == 8 and args.boot_seeds == 4 and args.boot_a == 100 and args.boot_b == 200, \
        "spec: seeds 0-7, bootstrap seeds 0-3, design A B=100, design B B=200"
    os.makedirs(args.out, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    budget = 0.25
    bs = args.boot_seeds
    seeds = list(range(args.seeds))
    pool = load_pool(args.pool)
    n = len(pool["y"])
    n_err = int(pool["y"].sum())
    m = n // 2
    l3s = load_json(L3S_RESULT)
    log(f"L3h pool {args.pool}: groups={n} proxy_errors={n_err} step={STEP} steps={math.ceil(n / STEP)} "
        f"budget={budget} verifier_calls_per_run={round(budget * n)}")
    log(f"feature columns ({len(FEAT_COLS)}): {FEAT_COLS}")
    log("tiebreak = TieBreakThreshold: Pipeline(StandardScaler, LogisticRegression(max_iter=1000)) refit_every=20; "
        "order (agreement asc, n_distinct desc, p desc), stable; no noise, no cells, no sampling")
    log("PRE-REGISTERED RULES (written into the log before the run):")
    log(f"  {VERDICT_RULE}")

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        cc = ConvCounter(wrec)

        thr_rows = [cc.run("threshold", "main", lambda s=s: run_arm(pool, "threshold", budget, s)) for s in seeds]
        thr_ok, thr_checks = check_threshold_vs_l3s(thr_rows, l3s)
        log(f"threshold vs spec and runs/L3s per seed: {'PASS' if thr_ok else 'FAIL'}; caught {[r['caught'] for r in thr_rows]}")

        if not thr_ok:
            verdict = "INVALID_CONTROL"
            log("INVALID_CONTROL: the threshold control did not reproduce. The tiebreak arm is not run and no tiebreak number is read.")
            result = {"id": "L3h", "pool": args.pool, "checks": {"threshold_vs_spec_and_l3s": thr_checks, "controls_pass": False},
                      "main_25": {"threshold": arm_stats(thr_rows)}, "verdict_rule": VERDICT_RULE, "verdict": verdict,
                      "metric": None, "convergence_warnings": {"threshold": {"main": cc.get("threshold", "main")}},
                      "wall_s": round(time.time() - t0, 1)}
            write_result(args.out, result)
            log(f"verdict {verdict}; wrote {os.path.join(args.out, 'result.json')}")
            return 3

        tie_rows = [cc.run("tiebreak", "main", lambda s=s: run_arm(pool, "tiebreak", budget, s)) for s in seeds]
        assert [r["calls"] for r in tie_rows] == [r["calls"] for r in thr_rows], "equal calls across arms"
        st_t, st_b = arm_stats(thr_rows), arm_stats(tie_rows)
        diag = {"model_steps_per_seed": [r["model_steps"] for r in tie_rows],
                "changed_steps_per_seed": [r["changed_steps"] for r in tie_rows],
                "changed_steps_total_seeds_0_to_7": int(sum(r["changed_steps"] for r in tie_rows)),
                "steps_per_run": math.ceil(n / STEP)}
        log(f"  tiebreak caught per seed {st_b['caught_per_seed']} mean {st_b['caught_mean']:.2f} "
            f"admitted mean {st_b['admitted_mean']:.4f}")
        log(f"diagnostic (no verdict effect): steps with a fitted model {diag['model_steps_per_seed']}; "
            f"steps where the tiebreak changed the verified set {diag['changed_steps_per_seed']}")
        log(f"ConvergenceWarnings main: tiebreak {cc.get('tiebreak', 'main')}, threshold {cc.get('threshold', 'main')}")

        main_diff = [b["admitted"] - t["admitted"] for b, t in zip(tie_rows, thr_rows)]
        main_mean_8 = float(np.mean(main_diff))
        point = float(np.mean(main_diff[:bs]))
        log(f"paired (tiebreak - threshold) admitted: seeds 0-7 mean {main_mean_8:+.4f}; seeds 0-{bs - 1} point {point:+.4f}")

        log(f"\nDesign A: B={args.boot_a} full-pool resamples WITH replacement (default_rng(10000+b)), seeds 0-{bs - 1}, 750 calls")
        diffs_a, _ = bootstrap(pool, budget, list(range(bs)), args.boot_a, True, n, "bootA", log, cc)
        log(f"\nDesign B: B={args.boot_b} subsamples of {m} groups WITHOUT replacement (default_rng(10000+b)), "
            f"seeds 0-{bs - 1}, 375 calls")
        diffs_b, thr_b = bootstrap(pool, budget, list(range(bs)), args.boot_b, False, m, "bootB", log, cc)

    sA = summarize(diffs_a, point)
    sB = summarize(diffs_b, point)
    U_A = sA["ci95_percentile"][1]
    verdict_b = verdict_from(sB["ci95_percentile"][1], sB["ci95_basic"][1], sB["bias"])
    verdict = overall_verdict(U_A, verdict_b)
    draws_match = bool(np.allclose(thr_b, l3s["metric"]["threshold_subsample_mean_seeds_0_to_3"], rtol=0, atol=1e-12))
    st_s_lo, st_s_hi = sB["ci95_percentile"]
    log(f"design A: U_A (97.5th percentile, per-resample denominator) {U_A:+.4f}; "
        f"95% percentile CI [{sA['ci95_percentile'][0]:+.4f}, {U_A:+.4f}]; resamples below 0: {sA['n_below_zero']}/{args.boot_a}")
    log(f"design B: percentile CI [{st_s_lo:+.4f}, {st_s_hi:+.4f}]; basic CI [{sB['ci95_basic'][0]:+.4f}, {sB['ci95_basic'][1]:+.4f}]; "
        f"bias {sB['bias']:+.4f} (tolerance {BIAS_TOL}); verdict_from: {verdict_b}")
    log(f"design B threshold draws match runs/L3s threshold subsample means: {draws_match}")
    log(f"caught mean seeds 0-7: tiebreak {st_b['caught_mean']:.2f}, threshold {st_t['caught_mean']:.2f}")
    log(f"overall L3h verdict: {verdict}")

    result = {
        "id": "L3h",
        "pool": args.pool,
        "n_groups": n,
        "n_proxy_errors": n_err,
        "step_size_groups": STEP,
        "budget": budget,
        "verifier_calls_per_run": round(budget * n),
        "verifier_calls_per_subsample_run": round(budget * m),
        "feature_cols": FEAT_COLS,
        "seeds_main": args.seeds,
        "seeds_bootstrap": bs,
        "B_design_A": args.boot_a,
        "B_design_B": args.boot_b,
        "design_B_subsample_size": m,
        "assumptions": ASSUMPTIONS,
        "checks": {
            "threshold_vs_spec_and_l3s": thr_checks,
            "controls_pass": True,
            "verifier_calls_equal_across_arms": True,
            "design_B_threshold_draws_match_l3s": draws_match,
        },
        "main_25": {"threshold": st_t, "tiebreak": st_b},
        "tiebreak_diagnostic": diag,
        "convergence_warnings": {arm: {phase: cc.get(arm, phase) for phase in ("main", "bootA", "bootB")}
                                 for arm in ("threshold", "tiebreak")},
        "metric": {
            "name": "admitted rate at 25% for tiebreak minus threshold, paired over seeds 0-7 at 750 calls "
                    "(main_mean_admitted_diff_seeds_0_to_7); decision statistics: design A U_A (97.5th percentile, with "
                    "replacement, B=100, seeds 0-3, per-resample denominator) and design B percentile and basic upper bounds "
                    "and bias (without replacement, 1500-group subsamples, B=200, seeds 0-3, per-subsample denominator, "
                    "verdict_from as in L3s)",
            "budget": budget,
            "main_mean_admitted_diff_seeds_0_to_7": main_mean_8,
            "point_estimate_seeds_0_to_3": point,
            "U_A": U_A,
            "design_A": {"B": args.boot_a, "replacement": True, "denominator": "per-resample errors", "seeds": bs,
                         "per_resample_diff": diffs_a, **sA},
            "design_B": {"B": args.boot_b, "subsample_size": m, "replacement": False,
                         "denominator": "per-subsample errors", "seeds": bs, "verdict_from": verdict_b,
                         "per_subsample_diff": diffs_b, "threshold_subsample_mean_seeds_0_to_3": thr_b, **sB},
            "bias_tolerance": BIAS_TOL,
            "verdict_rule": VERDICT_RULE,
        },
        "verdict_rule": VERDICT_RULE,
        "verdict": verdict,
        "wall_s": round(time.time() - t0, 1),
    }
    path = write_result(args.out, result)
    log(f"\nwrote {path} (wall {result['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
