#!/usr/bin/env python3
"""L3v: null calibration of the L3s design.

Copied from avrl/exp/l3s_subsample_centring.py (the L3s file is not edited). Same loop, arms, seeds, bootstrap and
verdict_from as L3s. Two changes only:
  1. Pool: a label-independent null pool, runs/L3v/pools/null_k.parquet. Its six feature columns are permuted together
     against proxy_error, so no feature carries label information. Built by avrl/exp/l3v_build_null_pools.py.
  2. Controls: the real-pool gates (E03, L3i and L3p result files, and the SPEC_* lists) cannot apply to a null pool, so
     they are replaced by same-pool determinism. Each arm is rerun once per seed 0-7 on the same pool and must reproduce
     its per-seed caught counts. A failure gives INVALID_CONTROL and no bootstrap, as in L3s.

Arms: threshold (ThresholdAllocator), greedy6u (GreedyOnlineAllocator, unscaled), greedy6s (GreedyScaled, as in L3s).
Budget 0.25 only. Bootstrap: B = 200 half-pool subsamples WITHOUT replacement (size n // 2, rng default_rng(10000 + b)),
seeds 0-3, budget 0.25, as in L3s. Gold (proxy_error) is used only for scoring and as the update label of verified
groups, never as a selection input.

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
BIAS_TOL = 0.003
VERDICT_RULE = (
    "PASS iff the percentile 95% upper bound AND the basic 95% upper bound of (greedy6s admitted minus threshold admitted) "
    "at 25% are both below 0, AND |bias| < 0.003 (bias = subsample bootstrap mean minus the seeds 0-3 full-pool point "
    "estimate; B=200 half-pool subsamples WITHOUT replacement, seeds 0-3). "
    "KILL iff either upper bound is 0 or above. "
    "UNASSIGNED CASE (both upper bounds below 0 and |bias| >= 0.003): the proposed default, recorded before the run, is "
    "INCONCLUSIVE: no claim-2 credit, L3r stays gated pending a critic ruling. "
    "INVALID_CONTROL if any arm fails same-pool determinism (a rerun on the same pool and seed does not reproduce the main "
    "run's caught count), in which case no bootstrap is run. "
    "L3v: this rule is applied to each null pool; the aggregate is in avrl/exp/l3v_aggregate.py."
)
ASSUMPTIONS = [
    "steps of 16 groups over a seed-permuted pool (3000 groups: 188 steps, last step 8 groups; 1500-group subsample: 94 steps, "
    "last step 12 groups); n_verify(step_groups, 0.25, step=t) as in E03 and L3i",
    "feat_vec = [agreement, n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen] (6 dims), as in L3i and L3p",
    "greedy6u = allocators.GreedyOnlineAllocator unchanged (control): unscaled LogisticRegression(max_iter=1000), refit_every 20, prior_strength 2.0",
    "greedy6s = GreedyScaled, as in L3p: the cumulative verified set is refit as Pipeline(StandardScaler, LogisticRegression(max_iter=1000)) "
    "on the same cadence; select() inherited, so the cold start before the first fit is the threshold order",
    "cell key is a constant 0 in every record and verified tuple; greedy selection never reads cells",
    "admitted = (errors - caught) / errors, with the errors of the run's own pool (main: 436; each subsample: its own count). "
    "No fixed-denominator variant: 436 is not a valid denominator for a half-pool",
    "permutation rng default_rng(1000+seed); selection rng default_rng(2000+seed) per run, as in L3p",
    "bootstrap: subsample of n // 2 = 1500 groups WITHOUT replacement per draw, rng default_rng(10000+b), B=200; "
    "threshold and greedy6s only, seeds 0-3, budget 0.25",
    "bias = mean of the B paired bootstrap diffs minus the seeds 0-3 full-pool point estimate; subsample SD = sample SD (ddof=1) of the B diffs",
    "percentile CI = 2.5 and 97.5 percentiles of the B diffs; basic CI = [2 * point - q97.5, 2 * point - q2.5]",
    "gold (proxy_error) used only for scoring and as the update label of verified groups",
    "gating: same-pool determinism. Each arm is rerun once per seed 0-7 on the same null pool; the rerun must reproduce the main "
    "run's per-seed caught counts. All checked before the bootstrap is read",
    "null pool (runs/L3v/pools/null_k.parquet): the six feature columns are permuted together by default_rng(20000+k).permutation(3000); "
    "proxy_error and cell100 stay in place, so the labels are unchanged (436 errors) and no feature carries label information",
    "caveat: a half-pool subsample changes the loop itself (94 steps and 375 verified labels per run, versus 188 steps and 750 on the "
    "full pool), so the bias term can reflect learning-curve shrinkage as well as resampling. The threshold-only subsample mean is "
    "reported as a reference, because threshold fits no model",
]


class GreedyScaled(A.GreedyOnlineAllocator):
    """GreedyOnlineAllocator with the regression fit inside a StandardScaler pipeline. Only update() differs."""
    name = "greedy6s"

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


ARMS = {"threshold": A.ThresholdAllocator, "greedy6u": A.GreedyOnlineAllocator, "greedy6s": GreedyScaled}


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
        # cheap, gold-free fields only. cell is a constant: greedy selection never reads cells, and the key
        # only satisfies the (feat_vec, label, cell) update signature.
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
            "admitted": (errors - caught) / errors, "calls": calls}


def arm_stats(rows):
    return {
        "caught_per_seed": [r["caught"] for r in rows],
        "caught_mean": float(np.mean([r["caught"] for r in rows])),
        "admitted_per_seed": [round(r["admitted"], 6) for r in rows],
        "admitted_mean": float(np.mean([r["admitted"] for r in rows])),
        "calls_per_seed": [r["calls"] for r in rows],
    }


def summarize(x, point):
    """Subsample bootstrap summary. point is the seeds 0-3 full-pool paired mean. The percentile CI comes from the draws;
    the basic CI is the reflected interval [2 * point - q97.5, 2 * point - q2.5]."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {"mean": float(x.mean()), "bias": float(x.mean() - point), "sd": float(x.std(ddof=1)),
            "ci95_percentile": [float(lo), float(hi)],
            "ci95_basic": [float(2 * point - hi), float(2 * point - lo)],
            "n_below_zero": int((x < 0).sum())}


def verdict_from(pct_upper, basic_upper, bias):
    """Pre-registered rule (VERDICT_RULE). KILL is checked first: either upper bound at or above 0."""
    if pct_upper >= 0 or basic_upper >= 0:
        return "KILL"
    if abs(bias) < BIAS_TOL:
        return "PASS"
    return "INCONCLUSIVE"


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


def check_determinism(main_rows, rerun_rows):
    """Same-pool determinism, replacing the real-pool gates (which cannot apply to a null pool). A rerun of an arm on the
    same pool and seed must reproduce its per-seed caught counts."""
    got = [r["caught"] for r in main_rows]
    again = [r["caught"] for r in rerun_rows]
    return got == again, {"main_caught_per_seed": got, "rerun_caught_per_seed": again, "rerun_equals_main": got == again}


def bootstrap(pool, budget, seeds, n_boot, log, cc):
    """Subsample bootstrap over pool groups, threshold and greedy6s only. Each draw is a half-pool subsample WITHOUT
    replacement, so no group appears twice. diffs uses each subsample's own error count as the admitted denominator."""
    n = len(pool["y"])
    m = n // 2
    diffs, thr_boot = [], []
    for b in range(n_boot):
        ids = np.random.default_rng(10000 + b).choice(n, size=m, replace=False)
        assert len(np.unique(ids)) == m, b
        sub = subset(pool, ids)
        thr = [cc.run("threshold", "bootstrap", lambda s=s: run_arm(sub, "threshold", budget, s)) for s in seeds]
        grd = [cc.run("greedy6s", "bootstrap", lambda s=s: run_arm(sub, "greedy6s", budget, s)) for s in seeds]
        thr_boot.append(float(np.mean([r["admitted"] for r in thr])))
        diffs.append(float(np.mean([g["admitted"] - t["admitted"] for g, t in zip(grd, thr)])))
        if (b + 1) % 20 == 0:
            log(f"bootstrap {b + 1}/{n_boot} greedy6s-threshold {diffs[-1]:+.4f}")
    return diffs, thr_boot


def write_result(out, result):
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    return os.path.join(out, "result.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.25])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--boot_budget", type=float, default=0.25)
    ap.add_argument("--boot_seeds", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.budgets == [0.25] and args.boot_budget == 0.25, "L3s is defined at budget 0.25 only"
    assert args.seeds == 8 and args.boot_seeds == 4 and args.boot == 200, "spec: seeds 0-7, bootstrap seeds 0-3, B=200"
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
    log(f"L3v null pool {args.pool}: groups={n} proxy_errors={n_err} step={STEP} steps={math.ceil(n / STEP)} "
        f"budget={budget} verifier_calls_per_run={round(budget * n)}")
    log(f"bootstrap draws: B={args.boot} subsamples of {m} groups WITHOUT replacement (default_rng(10000+b)), "
        f"each run at 25% = {round(budget * m)} calls per arm")
    log(f"feature columns ({len(FEAT_COLS)}): {FEAT_COLS}")
    log("greedy6s = GreedyScaled: Pipeline(StandardScaler, LogisticRegression(max_iter=1000)) refit_every=20 prior_strength=2.0; "
        "no noise, no cells in selection")
    log("arms: threshold, greedy6u (unscaled), greedy6s; each arm is rerun once per seed for same-pool determinism")
    log("PRE-REGISTERED RULES (L3v pre-registration in avrl/RESEARCH_LOG.md, written before launch; VERDICT_RULE as in L3s):")
    log(f"  {VERDICT_RULE}")

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        cc = ConvCounter(wrec)

        rows, rerun = {}, {}
        for arm in ARMS:
            rows[arm] = [cc.run(arm, "main", lambda s=s: run_arm(pool, arm, budget, s)) for s in seeds]
            rerun[arm] = [cc.run(arm, "determinism", lambda s=s: run_arm(pool, arm, budget, s)) for s in seeds]
        det_checks, controls_ok = {}, True
        for arm in ARMS:
            ok, det_checks[arm] = check_determinism(rows[arm], rerun[arm])
            controls_ok = controls_ok and ok
            log(f"{arm} same-pool determinism per seed: {'PASS' if ok else 'FAIL'}; caught {[r['caught'] for r in rows[arm]]}")
        thr_rows, g6u_rows, g6s_rows = rows["threshold"], rows["greedy6u"], rows["greedy6s"]

        main25 = {arm: arm_stats(rows[arm]) for arm in ARMS}
        result = {
            "id": "L3v",
            "pool": args.pool,
            "null_pool": os.path.basename(args.pool),
            "n_groups": n,
            "n_proxy_errors": n_err,
            "step_size_groups": STEP,
            "budget": budget,
            "verifier_calls_per_run": round(budget * n),
            "feature_cols": FEAT_COLS,
            "seeds_main": args.seeds,
            "seeds_bootstrap": bs,
            "B": args.boot,
            "bootstrap_subsample_size": m,
            "assumptions": ASSUMPTIONS,
            "checks": {
                "same_pool_determinism": det_checks,
                "controls_pass": controls_ok,
            },
            "main_25": main25,
            "verdict_rule": VERDICT_RULE,
        }

        if not controls_ok:
            verdict = "INVALID_CONTROL"
            log("INVALID_CONTROL: a same-pool determinism rerun did not reproduce. The bootstrap is not run and no bootstrap number is read.")
            result.update({"verdict": verdict, "metric": None,
                           "convergence_warnings": {
                               "threshold": {"main": cc.get("threshold", "main")},
                               "greedy6u": {"main": cc.get("greedy6u", "main")},
                               "greedy6s": {"main": cc.get("greedy6s", "main")}}})
            result["wall_s"] = round(time.time() - t0, 1)
            write_result(args.out, result)
            log(f"verdict {verdict}; wrote {os.path.join(args.out, 'result.json')}")
            return 3

        st_t, st_u, st_s = arm_stats(thr_rows), arm_stats(g6u_rows), arm_stats(g6s_rows)
        log(f"  greedy6s caught per seed {st_s['caught_per_seed']} mean {st_s['caught_mean']:.2f} "
            f"admitted mean {st_s['admitted_mean']:.4f}")
        log(f"ConvergenceWarnings main: greedy6s {cc.get('greedy6s', 'main')}, greedy6u {cc.get('greedy6u', 'main')}, "
            f"threshold {cc.get('threshold', 'main')}")

        main_diff = [g["admitted"] - t["admitted"] for g, t in zip(g6s_rows, thr_rows)]
        main_mean_8 = float(np.mean(main_diff))
        point = float(np.mean(main_diff[:bs]))
        thr_full = float(np.mean([r["admitted"] for r in thr_rows[:bs]]))
        log(f"paired (greedy6s - threshold) admitted: seeds 0-7 mean {main_mean_8:+.4f}; seeds 0-{bs - 1} point {point:+.4f}")

        log(f"\nbootstrap: B={args.boot} subsamples of {m} groups without replacement (default_rng(10000+b)), "
            f"seeds 0-{bs - 1}, budget {budget:.2f}")
        diffs, thr_boot = bootstrap(pool, budget, list(range(bs)), args.boot, log, cc)

    s = summarize(diffs, point)
    verdict = verdict_from(s["ci95_percentile"][1], s["ci95_basic"][1], s["bias"])
    lo_p, hi_p = s["ci95_percentile"]
    lo_b, hi_b = s["ci95_basic"]
    thr_boot_mean = float(np.mean(thr_boot))
    log(f"CI95 percentile (greedy6s - threshold) admitted at 0.25: [{lo_p:+.4f}, {hi_p:+.4f}]")
    log(f"CI95 basic (2 x point minus percentile bounds): [{lo_b:+.4f}, {hi_b:+.4f}]")
    log(f"bootstrap mean {s['mean']:+.4f}; point {point:+.4f}; bias {s['bias']:+.4f} (tolerance |bias| < {BIAS_TOL})")
    log(f"subsample SD {s['sd']:.4f} (ddof=1 over B={args.boot}); resamples below 0: {s['n_below_zero']}/{args.boot}")
    log(f"threshold-only reference (no model fit): subsample mean {thr_boot_mean:.4f} vs full-pool seeds 0-{bs - 1} "
        f"{thr_full:.4f}, shift {thr_boot_mean - thr_full:+.4f}")
    log(f"ConvergenceWarnings bootstrap: greedy6s {cc.get('greedy6s', 'bootstrap')}, "
        f"threshold {cc.get('threshold', 'bootstrap')}")
    log(f"caught mean seeds 0-7: greedy6s {st_s['caught_mean']:.2f}, greedy6u {st_u['caught_mean']:.2f}, "
        f"threshold {st_t['caught_mean']:.2f}")
    log(f"overall L3v per-pool verdict on {os.path.basename(args.pool)}: {verdict}")
    if verdict == "PASS":
        log("PASS on this null pool: both upper bounds below 0 and |bias| < 0.003. Counted as a null PASS (false positive).")
    elif verdict == "KILL":
        log("KILL on this null pool: an upper bound is 0 or above (the usual null outcome).")
    else:
        log("INCONCLUSIVE on this null pool: both upper bounds below 0 but |bias| >= 0.003.")

    result.update({
        "convergence_warnings": {
            "greedy6s": {"main": cc.get("greedy6s", "main"), "bootstrap": cc.get("greedy6s", "bootstrap"),
                         "total": cc.get("greedy6s", "main") + cc.get("greedy6s", "bootstrap")},
            "greedy6u": {"main": cc.get("greedy6u", "main"), "bootstrap": 0,
                         "note": "greedy6u is not run in the bootstrap"},
            "threshold": {"main": cc.get("threshold", "main"), "bootstrap": cc.get("threshold", "bootstrap"),
                          "note": "threshold fits no model"},
        },
        "caught_mean_seeds_0_to_7": {
            "threshold": st_t["caught_mean"],
            "greedy6u": st_u["caught_mean"],
            "greedy6s": st_s["caught_mean"],
            "greedy6s_minus_threshold": st_s["caught_mean"] - st_t["caught_mean"],
            "greedy6s_minus_greedy6u": st_s["caught_mean"] - st_u["caught_mean"],
        },
        "main_paired_greedy6s_minus_threshold_admitted_per_seed": [round(x, 6) for x in main_diff],
        "metric": {
            "name": "admitted rate at 25% for greedy6s minus threshold: point estimate = paired mean over seeds 0-3 on the "
                    "full pool (750 calls); decision statistic = percentile and basic 95% upper bounds of a subsample "
                    "bootstrap (B=200 half-pool subsamples WITHOUT replacement, 375 calls per arm, seeds 0-3, per-subsample "
                    "error denominator)",
            "budget": budget,
            "main_mean_admitted_diff_seeds_0_to_7": main_mean_8,
            "point_estimate_seeds_0_to_3": point,
            "bootstrap": {"n_resamples": args.boot, "seeds": bs, "budget": budget, "subsample_size": m,
                          "replacement": False, "denominator": "per-subsample errors",
                          "mean": s["mean"], "bias": s["bias"], "sd": s["sd"],
                          "ci95_percentile": s["ci95_percentile"], "ci95_basic": s["ci95_basic"],
                          "n_below_zero": s["n_below_zero"]},
            "bias_tolerance": BIAS_TOL,
            "threshold_subsample_mean_seeds_0_to_3": thr_boot,
            "threshold_full_pool_admitted_mean_seeds_0_to_3": thr_full,
            "greedy6s_minus_threshold_admitted_per_resample": diffs,
            "verdict_rule": VERDICT_RULE,
        },
        "verdict": verdict,
        "wall_s": round(time.time() - t0, 1),
    })
    path = write_result(args.out, result)
    log(f"\nwrote {path} (wall {result['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
