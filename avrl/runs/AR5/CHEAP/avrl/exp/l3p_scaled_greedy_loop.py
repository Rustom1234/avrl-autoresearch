#!/usr/bin/env python3
"""L3p: does feature scaling let the online greedy loop admit fewer corrupted rewards than the per-step threshold at 25%?

Copied from avrl/exp/l3i_no_embedding_regression.py (the L3i file is not edited). Same pool, seed permutations, step size,
selection RNG, control checks, paired bootstrap and verdict structure as L3i. Budget 0.25 only. The one new piece is
GreedyScaled, which overrides update() only: the regression is refit inside a StandardScaler pipeline. select() is
inherited from GreedyOnlineAllocator unchanged. No Thompson step, no cells, no noise.

Arms:
  threshold  ThresholdAllocator (allocators.py). Checked per seed against runs/E03/result.json before any greedy arm runs.
  greedy6u   GreedyOnlineAllocator (allocators.py, unchanged), the unscaled control. Checked per seed against
             runs/L3i/result.json main_25.greedy6 before greedy6s runs.
  greedy6s   GreedyScaled (tested arm): Pipeline(StandardScaler, LogisticRegression(max_iter=1000)) refit on verified groups.

If either control fails the verdict is INVALID_CONTROL and greedy6s is never run, so no greedy6s number is read.

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
E03_RESULT = os.path.join(AVRL_DIR, "runs", "E03", "result.json")
L3I_RESULT = os.path.join(AVRL_DIR, "runs", "L3i", "result.json")
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
SPEC_GREEDY6U_CAUGHT_025 = [332, 327, 323, 323, 317, 317, 318, 317]
NEAR_MISS_CAUGHT = 331.25
VERDICT_RULE = (
    "PASS iff the 95% percentile CI upper bound of (greedy6s admitted minus threshold admitted) at 25% is below 0 "
    "(primary: per-resample error denominator; paired bootstrap over the 3000 groups, B=100, seeds 0-3) AND both "
    "controls reproduce (threshold per seed vs E03, greedy6u per seed vs L3i). INVALID_CONTROL if either control fails, "
    "in which case greedy6s is not run. Otherwise KILL (CI upper bound 0 or above). A greedy6s caught mean of 331.25 or "
    "more without a PASS is a near miss that motivates L3r with the scaled arm, not a pass. A PASS is a candidate "
    "re-specification only: claim-2 credit needs L3r on a second pool."
)
ASSUMPTIONS = [
    "steps of 16 groups over a seed-permuted pool (last step 8 groups); n_verify(step_groups, 0.25, step=t) as in E03 and L3i",
    "feat_vec = [agreement, n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen] (6 dims), as in L3i",
    "greedy6u = allocators.GreedyOnlineAllocator unchanged (control): unscaled LogisticRegression(max_iter=1000), refit_every 20, prior_strength 2.0",
    "greedy6s = GreedyScaled: update() appends verified feat_vec and label and refits on the same cadence (refit when the verified count "
    "crosses a multiple of 20, with both classes present) as Pipeline(StandardScaler, LogisticRegression(max_iter=1000)) on the cumulative verified set; "
    "select() inherited from GreedyOnlineAllocator, so the cold start before the first fit is the threshold order",
    "cell key is a constant 0 in every record and verified tuple; greedy selection never reads cells; the cell_beta bookkeeping of "
    "LearnedAllocator.update is not needed by greedy selection and is dropped in greedy6s",
    "admitted = (errors - caught) / errors. Main table: errors = 436. Primary bootstrap: errors of each resample (as L3i). Supplementary bootstrap: fixed 436 denominator",
    "permutation rng default_rng(1000+seed); selection rng default_rng(2000+seed) per run; bootstrap resample default_rng(10000+b), as in L3i",
    "gold (proxy_error) used only for scoring and as the update label of verified groups",
    "gating: greedy6s runs only if threshold reproduces E03 per seed and greedy6u reproduces runs/L3i/result.json per seed",
    "bootstrap runs threshold and greedy6s only (greedy6u is a main-table control, not in the bootstrap)",
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


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci95": [float(lo), float(hi)],
            "upper_below_0": bool(hi < 0), "frac_below_zero": float((x < 0).mean()),
            "n_below_zero": int((x < 0).sum()), "n_resamples": int(len(x))}


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


def check_threshold_vs_e03(rows, e03, n_err):
    """Threshold must reproduce E03 per seed at 25%. Checked before any greedy arm runs."""
    e = e03["per_budget"]["0.25"]
    ref = e["arms"]["threshold"]
    got_caught = [r["caught"] for r in rows]
    got_adm = [round(r["admitted"], 6) for r in rows]
    checks = {
        "caught_per_seed_equals_spec": got_caught == SPEC_THRESHOLD_CAUGHT_025,
        "caught_per_seed_equals_e03": got_caught == ref["caught_per_seed"],
        "admitted_per_seed_equals_e03": got_adm == ref["admitted_per_seed"],
        "errors_per_seed_equal_e03": all(x == n_err for x in e["errors_per_seed"]),
    }
    return all(checks.values()), checks


def check_greedy6u_vs_l3i(rows, l3i):
    """Unscaled greedy control must reproduce L3i per seed at 25%. Checked before greedy6s runs."""
    ref = l3i["main_25"]["greedy6"]["caught_per_seed"]
    got = [r["caught"] for r in rows]
    checks = {
        "caught_per_seed_equals_spec": got == SPEC_GREEDY6U_CAUGHT_025,
        "caught_per_seed_equals_l3i": got == ref,
    }
    return all(checks.values()), checks


def bootstrap(pool, budget, seeds, n_boot, e_fixed, log, cc):
    """Paired bootstrap over pool groups, threshold and greedy6s only. diffs uses each resample's own error count as the
    admitted denominator (primary). fixed uses the main pool's error count e_fixed (436), the literal L3p spec formula."""
    n = len(pool["y"])
    diffs, fixed, thr_boot = [], [], []
    for b in range(n_boot):
        ids = np.random.default_rng(10000 + b).choice(n, size=n, replace=True)
        sub = subset(pool, ids)
        thr = [cc.run("threshold", "bootstrap", lambda s=s: run_arm(sub, "threshold", budget, s)) for s in seeds]
        grd = [cc.run("greedy6s", "bootstrap", lambda s=s: run_arm(sub, "greedy6s", budget, s)) for s in seeds]
        thr_boot.append(float(np.mean([r["admitted"] for r in thr])))
        diffs.append(float(np.mean([g["admitted"] - t["admitted"] for g, t in zip(grd, thr)])))
        fixed.append(float(np.mean([t["caught"] - g["caught"] for g, t in zip(grd, thr)])) / e_fixed)
        if (b + 1) % 10 == 0:
            log(f"bootstrap {b + 1}/{n_boot} greedy6s-threshold {diffs[-1]:+.4f}")
    return diffs, fixed, thr_boot


def write_result(out, result):
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    return os.path.join(out, "result.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.25])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--boot_budget", type=float, default=0.25)
    ap.add_argument("--boot_seeds", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.budgets == [0.25] and args.boot_budget == 0.25, "L3p is defined at budget 0.25 only"
    assert args.seeds == 8 and args.boot_seeds == 4 and args.boot == 100, "spec: seeds 0-7, bootstrap seeds 0-3, B=100"
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
    e03 = load_json(E03_RESULT)
    l3i = load_json(L3I_RESULT)
    log(f"L3p pool {args.pool}: groups={n} proxy_errors={n_err} step={STEP} "
        f"steps={math.ceil(n / STEP)} budget={budget} verifier_calls_per_run={round(budget * n)}")
    log(f"feature columns ({len(FEAT_COLS)}): {FEAT_COLS}")
    log("greedy6s = GreedyScaled: Pipeline(StandardScaler, LogisticRegression(max_iter=1000)) refit_every=20 prior_strength=2.0; "
        "no noise, no cells in selection")
    log("arms: threshold (control vs E03), greedy6u (control vs L3i, unscaled), greedy6s (tested arm)")

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        cc = ConvCounter(wrec)

        thr_rows = [cc.run("threshold", "main", lambda s=s: run_arm(pool, "threshold", budget, s)) for s in seeds]
        thr_ok, thr_checks = check_threshold_vs_e03(thr_rows, e03, n_err)
        log(f"threshold vs E03 per seed: {'PASS' if thr_ok else 'FAIL'}; caught {[r['caught'] for r in thr_rows]}")

        g6u_rows = [cc.run("greedy6u", "main", lambda s=s: run_arm(pool, "greedy6u", budget, s)) for s in seeds]
        g6u_ok, g6u_checks = check_greedy6u_vs_l3i(g6u_rows, l3i)
        log(f"greedy6u vs L3i per seed: {'PASS' if g6u_ok else 'FAIL'}; caught {[r['caught'] for r in g6u_rows]}")

        controls_ok = thr_ok and g6u_ok
        result = {
            "id": "L3p",
            "pool": args.pool,
            "n_groups": n,
            "n_proxy_errors": n_err,
            "step_size_groups": STEP,
            "budget": budget,
            "verifier_calls_per_run": round(budget * n),
            "feature_cols": FEAT_COLS,
            "seeds_main": args.seeds,
            "seeds_bootstrap": bs,
            "B": args.boot,
            "assumptions": ASSUMPTIONS,
            "checks": {
                "threshold_vs_e03": thr_checks,
                "greedy6u_vs_l3i": g6u_checks,
                "controls_pass": controls_ok,
                "verifier_calls_equal_750_all_arms": True,
            },
            "main_25": {"threshold": arm_stats(thr_rows), "greedy6u": arm_stats(g6u_rows)},
            "verdict_rule": VERDICT_RULE,
        }

        if not controls_ok:
            verdict = "INVALID_CONTROL"
            log("INVALID_CONTROL: a control did not reproduce. greedy6s is not run and no greedy6s number is read.")
            result.update({"verdict": verdict, "near_miss": None,
                           "convergence_warnings": {
                               "threshold": {"main": cc.get("threshold", "main")},
                               "greedy6u": {"main": cc.get("greedy6u", "main")}}})
            result["wall_s"] = round(time.time() - t0, 1)
            write_result(args.out, result)
            log(f"verdict {verdict}; wrote {os.path.join(args.out, 'result.json')}")
            return 3

        g6s_rows = [cc.run("greedy6s", "main", lambda s=s: run_arm(pool, "greedy6s", budget, s)) for s in seeds]
        st_t, st_u, st_s = arm_stats(thr_rows), arm_stats(g6u_rows), arm_stats(g6s_rows)
        log(f"  greedy6s caught per seed {st_s['caught_per_seed']} mean {st_s['caught_mean']:.2f} "
            f"admitted mean {st_s['admitted_mean']:.4f}")
        log(f"ConvergenceWarnings main: greedy6s {cc.get('greedy6s', 'main')}, greedy6u {cc.get('greedy6u', 'main')}, "
            f"threshold {cc.get('threshold', 'main')}")

        main_diff = [g["admitted"] - t["admitted"] for g, t in zip(g6s_rows, thr_rows)]
        main_mean_8 = float(np.mean(main_diff))
        point = float(np.mean(main_diff[:bs]))
        log(f"paired (greedy6s - threshold) admitted: seeds 0-7 mean {main_mean_8:+.4f}; seeds 0-{bs - 1} {point:+.4f}")

        log(f"\nbootstrap: B={args.boot} resamples of {n} groups with replacement (default_rng(10000+b)), "
            f"seeds 0-{bs - 1}, budget {budget:.2f}")
        diffs, fixed, thr_boot = bootstrap(pool, budget, list(range(bs)), args.boot, n_err, log, cc)

    s = summarize(diffs)
    s_fixed = summarize(fixed)
    verdict = "PASS" if s["upper_below_0"] else "KILL"
    near_miss = bool(verdict == "KILL" and st_s["caught_mean"] >= NEAR_MISS_CAUGHT)
    log(f"CI95 (greedy6s - threshold) admitted at 0.25, per-resample denominator: {[round(x, 4) for x in s['ci95']]}, "
        f"resamples below 0: {s['n_below_zero']}/{s['n_resamples']}")
    log(f"CI95 (greedy6s - threshold) admitted at 0.25, fixed denominator {n_err}: "
        f"{[round(x, 4) for x in s_fixed['ci95']]}, resamples below 0: {s_fixed['n_below_zero']}/{s_fixed['n_resamples']}")
    log(f"caught mean seeds 0-7: greedy6s {st_s['caught_mean']:.2f}, greedy6u {st_u['caught_mean']:.2f}, "
        f"threshold {st_t['caught_mean']:.2f}")
    log(f"near miss (KILL with greedy6s caught mean >= {NEAR_MISS_CAUGHT}): {near_miss}")
    log(f"overall L3p verdict: {verdict}")

    result.update({
        "main_25": {"threshold": st_t, "greedy6u": st_u, "greedy6s": st_s},
        "caught_mean_seeds_0_to_7": {
            "threshold": st_t["caught_mean"],
            "greedy6u": st_u["caught_mean"],
            "greedy6s": st_s["caught_mean"],
            "greedy6s_minus_threshold": st_s["caught_mean"] - st_t["caught_mean"],
            "greedy6s_minus_greedy6u": st_s["caught_mean"] - st_u["caught_mean"],
        },
        "main_paired_greedy6s_minus_threshold_admitted_per_seed": [round(x, 6) for x in main_diff],
        "convergence_warnings": {
            "greedy6s": {"main": cc.get("greedy6s", "main"), "bootstrap": cc.get("greedy6s", "bootstrap"),
                         "total": cc.get("greedy6s", "main") + cc.get("greedy6s", "bootstrap")},
            "greedy6u": {"main": cc.get("greedy6u", "main"), "bootstrap": 0,
                         "note": "greedy6u is not run in the bootstrap"},
            "threshold": {"main": cc.get("threshold", "main"), "bootstrap": cc.get("threshold", "bootstrap"),
                          "note": "threshold fits no model"},
        },
        "metric": {
            "name": "admitted rate at 25% for greedy6s minus threshold, paired over seeds 0-7 at 750 calls "
                    "(main_mean_admitted_diff_seeds_0_to_7); decision statistic is the 95% percentile CI "
                    "from a paired bootstrap over the 3000 groups (B resamples, seeds 0-3, per-resample error denominator)",
            "budget": budget,
            "main_mean_admitted_diff_seeds_0_to_7": main_mean_8,
            "point_estimate_seeds_0_to_3": point,
            "bootstrap": {"B": args.boot, "seeds": bs, "budget": budget,
                          "denominator": "per-resample errors (primary)", **s},
            "bootstrap_fixed_denominator_436": {"denominator": n_err, **s_fixed},
            "threshold_admitted_per_resample_mean_seeds_0_to_3": thr_boot,
            "greedy6s_minus_threshold_admitted_per_resample": diffs,
            "verdict_rule": VERDICT_RULE,
        },
        "verdict": verdict,
        "near_miss": near_miss,
        "wall_s": round(time.time() - t0, 1),
    })
    path = write_result(args.out, result)
    log(f"\nwrote {path} (wall {result['wall_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
