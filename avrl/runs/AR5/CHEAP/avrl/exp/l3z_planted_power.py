#!/usr/bin/env python3
"""L3z: planted-effect power check of the L3s gate at a -0.03 greedy advantage.

Copied from avrl/exp/l3v_null_calibration.py (that file is not edited). The L3s loop, arms, seeds, bootstrap and
verdict_from are unchanged. Changes:
  1. Planted pool (build_planted): the L3v null construction (the six FEAT_COLS shuffled together by
     default_rng(POOL_SEED).permutation(3000), proxy_error in place) plus one label-informed injection at strength s:
     X[i, c] = shuffled[i, c] + s * sgn_c * sd_c * proxy_error_i for the six columns. sgn_c is the sign of the base
     correlation with proxy_error and sd_c the base population SD. The injection has no random draw.
  2. run_arm reads n_distinct as float (the injected value is not an integer). Otherwise identical.
  3. Two modes. --pilot runs the fixed grid on POOL_SEED 30000 and 30001 (main runs only) and selects s by the
     pre-registered rule ("L3z 2026-10-19 pre-registration" in avrl/RESEARCH_LOG.md). Without --pilot, one planted pool
     at --strength and --pool_seed (40000 + k) gets the full L3s design.
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
N_GROUPS = 3000
BUDGET = 0.25
FEAT_COLS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
BASE_FEATS = os.path.join(AVRL_DIR, "runs", "p0", "feats.parquet")
BIAS_TOL = 0.003
PILOT_SEEDS = (30000, 30001)
EVAL_SEEDS = range(40000, 40020)
GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00, 1.25, 1.50]
TARGET = -0.03
TARGET_TOL = 0.005
PILOT_RULE = (
    "Pilot statistic for each grid strength s = mean over POOL_SEED 30000 and 30001 and seeds 0-7 of (greedy6s admitted "
    "minus threshold admitted) at budget 0.25 on the full planted pool (main runs only). s* = the grid value whose pilot "
    "statistic is closest to -0.03 (ties to the smaller s). Status OK if |pilot statistic(s*) + 0.03| <= 0.005, else "
    "INFEASIBLE_PLANT (no evaluation is launched)."
)
VERDICT_RULE = (
    "PASS iff the percentile 95% upper bound AND the basic 95% upper bound of (greedy6s admitted minus threshold admitted) "
    "at 25% are both below 0, AND |bias| < 0.003 (bias = subsample bootstrap mean minus the seeds 0-3 full-pool point "
    "estimate; B=200 half-pool subsamples WITHOUT replacement, seeds 0-3). "
    "KILL iff either upper bound is 0 or above. "
    "UNASSIGNED CASE (both upper bounds below 0 and |bias| >= 0.003): INCONCLUSIVE, no claim credit. "
    "INVALID_CONTROL if any arm fails same-pool determinism (a rerun on the same pool and seed does not reproduce the main "
    "run's caught count), in which case no bootstrap is run. "
    "L3z: this rule is applied to each planted pool; the aggregate is in avrl/exp/l3z_aggregate.py."
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
    "gold (proxy_error) used only for scoring and as the update label of verified groups, never as a selection input",
    "gating: same-pool determinism. Each arm is rerun once per seed 0-7 on the same planted pool; the rerun must reproduce the main "
    "run's per-seed caught counts. All checked before the bootstrap is read",
    "planted pool (build_planted): the six feature columns are shuffled together by default_rng(POOL_SEED).permutation(3000), proxy_error "
    "stays in place (436 errors), then X[i, c] += s * sgn_c * sd_c * proxy_error_i for the six columns. sgn_c = sign of the base "
    "correlation with proxy_error, sd_c = population SD of the base column. The injection is the only label information in X",
    "agreement and n_distinct in the records are the injected values (n_distinct carried as float); threshold order is (agreement ascending, n_distinct descending)",
    "caveat: the shuffle removes the E01 feature-label structure, so these pools sit in the L3v null regime (no real signal), not in the real E01 regime; "
    "a half-pool subsample changes the loop itself (94 steps and 375 verified labels per run, versus 188 steps and 750 on the full pool)",
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


def base_table():
    df = pd.read_parquet(BASE_FEATS)
    assert len(df) == N_GROUPS, f"expected 3000 groups, got {len(df)}"
    base = df[FEAT_COLS].to_numpy(dtype=np.float64)
    y = df["proxy_error"].to_numpy(dtype=np.int64)
    sd = base.std(axis=0)
    sgn = np.sign([np.corrcoef(base[:, c], y.astype(np.float64))[0, 1] for c in range(len(FEAT_COLS))])
    return base, y, sd, sgn


def build_planted(pool_seed, s):
    """Planted pool: L3v null shuffle (labels in place), then the label-informed injection at strength s."""
    base, y, sd, sgn = base_table()
    perm = np.random.default_rng(pool_seed).permutation(N_GROUPS)
    X = base[perm] + s * y[:, None].astype(np.float64) * (sgn * sd)[None, :]
    pool = {"X": X, "agreement": X[:, 0].copy(), "n_distinct": X[:, 1].copy(), "y": y}
    diag = {
        "pool_seed": pool_seed,
        "strength": s,
        "sgn": [float(v) for v in sgn],
        "sd": [float(v) for v in sd],
        "planted_corr_with_label": [float(np.corrcoef(X[:, c], y.astype(np.float64))[0, 1]) for c in range(len(FEAT_COLS))],
    }
    return pool, diag


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
                "n_distinct": float(pool["n_distinct"][g]),
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
    """Same-pool determinism. A rerun of an arm on the same pool and seed must reproduce its per-seed caught counts."""
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


def pilot(out):
    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    log(f"L3z pilot: grid {GRID}, pilot POOL_SEED {list(PILOT_SEEDS)}, seeds 0-7, budget {BUDGET}, main runs only")
    log(f"PILOT RULE: {PILOT_RULE}")
    rows = []
    for s in GRID:
        diffs, thr_means, grd_means = [], [], []
        for pseed in PILOT_SEEDS:
            pool, _ = build_planted(pseed, s)
            thr = [run_arm(pool, "threshold", BUDGET, sd) for sd in range(8)]
            grd = [run_arm(pool, "greedy6s", BUDGET, sd) for sd in range(8)]
            diffs.append(float(np.mean([g["admitted"] - t["admitted"] for g, t in zip(grd, thr)])))
            thr_means.append(float(np.mean([t["admitted"] for t in thr])))
            grd_means.append(float(np.mean([g["admitted"] for g in grd])))
        stat = float(np.mean(diffs))
        rows.append({"strength": s, "pilot_statistic": stat, "per_pool_diff": diffs,
                     "threshold_admitted_mean": float(np.mean(thr_means)),
                     "greedy6s_admitted_mean": float(np.mean(grd_means))})
        log(f"s={s:.2f} pilot statistic {stat:+.4f} per pool {[round(x, 4) for x in diffs]} "
            f"threshold admitted {np.mean(thr_means):.4f} greedy6s admitted {np.mean(grd_means):.4f}")
    dist = [abs(r["pilot_statistic"] - TARGET) for r in rows]
    best = min(range(len(rows)), key=lambda i: (dist[i], rows[i]["strength"]))
    ok = dist[best] <= TARGET_TOL
    status = "OK" if ok else "INFEASIBLE_PLANT"
    result = {
        "id": "L3z", "stage": "pilot", "status": status,
        "chosen_strength": rows[best]["strength"] if ok else None,
        "closest_grid_strength": rows[best]["strength"],
        "closest_pilot_statistic": rows[best]["pilot_statistic"],
        "distance_to_target": dist[best],
        "target": TARGET, "tolerance": TARGET_TOL, "grid": GRID,
        "pilot_pool_seeds": list(PILOT_SEEDS), "pilot_rule": PILOT_RULE,
        "rows": rows, "wall_s": round(time.time() - t0, 1),
    }
    path = write_result(out, result)
    log(f"closest grid strength s={rows[best]['strength']:.2f} pilot statistic {rows[best]['pilot_statistic']:+.4f}, "
        f"distance to -0.03 {dist[best]:.4f} (tolerance {TARGET_TOL})")
    log(f"PILOT STATUS: {status}; chosen_strength {result['chosen_strength']}")
    log(f"wrote {path} (wall {result['wall_s']} s)")
    return 0


def evaluate(args):
    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    budget = BUDGET
    bs = args.boot_seeds
    seeds = list(range(args.seeds))
    pool, diag = build_planted(args.pool_seed, args.strength)
    n = len(pool["y"])
    n_err = int(pool["y"].sum())
    m = n // 2
    log(f"L3z planted pool: POOL_SEED={args.pool_seed} strength={args.strength} groups={n} proxy_errors={n_err} "
        f"step={STEP} steps={math.ceil(n / STEP)} budget={budget} verifier_calls_per_run={round(budget * n)}")
    log("planted feature-label correlations: " + ", ".join(
        f"{c}={v:+.3f}" for c, v in zip(FEAT_COLS, diag["planted_corr_with_label"])))
    log(f"bootstrap draws: B={args.boot} subsamples of {m} groups WITHOUT replacement (default_rng(10000+b)), "
        f"each run at 25% = {round(budget * m)} calls per arm")
    log(f"feature columns ({len(FEAT_COLS)}): {FEAT_COLS}")
    log("arms: threshold, greedy6u (unscaled), greedy6s; each arm is rerun once per seed for same-pool determinism")
    log("PRE-REGISTERED RULES (L3z pre-registration in avrl/RESEARCH_LOG.md, written before the pilot; VERDICT_RULE as in L3s):")
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
            "id": "L3z",
            "pool": f"planted(POOL_SEED={args.pool_seed}, strength={args.strength}) from {BASE_FEATS}",
            "pool_seed": args.pool_seed,
            "strength": args.strength,
            "planted_diagnostics": diag,
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
    log(f"overall L3z per-pool verdict on planted POOL_SEED {args.pool_seed} s={args.strength}: {verdict}")

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
            "name": "admitted rate at 25% for greedy6s minus threshold on a planted pool: achieved = paired mean over seeds 0-7 "
                    "on the full planted pool (750 calls); point = paired mean over seeds 0-3; decision statistic = percentile and "
                    "basic 95% upper bounds of a subsample bootstrap (B=200 half-pool subsamples WITHOUT replacement, 375 calls "
                    "per arm, seeds 0-3, per-subsample error denominator)",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pool_seed", type=int)
    ap.add_argument("--strength", type=float)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.25])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--boot_budget", type=float, default=0.25)
    ap.add_argument("--boot_seeds", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.pilot:
        return pilot(args.out)
    assert args.pool_seed is not None and args.strength is not None, "evaluation needs --pool_seed and --strength"
    assert args.pool_seed in EVAL_SEEDS, f"evaluation pools use seeds 40000..40019, got {args.pool_seed}"
    assert args.budgets == [0.25] and args.boot_budget == 0.25, "L3z is defined at budget 0.25 only"
    assert args.seeds == 8 and args.boot_seeds == 4 and args.boot == 200, "spec: seeds 0-7, bootstrap seeds 0-3, B=200"
    os.makedirs(args.out, exist_ok=True)
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
