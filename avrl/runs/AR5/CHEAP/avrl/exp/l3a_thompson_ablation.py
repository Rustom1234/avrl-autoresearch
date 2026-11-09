#!/usr/bin/env python3
"""L3a: is the E03 loss in LearnedAllocator's Thompson-over-cells step or in the regression?

Derived from avrl/exp/e03_real_allocators.py (same pool, permutation, step size, selection RNG,
bootstrap resampling and verdict rule). Budget 0.25 only.

Arms:
  threshold        ThresholdAllocator, the baseline. Re-run, and checked against E03 before any variant runs.
  learned          E03 reference, read from runs/E03/result.json. Not rerun.
  pergroup_beta    (a) score_i ~ Beta(2.0 p_i, 2.0 (1 - p_i)) per group, no cell_beta dict.
  cell_mean_prior  (b) an unseen cell's Beta is set from the mean regression p over that cell's candidates
                       in the current select step; label updates as LearnedAllocator.update.
  logit_noise      (c) score_i = logit(p_i) + N(0, 1), sigma fixed at 1.0, no cells.

Each variant subclasses allocators.LearnedAllocator with refit_every=20, prior_strength=2.0, the unscaled
LogisticRegression(max_iter=1000) and E03's 22-dim feature vector. Cold start (before the first fit) is the
threshold order, as in LearnedAllocator.

Pool: runs/p0/feats.parquet (3000 groups, 8 rollouts each), read only. Gold (proxy_error) is used only for
scoring and as the update label of verified groups, never as a selection input.

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
sys.path.insert(0, AVRL_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import allocators as A  # noqa: E402

STEP = 16
PC_COLS = [f"pc{i}" for i in range(16)]
FEAT_COLS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"] + PC_COLS
P_CLIP = 1e-3
LOGIT_SIGMA = 1.0
SPEC_THRESHOLD_CAUGHT_025 = [340, 341, 344, 332, 331, 328, 325, 333]
BONF_TAIL = 100 * (0.05 / 3) / 2  # two-sided 98.33% interval (Bonferroni over 3 variants), percentile tail
VARIANTS = ["pergroup_beta", "cell_mean_prior", "logit_noise"]


def _probs(alloc, group_records):
    X = np.array([r["feat_vec"] for r in group_records])
    return np.clip(alloc.model.predict_proba(X)[:, 1], P_CLIP, 1 - P_CLIP)


def _top_k(idx, scores, k):
    order = np.argsort(-scores, kind="stable")
    return [idx[i] for i in order[:k]]


def _update_no_cells(alloc, verified):
    """LearnedAllocator.update without cell_beta bookkeeping. Same accumulation and refit schedule."""
    for feat_vec, label, _cell in verified:
        alloc.X.append(feat_vec)
        alloc.y.append(label)
    n = len(alloc.y)
    due = n >= alloc.refit_every and n // alloc.refit_every > alloc.last_fit_n // alloc.refit_every
    if due and len(set(alloc.y)) > 1:
        alloc.model = LogisticRegression(max_iter=1000).fit(np.array(alloc.X), np.array(alloc.y))
        alloc.last_fit_n = n


class PerGroupBeta(A.LearnedAllocator):
    """(a) Each candidate draws its own Beta from its own regression p. No cells."""
    name = "pergroup_beta"

    def select(self, group_records, k, rng):
        if k == 0:
            return []
        if self.model is None:
            return [r["idx"] for r in A.ThresholdAllocator._order(group_records)[:k]]
        p = _probs(self, group_records)
        scores = rng.beta(self.prior_strength * p, self.prior_strength * (1 - p))
        return _top_k([r["idx"] for r in group_records], scores, k)

    def update(self, verified):
        _update_no_cells(self, verified)


class CellMeanPrior(A.LearnedAllocator):
    """(b) Unseen cells start at the mean regression p of their candidates in this step."""
    name = "cell_mean_prior"

    def select(self, group_records, k, rng):
        if k == 0:
            return []
        if self.model is None:
            return [r["idx"] for r in A.ThresholdAllocator._order(group_records)[:k]]
        p = _probs(self, group_records)
        sums, counts = {}, {}
        for r, pi in zip(group_records, p):
            sums[r["cell"]] = sums.get(r["cell"], 0.0) + pi
            counts[r["cell"]] = counts.get(r["cell"], 0) + 1
        for cell in sums:
            if cell not in self.cell_beta:
                m = sums[cell] / counts[cell]
                self.cell_beta[cell] = [self.prior_strength * m, self.prior_strength * (1 - m)]
        scores = np.empty(len(group_records))
        for i, r in enumerate(group_records):
            a, b = self.cell_beta[r["cell"]]
            scores[i] = rng.beta(a, b)
        return _top_k([r["idx"] for r in group_records], scores, k)


class LogitNoise(A.LearnedAllocator):
    """(c) Regression ranking with Gaussian noise on the logit scale. No cells."""
    name = "logit_noise"

    def select(self, group_records, k, rng):
        if k == 0:
            return []
        if self.model is None:
            return [r["idx"] for r in A.ThresholdAllocator._order(group_records)[:k]]
        p = _probs(self, group_records)
        scores = np.log(p / (1 - p)) + rng.normal(0.0, LOGIT_SIGMA, size=len(p))
        return _top_k([r["idx"] for r in group_records], scores, k)

    def update(self, verified):
        _update_no_cells(self, verified)


ARMS = {
    "threshold": A.ThresholdAllocator,
    "pergroup_beta": PerGroupBeta,
    "cell_mean_prior": CellMeanPrior,
    "logit_noise": LogitNoise,
}


def load_pool(path):
    df = pd.read_parquet(path)
    assert len(df) == 3000, f"expected 3000 groups, got {len(df)}"
    return {
        "X": df[FEAT_COLS].to_numpy(dtype=np.float64),
        "agreement": df["agreement"].to_numpy(dtype=np.float64),
        "n_distinct": df["n_distinct"].to_numpy(dtype=np.int64),
        "cell": df["cell100"].to_numpy(dtype=np.int64),
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
        # cheap, gold-free fields only
        records = [
            {
                "idx": j,
                "feat_vec": pool["X"][g],
                "cell": int(pool["cell"][g]),
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
            verified.append((pool["X"][g], int(pool["y"][g]), int(pool["cell"][g])))
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
    blo, bhi = np.percentile(x, [BONF_TAIL, 100 - BONF_TAIL])
    return {"mean": float(x.mean()), "ci95": [float(lo), float(hi)],
            "ci98_3_bonferroni": [float(blo), float(bhi)],
            "upper_below_0": bool(hi < 0), "frac_below_zero": float((x < 0).mean()),
            "n_below_zero": int((x < 0).sum()), "n_resamples": int(len(x))}


def load_e03():
    with open(E03_RESULT) as f:
        return json.load(f)


def check_threshold_vs_e03(thr_rows, e03, n_err):
    """Stop before any variant number is read if the threshold arm does not reproduce E03."""
    e = e03["per_budget"]["0.25"]
    ref = e["arms"]["threshold"]
    got_caught = [r["caught"] for r in thr_rows]
    got_adm = [round(r["admitted"], 6) for r in thr_rows]
    checks = {
        "threshold_caught_per_seed_equals_spec": got_caught == SPEC_THRESHOLD_CAUGHT_025,
        "threshold_caught_per_seed_equals_e03": got_caught == ref["caught_per_seed"],
        "threshold_admitted_per_seed_equals_e03": got_adm == ref["admitted_per_seed"],
        "errors_per_seed_equal_e03": all(x == n_err for x in e["errors_per_seed"]),
    }
    if not all(checks.values()):
        sys.exit("STOP: threshold does not reproduce E03 at 25%. "
                 f"got caught {got_caught}, got admitted {got_adm}, "
                 f"E03 caught {ref['caught_per_seed']}, E03 admitted {ref['admitted_per_seed']}, "
                 f"checks {checks}")
    return checks


def bootstrap(pool, budget, seeds, n_boot, log):
    n = len(pool["y"])
    diffs = {arm: [] for arm in VARIANTS}
    thr_adm = []
    for b in range(n_boot):
        ids = np.random.default_rng(10000 + b).choice(n, size=n, replace=True)
        sub = subset(pool, ids)
        adm = {arm: [] for arm in ARMS}
        for s in seeds:
            for arm in ARMS:
                adm[arm].append(run_arm(sub, arm, budget, s)["admitted"])
        thr_adm.append(float(np.mean(adm["threshold"])))
        for arm in VARIANTS:
            diffs[arm].append(float(np.mean([v - t for v, t in zip(adm[arm], adm["threshold"])])))
        if (b + 1) % 10 == 0:
            log(f"bootstrap {b + 1}/{n_boot} " + " ".join(f"{arm} {diffs[arm][-1]:+.4f}" for arm in VARIANTS))
    return diffs, thr_adm


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
    assert args.budgets == [0.25] and args.boot_budget == 0.25, "L3a is defined at budget 0.25 only"
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
    e03 = load_e03()
    e_arms = e03["per_budget"]["0.25"]["arms"]
    log(f"L3a pool {args.pool}: groups={n} proxy_errors={n_err} step={STEP} "
        f"steps={math.ceil(n / STEP)} budget={budget} verifier_calls_per_run={round(budget * n)}")
    log(f"feature columns ({len(FEAT_COLS)}): {FEAT_COLS}")
    log("allocator defaults: refit_every=20 prior_strength=2.0 LogisticRegression(max_iter=1000) unscaled; cell=cell100")
    log(f"arms: threshold (baseline), learned (E03 reference, not rerun), variants {VARIANTS}")

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")

        thr_rows = [run_arm(pool, "threshold", budget, s) for s in seeds]
        thr_checks = check_threshold_vs_e03(thr_rows, e03, n_err)
        log(f"threshold check vs E03 passed. caught per seed {[r['caught'] for r in thr_rows]}")

        rows = {"threshold": thr_rows}
        for arm in VARIANTS:
            rows[arm] = [run_arm(pool, arm, budget, s) for s in seeds]
            st = arm_stats(rows[arm])
            log(f"  {arm:15s} caught per seed {st['caught_per_seed']} mean {st['caught_mean']:.1f} "
                f"admitted mean {st['admitted_mean']:.4f}")
        n_conv_main = sum(issubclass(w.category, ConvergenceWarning) for w in wrec)
        log(f"ConvergenceWarnings during main runs: {n_conv_main}")

        main_diff = {arm: [v["admitted"] - t["admitted"] for v, t in zip(rows[arm], thr_rows)] for arm in VARIANTS}
        point = {arm: float(np.mean(main_diff[arm][:bs])) for arm in VARIANTS}
        log(f"point estimate (variant - threshold admitted) at 0.25, seeds 0-{bs - 1}: "
            + ", ".join(f"{arm} {point[arm]:+.4f}" for arm in VARIANTS))

        log(f"\nbootstrap: B={args.boot} resamples of {n} groups with replacement (default_rng(10000+b)), "
            f"seeds 0-{bs - 1}, budget {budget:.2f}")
        diffs, thr_boot = bootstrap(pool, budget, list(range(bs)), args.boot, log)
        n_conv_total = sum(issubclass(w.category, ConvergenceWarning) for w in wrec)

    n_conv_boot = n_conv_total - n_conv_main
    log(f"ConvergenceWarnings during bootstrap: {n_conv_boot}; total {n_conv_total}")

    boot_summary = {arm: summarize(diffs[arm]) for arm in VARIANTS}
    per_variant = {arm: ("PASS" if boot_summary[arm]["upper_below_0"] else "KILL") for arm in VARIANTS}
    verdict = "PASS" if any(v == "PASS" for v in per_variant.values()) else "KILL"
    for arm in VARIANTS:
        s = boot_summary[arm]
        log(f"CI95 ({arm} - threshold) admitted at 0.25: {[round(x, 4) for x in s['ci95']]}, "
            f"98.3% Bonferroni {[round(x, 4) for x in s['ci98_3_bonferroni']]}, "
            f"resamples below 0: {s['n_below_zero']}/{s['n_resamples']} -> {per_variant[arm]}")
    log(f"overall L3a verdict: {verdict}")

    result = {
        "id": "L3a",
        "pool": args.pool,
        "n_groups": n,
        "n_proxy_errors": n_err,
        "step_size_groups": STEP,
        "budget": budget,
        "verifier_calls_per_run": round(budget * n),
        "seeds_main": args.seeds,
        "seeds_bootstrap": bs,
        "B": args.boot,
        "variants": {
            "pergroup_beta": "(a) score_i ~ Beta(2 p_i, 2 (1 - p_i)) per group, no cell_beta",
            "cell_mean_prior": "(b) unseen cell Beta from mean regression p of its candidates in the current step, LearnedAllocator label updates",
            "logit_noise": "(c) score_i = logit(p_i) + N(0, 1), no cells",
        },
        "assumptions": [
            "steps of 16 groups over a seed-permuted pool (last step 8 groups); n_verify(step_groups, 0.25, step=t) as in E03",
            "feat_vec = the same 22 dims as E03",
            "cell = cell100 (KMeans k=100 from p0_analyze.py); used only by cell_mean_prior",
            "cold start before the first fit: ThresholdAllocator order (inherited, as in LearnedAllocator)",
            "cell_mean_prior: a cell with no cell_beta entry gets Beta(2m, 2(1-m)), m = mean clipped regression p over that cell's candidates in this step; cells already in cell_beta keep their posterior, including cells that LearnedAllocator.update created with the default Beta(2,2) during cold start",
            "pergroup_beta and logit_noise never read or write cell_beta",
            "permutation rng default_rng(1000+seed); selection rng default_rng(2000+seed) shared across steps of one run; bootstrap resample default_rng(10000+b), as in E03",
            "gold (proxy_error) used only for scoring and as the update label of verified groups",
            "learned arm is the E03 reference read from runs/E03/result.json, not rerun; greedy is not run",
        ],
        "checks": {
            **thr_checks,
            "verifier_calls_equal_750_all_arms": True,
            "threshold_bootstrap_vs_e03": "not checkable: E03 result.json stores learned and greedy bootstrap CIs only, not per-resample threshold values",
        },
        "main_25": {
            "threshold": arm_stats(thr_rows),
            "learned_e03_reference": {
                "caught_per_seed": e_arms["learned"]["caught_per_seed"],
                "caught_mean": float(np.mean(e_arms["learned"]["caught_per_seed"])),
                "admitted_per_seed": e_arms["learned"]["admitted_per_seed"],
                "admitted_mean": float(np.mean(e_arms["learned"]["admitted_per_seed"])),
                "calls_per_seed": e_arms["learned"]["calls_per_seed"],
                "source": E03_RESULT,
            },
            **{arm: arm_stats(rows[arm]) for arm in VARIANTS},
        },
        "main_paired_minus_threshold_admitted_per_seed": {arm: [round(x, 6) for x in main_diff[arm]] for arm in VARIANTS},
        "e03_reference_learned_minus_threshold": e03["metric"],
        "metric": {
            "name": "paired bootstrap of (variant admitted - threshold admitted) at 25%, mean over seeds 0-3, "
                    "percentile 95% CI over B resamples, with the count of resamples below 0; "
                    "98.3% Bonferroni bound over 3 variants reported alongside",
            "budget": budget,
            "point_estimate_seeds_0_to_3": point,
            "bootstrap": {"B": args.boot, "seeds": bs, "budget": budget,
                          **{arm: boot_summary[arm] for arm in VARIANTS}},
            "threshold_admitted_per_resample_mean_seeds_0_to_3": thr_boot,
        },
        "per_variant_verdict": per_variant,
        "verdict": verdict,
        "verdict_rule": "a variant PASSes iff the 95% percentile CI upper bound of (variant admitted - threshold admitted) at 0.25 is < 0; overall PASS iff any variant passes. A PASS is a candidate re-specification, not a confirmed claim.",
        "convergence_warnings": {"main": n_conv_main, "bootstrap": n_conv_boot, "total": n_conv_total},
        "wall_s": round(time.time() - t0, 1),
    }
    out_path = os.path.join(args.out, "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    log(f"\nwrote {out_path} (wall {result['wall_s']} s)")


if __name__ == "__main__":
    main()
