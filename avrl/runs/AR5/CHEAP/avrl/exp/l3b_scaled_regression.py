#!/usr/bin/env python3
"""L3b: does standardizing the features inside the learned regression lift the learned arm at 25%?

Copied from avrl/exp/e03_real_allocators.py (E03) and modified minimally. Same pool, step size (16 groups),
cell/Thompson logic, permutation and selection seeds, and bootstrap design. What changes:
  - arms: threshold; learned (unscaled control, exactly as in avrl/allocators.py); learned_scaled
    (LearnedScaled below: Pipeline(StandardScaler, LogisticRegression(max_iter=1000)) refit on the cumulative
    verified X and y; select(), Thompson step, threshold cold start, refit_every=20, prior_strength=2.0 inherited).
  - ConvergenceWarnings are tallied per arm and per phase (main runs vs bootstrap fits).
  - the unscaled 25% control is checked against runs/E03/result.json.

Gold (proxy_error) is used only for scoring and as the update label of verified groups, as in E03.
CPU only. No GPU, no pip, no ssh. Writes result.json into --out. Log goes to stdout (the runner redirects it).
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
E03_RESULT = "/homes/rdubash/avrl_live/CHEAP/avrl/runs/E03/result.json"
sys.path.insert(0, AVRL_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import allocators as A  # noqa: E402

STEP = 16
PC_COLS = [f"pc{i}" for i in range(16)]
FEAT_COLS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"] + PC_COLS


class LearnedScaled(A.LearnedAllocator):
    """LearnedAllocator with standardized features inside the regression. Only the fit line differs from
    LearnedAllocator.update(); the cell_beta update, refit cadence and select() are inherited or copied verbatim."""
    name = "learned_scaled"

    def update(self, verified):
        for feat_vec, label, cell in verified:
            self.X.append(feat_vec)
            self.y.append(label)
            a, b = self.cell_beta.get(cell, [self.prior_strength, self.prior_strength])
            self.cell_beta[cell] = [a + label, b + (1 - label)]
        n = len(self.y)
        due = n >= self.refit_every and n // self.refit_every > self.last_fit_n // self.refit_every
        if due and len(set(self.y)) > 1:
            self.model = Pipeline([("scale", StandardScaler()),
                                   ("lr", LogisticRegression(max_iter=1000))]).fit(np.array(self.X), np.array(self.y))
            self.last_fit_n = n


ARMS = {
    "threshold": A.ThresholdAllocator,
    "learned": A.LearnedAllocator,
    "learned_scaled": LearnedScaled,
}
BOOT_ARMS = ["threshold", "learned", "learned_scaled"]
CONV = {"main": {arm: 0 for arm in ARMS}, "boot": {arm: 0 for arm in ARMS}}


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
    errors = int(pool["y"].sum())
    return {"arm": arm, "budget": budget, "seed": seed, "errors": errors, "caught": caught,
            "admitted": (errors - caught) / errors, "calls": calls}


def counted_run(pool, arm, budget, seed, phase):
    """run_arm, with the ConvergenceWarnings raised by this arm's fits tallied under CONV[phase][arm]."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        row = run_arm(pool, arm, budget, seed)
    CONV[phase][arm] += sum(issubclass(x.category, ConvergenceWarning) for x in w)
    return row


def run_budget(pool, budget, seeds):
    out = {arm: [counted_run(pool, arm, budget, s, "main") for s in seeds] for arm in ARMS}
    ref = [r["calls"] for r in out["threshold"]]
    for arm in ARMS:
        got = [r["calls"] for r in out[arm]]
        assert got == ref, f"verifier-call mismatch at budget {budget}: {arm} {got} vs {ref}"
    return out


def ci95(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci95": [float(lo), float(hi)],
            "frac_below_zero": float((x < 0).mean()), "n_resamples": int(len(x))}


def bootstrap(pool, budget, seeds, n_boot, log):
    n = len(pool["y"])
    diff_scaled, diff_unscaled = [], []
    for b in range(n_boot):
        ids = np.random.default_rng(10000 + b).choice(n, size=n, replace=True)
        sub = subset(pool, ids)
        adm = {arm: [] for arm in BOOT_ARMS}
        for s in seeds:
            rs = {arm: counted_run(sub, arm, budget, s, "boot") for arm in BOOT_ARMS}
            assert len({r["calls"] for r in rs.values()}) == 1, "verifier-call mismatch in bootstrap"
            for arm in BOOT_ARMS:
                adm[arm].append(rs[arm]["admitted"])
        diff_scaled.append(float(np.mean([s_ - t for s_, t in zip(adm["learned_scaled"], adm["threshold"])])))
        diff_unscaled.append(float(np.mean([u - t for u, t in zip(adm["learned"], adm["threshold"])])))
        if (b + 1) % 10 == 0:
            log(f"bootstrap {b + 1}/{n_boot} scaled-thr {diff_scaled[-1]:+.4f} unscaled-thr {diff_unscaled[-1]:+.4f}")
    return diff_scaled, diff_unscaled


def arm_stats(rows):
    return {
        "caught_per_seed": [r["caught"] for r in rows],
        "caught_mean": float(np.mean([r["caught"] for r in rows])),
        "admitted_per_seed": [round(r["admitted"], 6) for r in rows],
        "admitted_mean": float(np.mean([r["admitted"] for r in rows])),
        "calls_per_seed": [r["calls"] for r in rows],
    }


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
    assert args.boot_budget in args.budgets, "boot_budget must be one of --budgets"
    os.makedirs(args.out, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    pool = load_pool(args.pool)
    n = len(pool["y"])
    n_err = int(pool["y"].sum())
    log(f"L3b pool {args.pool}: groups={n} proxy_errors={n_err} step={STEP} "
        f"steps={math.ceil(n / STEP)} (last step {n - STEP * (math.ceil(n / STEP) - 1)} groups)")
    log(f"feature columns ({len(FEAT_COLS)}): {FEAT_COLS}")
    log("learned and learned_scaled: refit_every=20 prior_strength=2.0; cell=cell100")

    seeds = list(range(args.seeds))
    main_rows = {}
    per_budget = {}
    for b in args.budgets:
        rows = run_budget(pool, b, seeds)
        main_rows[b] = rows
        calls = rows["threshold"][0]["calls"]
        log(f"\n== budget {b:.2f}: verifier calls per run = {calls} (identical across arms and seeds)")
        for arm in ARMS:
            st = arm_stats(rows[arm])
            log(f"  {arm:15s} caught per seed {st['caught_per_seed']} mean {st['caught_mean']:.1f} "
                f"admitted mean {st['admitted_mean']:.4f}")
        thr = rows["threshold"]
        per_budget[b] = {
            "verifier_calls_per_run": calls,
            "verifier_calls_equal_across_arms": True,
            "errors_per_seed": [r["errors"] for r in thr],
            "arms": {arm: arm_stats(rows[arm]) for arm in ARMS},
            "learned_scaled_minus_threshold_admitted_mean": float(np.mean(
                [r["admitted"] - t["admitted"] for r, t in zip(rows["learned_scaled"], thr)])),
            "learned_minus_threshold_admitted_mean": float(np.mean(
                [r["admitted"] - t["admitted"] for r, t in zip(rows["learned"], thr)])),
            "learned_scaled_minus_threshold_caught_mean": float(np.mean(
                [r["caught"] - t["caught"] for r, t in zip(rows["learned_scaled"], thr)])),
            "learned_minus_threshold_caught_mean": float(np.mean(
                [r["caught"] - t["caught"] for r, t in zip(rows["learned"], thr)])),
        }
    n_conv_main = dict(CONV["main"])
    log(f"\nConvergenceWarnings during main runs, per arm: {n_conv_main}")

    bs = args.boot_seeds
    point_rows = main_rows[args.boot_budget]
    pt_s = [r["admitted"] - t["admitted"] for r, t in zip(point_rows["learned_scaled"], point_rows["threshold"])]
    pt_u = [r["admitted"] - t["admitted"] for r, t in zip(point_rows["learned"], point_rows["threshold"])]
    log(f"\npoint estimate at budget {args.boot_budget:.2f}, seeds 0-{args.seeds - 1}: "
        f"learned_scaled-thr admitted {np.mean(pt_s):+.4f}, learned-thr admitted {np.mean(pt_u):+.4f}")
    log(f"point estimate at budget {args.boot_budget:.2f}, seeds 0-{bs - 1}: "
        f"learned_scaled-thr admitted {np.mean(pt_s[:bs]):+.4f}, learned-thr admitted {np.mean(pt_u[:bs]):+.4f}")

    # control: the unscaled learned arm must reproduce E03's per-seed counts at 25%
    with open(E03_RESULT) as f:
        e03 = json.load(f)["per_budget"][f"{args.boot_budget:.2f}"]["arms"]["learned"]
    ours = arm_stats(point_rows["learned"])
    control = {
        "e03_result": E03_RESULT,
        "e03_learned_caught_per_seed": e03["caught_per_seed"],
        "e03_learned_caught_mean": e03["caught_mean"],
        "ours_learned_caught_per_seed": ours["caught_per_seed"],
        "ours_learned_caught_mean": ours["caught_mean"],
        "reproduced": ours["caught_per_seed"] == e03["caught_per_seed"],
    }
    log(f"\ncontrol: unscaled learned at 25% caught per seed {ours['caught_per_seed']} mean {ours['caught_mean']:.1f}; "
        f"E03 {e03['caught_per_seed']} mean {e03['caught_mean']:.1f}; reproduced={control['reproduced']}")

    boot = None
    if args.boot > 0:
        log(f"\nbootstrap: B={args.boot} resamples of {n} groups with replacement (default_rng(10000+b)), "
            f"seeds 0-{bs - 1}, budget {args.boot_budget:.2f}")
        d_s, d_u = bootstrap(pool, args.boot_budget, list(range(bs)), args.boot, log)
        boot = {"B": args.boot, "seeds": bs, "budget": args.boot_budget,
                "learned_scaled_minus_threshold_admitted": ci95(d_s),
                "learned_minus_threshold_admitted": ci95(d_u)}
    n_conv_boot = dict(CONV["boot"])
    log(f"ConvergenceWarnings during bootstrap, per arm: {n_conv_boot}")

    verdict = None
    kill_attribution = None
    scaled_warn = n_conv_main["learned_scaled"] + n_conv_boot["learned_scaled"]
    if boot is not None:
        ci = boot["learned_scaled_minus_threshold_admitted"]["ci95"]
        if not control["reproduced"]:
            verdict = "INVALID_CONTROL"
        else:
            verdict = "PASS" if ci[1] < 0 else "KILL"
        log(f"\nCI95 (learned_scaled - threshold) admitted at {args.boot_budget:.2f}: {ci} -> {verdict}")
        if verdict == "KILL":
            kill_attribution = ("not an artifact of the unscaled fit: scaled fits logged zero ConvergenceWarnings"
                                if scaled_warn == 0 else
                                f"kill with non-convergence caveat: scaled fits logged {scaled_warn} ConvergenceWarnings")

    result = {
        "id": "L3b",
        "pool": args.pool,
        "n_groups": n,
        "n_proxy_errors": n_err,
        "step_size_groups": STEP,
        "budgets": args.budgets,
        "seeds": args.seeds,
        "assumptions": [
            "steps of 16 groups over a seed-permuted pool (last step 8 groups); n_verify(step_groups, budget, step=t) with cumulative rounding",
            "feat_vec = same 22 dims as E03: [agreement, n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen] + pc0..pc15",
            "cell = cell100 (KMeans k=100 over question embeddings from p0_analyze.py)",
            "learned: LogisticRegression(max_iter=1000) unscaled, as in avrl/allocators.py (control)",
            "learned_scaled: Pipeline(StandardScaler, LogisticRegression(max_iter=1000)) refit on cumulative verified X and y at the same cadence (refit_every=20) as learned",
            "LearnedScaled.update() copies LearnedAllocator.update() except the fit line; select(), Thompson step and threshold cold start are inherited",
            "permutation rng default_rng(1000+seed); selection rng default_rng(2000+seed) shared across steps of one run",
            "gold (proxy_error) used only for scoring and as the update label of verified groups",
            "ConvergenceWarnings counted per arm with warnings.catch_warnings(record=True) around each run_arm call",
        ],
        "per_budget": {f"{b:.2f}": v for b, v in per_budget.items()},
        "control_vs_e03": control,
        "metric": {
            "name": "admitted rate at 25% budget, learned_scaled minus threshold, paired per bootstrap resample, "
                    "mean over seeds, percentile 95% CI over B resamples",
            "budget": args.boot_budget,
            "point_estimate_seeds_0_to_7": {
                "learned_scaled_minus_threshold_admitted": float(np.mean(pt_s)),
                "learned_minus_threshold_admitted": float(np.mean(pt_u)),
            },
            "point_estimate_seeds_0_to_3": {
                "learned_scaled_minus_threshold_admitted": float(np.mean(pt_s[:bs])),
                "learned_minus_threshold_admitted": float(np.mean(pt_u[:bs])),
            },
            "bootstrap": boot,
        },
        "verdict": verdict,
        "verdict_rule": "PASS iff the 95% CI upper bound of (learned_scaled admitted - threshold admitted) at the boot budget is < 0 "
                        "and the unscaled control reproduces E03; otherwise KILL (CI upper bound >= 0) or INVALID_CONTROL",
        "kill_attribution": kill_attribution,
        "convergence_warnings": {
            "main": n_conv_main,
            "bootstrap": n_conv_boot,
            "learned_scaled_total": scaled_warn,
            "learned_total": n_conv_main["learned"] + n_conv_boot["learned"],
        },
        "convergence_note": "E03 logged 640 ConvergenceWarnings: 23 in its main run (learned and greedy arms, budgets 0.10, 0.25, 0.50) "
                            "and 617 in its bootstrap (learned and greedy arms). Those counts are not directly comparable to the per-arm "
                            "counts here, which cover only the learned variants at 25%.",
        "wall_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    log(f"\nwrote {os.path.join(args.out, 'result.json')} (wall {result['wall_s']} s)")


if __name__ == "__main__":
    main()
