#!/usr/bin/env python3
"""L3e: is the logit_noise caught deficit a generic cost of ranking noise?

Derived from avrl/exp/l3a_thompson_ablation.py (same pool, permutation, step size, selection RNG streams,
E03 reference and gate-before-outcome rule). Budget 0.25 only. Control run: no bootstrap, no CI.

Question. E03's noiseless regression ranking (greedy) caught 321.0 on average. logit_noise (the same
regression ranking plus N(0, 1) noise on the logit scale, from L3a) caught 302.6. The noise increment is
321.0 - 302.6 = 18.4. The regression ranking gap (greedy 321.0 vs threshold 334.25) is not noise and is not
used as the comparator. Does Gaussian noise on the threshold ranking, scaled so its within-step rank
displacement matches logit_noise's, cost a comparable amount?

Arms:
  threshold        ThresholdAllocator, noiseless baseline. Gate: reproduces E03 per seed.
  logit_noise      as in L3a (sigma 1.0, same code path and RNG draws). Gate: reproduces L3a per seed.
  threshold_noise  rank by agreement + N(0, s) with the n_distinct tie-break, no learning, no cells.
                   s = 0 is gate (a): must equal ThresholdAllocator. The main arm uses the s chosen by the
                   displacement pilot (seed 0, 25%, displacement only).

Displacement: within each step, rank every candidate under the noiseless order and under the noisy order,
record mean |rank_noisy - rank_clean|, then average over the steps where noise is in use. logit_noise is
noiseless during its cold start (before the first fit), so those steps are excluded for it. Computed from
arrays already in hand, with no extra RNG draws.

Pool: runs/p0/feats.parquet (3000 groups, 8 rollouts each), read only. Gold (proxy_error) is used only for
scoring and as the update label of verified groups, never as a selection input.

CPU only. No GPU, no pip, no ssh. Writes result.json into --out; the runner redirects output to run.log.
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
L3A_RESULT = os.path.join(AVRL_DIR, "runs", "L3a", "result.json")
sys.path.insert(0, AVRL_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import allocators as A  # noqa: E402

STEP = 16
BUDGET = 0.25
PILOT_SEED = 0
PC_COLS = [f"pc{i}" for i in range(16)]
FEAT_COLS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"] + PC_COLS
P_CLIP = 1e-3
LOGIT_SIGMA = 1.0
SPEC_THRESHOLD_CAUGHT_025 = [340, 341, 344, 332, 331, 328, 325, 333]
SPEC_THRESHOLD_MEAN = 334.25
SPEC_LOGIT_MEAN_1DP = 302.6
SPEC_GREEDY_MEAN = 321.0
SWEEP = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
GENERIC_MAX = 325.0
SPECIFIC_MIN = 330.0


def log(msg):
    print(msg, flush=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _probs(alloc, group_records):
    X = np.array([r["feat_vec"] for r in group_records])
    return np.clip(alloc.model.predict_proba(X)[:, 1], P_CLIP, 1 - P_CLIP)


def _rank(order):
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank


def _mean_displacement(clean_order, noisy_order):
    """Mean over the candidates of one step of |rank_noisy - rank_clean|."""
    return float(np.abs(_rank(noisy_order) - _rank(clean_order)).mean())


class LogitNoise(A.LearnedAllocator):
    """(c) Regression ranking with Gaussian noise on the logit scale. No cells. Same draws as L3a."""
    name = "logit_noise"

    def __init__(self):
        super().__init__()
        self.step_disp = []

    def select(self, group_records, k, rng):
        if k == 0:
            return []
        if self.model is None:
            return [r["idx"] for r in A.ThresholdAllocator._order(group_records)[:k]]
        p = _probs(self, group_records)
        clean = np.log(p / (1 - p))
        noisy = clean + rng.normal(0.0, LOGIT_SIGMA, size=len(p))
        clean_order = np.argsort(-clean, kind="stable")
        noisy_order = np.argsort(-noisy, kind="stable")
        self.step_disp.append(_mean_displacement(clean_order, noisy_order))
        idx = [r["idx"] for r in group_records]
        return [idx[i] for i in noisy_order[:k]]

    def update(self, verified):
        _update_no_cells(self, verified)


class ThresholdNoise(A.ThresholdAllocator):
    """Threshold ranking on agreement + N(0, scale), n_distinct tie-break. No learning, no cells."""
    name = "threshold_noise"

    def __init__(self, scale):
        self.scale = float(scale)
        self.step_disp = []

    def select(self, group_records, k, rng):
        if k == 0:
            return []
        agreement = np.array([r["agreement"] for r in group_records], dtype=np.float64)
        neg_distinct = -np.array([r["n_distinct"] for r in group_records], dtype=np.int64)
        noise = rng.normal(0.0, self.scale, size=len(group_records))
        # np.lexsort: last key is primary. Ascending: lowest (noisy) agreement first, then most distinct answers.
        clean_order = np.lexsort((neg_distinct, agreement))
        noisy_order = np.lexsort((neg_distinct, agreement + noise))
        self.step_disp.append(_mean_displacement(clean_order, noisy_order))
        idx = [r["idx"] for r in group_records]
        return [idx[i] for i in noisy_order[:k]]


def _update_no_cells(alloc, verified):
    """LearnedAllocator.update without cell_beta bookkeeping. Same accumulation and refit schedule as L3a."""
    for feat_vec, label, _cell in verified:
        alloc.X.append(feat_vec)
        alloc.y.append(label)
    n = len(alloc.y)
    due = n >= alloc.refit_every and n // alloc.refit_every > alloc.last_fit_n // alloc.refit_every
    if due and len(set(alloc.y)) > 1:
        alloc.model = LogisticRegression(max_iter=1000).fit(np.array(alloc.X), np.array(alloc.y))
        alloc.last_fit_n = n


def make_alloc(arm, scale):
    if arm == "threshold":
        return A.ThresholdAllocator()
    if arm == "logit_noise":
        return LogitNoise()
    if arm == "threshold_noise":
        return ThresholdNoise(scale)
    raise ValueError(arm)


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


def run_arm(pool, arm, budget, seed, scale=None):
    n = len(pool["y"])
    perm = np.random.default_rng(1000 + seed).permutation(n)
    alloc = make_alloc(arm, scale)
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
    disp = getattr(alloc, "step_disp", [])
    return {"arm": arm, "scale": scale, "budget": budget, "seed": seed, "errors": errors, "caught": caught,
            "admitted": (errors - caught) / errors, "calls": calls,
            "disp_mean": float(np.mean(disp)) if disp else None, "disp_steps": len(disp)}


def arm_stats(rows):
    out = {
        "caught_per_seed": [r["caught"] for r in rows],
        "caught_mean": float(np.mean([r["caught"] for r in rows])),
        "admitted_per_seed": [round(r["admitted"], 6) for r in rows],
        "admitted_mean": float(np.mean([r["admitted"] for r in rows])),
        "calls_per_seed": [r["calls"] for r in rows],
    }
    if rows[0]["disp_mean"] is not None:
        out["displacement_per_seed"] = [round(r["disp_mean"], 6) for r in rows]
        out["displacement_mean"] = float(np.mean([r["disp_mean"] for r in rows]))
    return out


def pilot(pool):
    """Displacement only. Caught counts of these runs are not read."""
    base = run_arm(pool, "logit_noise", BUDGET, PILOT_SEED)
    d_logit = base["disp_mean"]
    sweep = []
    for s in SWEEP:
        d = run_arm(pool, "threshold_noise", BUDGET, PILOT_SEED, scale=s)["disp_mean"]
        sweep.append({"s": s, "displacement": d})
    chosen = min(sweep, key=lambda row: abs(row["displacement"] - d_logit))
    gap = abs(chosen["displacement"] - d_logit) / d_logit
    log(f"pilot seed {PILOT_SEED}, budget {BUDGET}: logit_noise mean within-step displacement {d_logit:.4f} "
        f"over {base['disp_steps']} noisy steps")
    for row in sweep:
        log(f"  threshold_noise s={row['s']:<5} displacement {row['displacement']:.4f}")
    log(f"  chosen s={chosen['s']} displacement {chosen['displacement']:.4f}, relative gap {gap:.4f}, "
        f"within 10 pct: {gap <= 0.10}")
    return {"seed": PILOT_SEED, "logit_noise_displacement": d_logit,
            "logit_noise_noisy_steps": base["disp_steps"], "sweep": sweep,
            "chosen_s": chosen["s"], "chosen_displacement": chosen["displacement"],
            "relative_gap": gap, "within_10pct": bool(gap <= 0.10),
            "rule": "chosen s = argmin over SWEEP of |displacement(s) - logit_noise displacement| (seed 0)"}


def check_gates(thr_rows, s0_rows, logit_rows, e03, l3a, n_err):
    e = e03["per_budget"]["0.25"]
    ref = e["arms"]["threshold"]
    thr_caught = [r["caught"] for r in thr_rows]
    s0_caught = [r["caught"] for r in s0_rows]
    logit_caught = [r["caught"] for r in logit_rows]
    logit_mean = float(np.mean(logit_caught))
    checks = {
        "gate_a_threshold_caught_equals_spec": thr_caught == SPEC_THRESHOLD_CAUGHT_025,
        "gate_a_threshold_caught_equals_e03": thr_caught == ref["caught_per_seed"],
        "gate_a_threshold_admitted_equals_e03": [round(r["admitted"], 6) for r in thr_rows]
        == [round(x, 6) for x in ref["admitted_per_seed"]],
        "gate_a_errors_equal_e03": all(x == n_err for x in e["errors_per_seed"]),
        "gate_a_threshold_noise_s0_caught_equals_spec": s0_caught == SPEC_THRESHOLD_CAUGHT_025,
        "gate_a_threshold_noise_s0_calls_750": all(r["calls"] == 750 for r in s0_rows),
        "gate_a_threshold_noise_s0_displacement_zero": all(r["disp_mean"] == 0.0 for r in s0_rows),
        "gate_b_logit_noise_caught_mean_1dp_equals_302_6": round(logit_mean, 1) == SPEC_LOGIT_MEAN_1DP,
        "gate_b_logit_noise_caught_per_seed_equals_l3a":
            logit_caught == l3a["main_25"]["logit_noise"]["caught_per_seed"],
    }
    detail = {
        "threshold_caught": thr_caught,
        "threshold_noise_s0_caught": s0_caught,
        "logit_noise_caught": logit_caught,
        "logit_noise_caught_mean": logit_mean,
        "e03_threshold_caught": ref["caught_per_seed"],
        "l3a_logit_noise_caught": l3a["main_25"]["logit_noise"]["caught_per_seed"],
    }
    return checks, detail


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[BUDGET])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.budgets == [BUDGET], "L3e is defined at budget 0.25 only"
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "result.json")

    t0 = time.time()
    seeds = list(range(args.seeds))
    pool = load_pool(args.pool)
    n = len(pool["y"])
    n_err = int(pool["y"].sum())
    e03 = load_json(E03_RESULT)
    l3a = load_json(L3A_RESULT)
    greedy_caught = e03["per_budget"]["0.25"]["arms"]["greedy"]["caught_per_seed"]
    greedy_mean = float(np.mean(greedy_caught))
    log(f"L3e pool {args.pool}: groups={n} proxy_errors={n_err} step={STEP} "
        f"steps={math.ceil(n / STEP)} budget={BUDGET} verifier_calls_per_run={round(BUDGET * n)}")
    log(f"reference (E03 read, not rerun): greedy caught per seed {greedy_caught} mean {greedy_mean:.3f}")
    log(f"reference (L3a read, not rerun): logit_noise caught per seed "
        f"{l3a['main_25']['logit_noise']['caught_per_seed']}")

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")

        pil = pilot(pool)
        chosen_s = pil["chosen_s"]

        thr_rows = [run_arm(pool, "threshold", BUDGET, s) for s in seeds]
        s0_rows = [run_arm(pool, "threshold_noise", BUDGET, s, scale=0.0) for s in seeds]
        logit_rows = [run_arm(pool, "logit_noise", BUDGET, s) for s in seeds]
        gate_checks, gate_detail = check_gates(thr_rows, s0_rows, logit_rows, e03, l3a, n_err)
        log("gates: " + json.dumps(gate_checks))
        log("gate detail: " + json.dumps(gate_detail))
        if not all(gate_checks.values()):
            write_json(out_path, {"id": "L3e", "status": "GATE_FAILED", "pilot": pil,
                                  "gates": gate_checks, "gate_detail": gate_detail})
            sys.exit(f"STOP: gate failure, no main-run outcome read. checks {gate_checks} detail {gate_detail}")
        log("gates passed (threshold and logit_noise rows are reused as the main-run rows; "
            "same code, same seeds, deterministic)")

        tn_rows = [run_arm(pool, "threshold_noise", BUDGET, s, scale=chosen_s) for s in seeds]
        n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in wrec)

    thr_mean = float(np.mean([r["caught"] for r in thr_rows]))
    logit_mean = float(np.mean([r["caught"] for r in logit_rows]))
    tn_caught = [r["caught"] for r in tn_rows]
    tn_mean = float(np.mean(tn_caught))
    assert round(thr_mean, 2) == SPEC_THRESHOLD_MEAN, thr_mean
    for name, rows in (("threshold", thr_rows), ("logit_noise", logit_rows), ("threshold_noise", tn_rows)):
        log(f"  {name:15s} caught per seed {[r['caught'] for r in rows]} mean {np.mean([r['caught'] for r in rows]):.3f}"
            + (f" displacement mean {np.mean([r['disp_mean'] for r in rows]):.4f}" if rows[0]["disp_mean"] is not None else ""))
    log(f"  threshold_noise chosen s = {chosen_s}; ConvergenceWarnings during the run: {n_conv}")

    if tn_mean <= GENERIC_MAX:
        verdict = "GENERIC"
    elif tn_mean >= SPECIFIC_MIN:
        verdict = "REGRESSION-SPECIFIC"
    else:
        verdict = "INCONCLUSIVE"
    logit_increment = greedy_mean - logit_mean
    tn_cost = thr_mean - tn_mean
    log(f"threshold_noise caught mean {tn_mean:.3f}, cost {tn_cost:.3f} (= {SPEC_THRESHOLD_MEAN} minus caught)")
    log(f"logit_noise noise increment {greedy_mean:.1f} - {logit_mean:.3f} = {logit_increment:.3f}")
    log(f"verdict (readings fixed before run): {verdict}")

    result = {
        "id": "L3e",
        "status": "OK",
        "pool": args.pool,
        "n_groups": n,
        "n_proxy_errors": n_err,
        "step_size_groups": STEP,
        "budget": BUDGET,
        "verifier_calls_per_run": round(BUDGET * n),
        "seeds": args.seeds,
        "assumptions": [
            "steps of 16 groups over a seed-permuted pool (last step 8 groups); n_verify(step_groups, 0.25, step=t) as in E03",
            "feat_vec, cell and selection RNG streams as in L3a: permutation default_rng(1000+seed), selection default_rng(2000+seed)",
            "threshold_noise: key = agreement + N(0, s) drawn from the selection stream every step, order = np.lexsort((-n_distinct, key)); s = 0 equals ThresholdAllocator order",
            "logit_noise: as L3a; displacement recorded only on noisy steps (after the first fit)",
            "displacement: per step, mean over candidates of |rank_noisy - rank_clean|; averaged over noisy steps",
            "threshold_noise has no learning, so update is a no-op and its selections do not depend on labels",
            "main-run threshold and logit_noise rows are the gate runs (deterministic, same seeds)",
            "gold (proxy_error) used only for scoring and as the update label of verified groups",
            "greedy is E03's noiseless regression ranking, read from runs/E03/result.json, not rerun",
        ],
        "pilot": pil,
        "gates": gate_checks,
        "gate_detail": gate_detail,
        "main_25": {
            "threshold": arm_stats(thr_rows),
            "logit_noise": arm_stats(logit_rows),
            "threshold_noise": {**arm_stats(tn_rows), "chosen_s": chosen_s},
            "threshold_noise_s0_gate": arm_stats(s0_rows),
            "greedy_e03_reference": {
                "caught_per_seed": greedy_caught,
                "caught_mean": greedy_mean,
                "source": E03_RESULT,
            },
        },
        "metric": {
            "name": "cost = 334.25 minus caught, ThresholdNoise at the displacement-matched s, 25%, seeds 0-7",
            "threshold_noise_chosen_s": chosen_s,
            "threshold_noise_caught_per_seed": tn_caught,
            "threshold_noise_caught_mean": tn_mean,
            "threshold_noise_cost_mean": tn_cost,
            "threshold_noise_cost_per_seed": [round(thr_mean - c, 4) for c in tn_caught],
            "threshold_caught_mean": thr_mean,
            "logit_noise_caught_mean": logit_mean,
            "greedy_caught_mean": greedy_mean,
            "logit_noise_noise_increment": logit_increment,
            "matched_displacement_seed0": {
                "logit_noise": pil["logit_noise_displacement"],
                "threshold_noise_at_chosen_s": pil["chosen_displacement"],
                "relative_gap": pil["relative_gap"],
                "within_10pct": pil["within_10pct"],
            },
            "matched_displacement_mean_seeds_0_7": {
                "logit_noise": arm_stats(logit_rows).get("displacement_mean"),
                "threshold_noise_at_chosen_s": arm_stats(tn_rows).get("displacement_mean"),
            },
        },
        "verdict_rule": (f"GENERIC if ThresholdNoise caught mean <= {GENERIC_MAX} (noise cost >= 9.25); "
                         f"REGRESSION-SPECIFIC if >= {SPECIFIC_MIN} (noise cost <= 4.25); otherwise INCONCLUSIVE. "
                         "Control: not a pass/kill test, cannot change any claim verdict."),
        "verdict": verdict,
        "convergence_warnings": {"total": n_conv},
        "wall_s": round(time.time() - t0, 1),
    }
    write_json(out_path, result)
    log(f"\nwrote {out_path} (wall {result['wall_s']} s)")


if __name__ == "__main__":
    main()
