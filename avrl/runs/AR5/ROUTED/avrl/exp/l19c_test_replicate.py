"""L19c: frozen config B on the registered test rows of a fresh GSM8K rollout replicate (pids 0 to 1318). CPU only.

Copied from avrl/exp/l6a_test_rows.py (L6a is unchanged). The fit, frozen-table, split, tie-break and spend code is the L6a
code. Changes, all registered in avrl/RESEARCH_LOG.md (L19c, 2026-10-19) before the run:
  - the L5a stand-in reproduction gate is replaced by a self-check: plain fixed at 10% and the 3,000-label gap on the test
    rows are recomputed in a second pass and must match the first within 0.05 pp (the L4b values do not apply to fresh
    rollouts, so they are not compared);
  - the verdict is on the 200-label tables only: PASS if the mean of the 20 subset means is above +1.0 pp and inside
    [p5, p95] of the 48 L16a L2b cut means (read from runs/L16a/result.json at run time); KILL if it is at or below +1.0 pp
    or below p5; INCONCLUSIVE if above p95;
  - pool_summary no longer carries a per-pool verdict (the L6a version used the +2.0 pp rule);
  - context only, outside the verdict: the 3,000-label gap on the test rows, and the 200-label gap on pids above
    --eval_pid_max (the remaining train rows), using the same 20 frozen tables and seeds;
  - raw hit counts per subset and split, and for the 3,000-label table, are written to result.json for the critic check
    (avrl/exp/l19c_check.py). The 64 random-half context of L6a is dropped.

Gates, before any verdict (a failed gate writes verdict GATE_FAILED and exits 1):
  1. Self-check: plain fixed at 10% and the 3,000-label gap on the test rows match the second pass within 0.05 pp, and
     the fixed rate lies in (0, 100).
  2. Spend: every policy takes exactly int(budget x n_sim) distinct docs on every split, seed and table (0 failures).
  3. Each subset has exactly subset_size distinct prompts, fixed picks are identical across all tables, and no test
     question text occurs in p0.

Usage: python -I l19c_test_replicate.py --fit_feats P0 --eval_feats runs/L19c/feats.parquet --eval_pid_max 1318 --budget 0.10
           --k0 50 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8 --out OUTDIR
Writes <out>/result.json and <out>/table.md. A failed gate writes result.json with verdict GATE_FAILED and exits 1.
"""
import os, sys, json, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import l1b_tuned_threshold as L1B

RUNS = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
P0_FEATS, L19C_FEATS = f"{RUNS}/p0/feats.parquet", f"{RUNS}/L19c/feats.parquet"
L16A_RESULT, L6A_RESULT = f"{RUNS}/L16a/result.json", f"{RUNS}/L6a/result.json"
TEST_PID_MAX = 1318  # registered test rows: pids 0 to 1318, 1,319 prompts
LABEL, GROUP = "proxy_error", "cell100"
STRATUM = ["agreement", "n_distinct", "any_truncated"]
GATE_TOL_PP = 0.05
VERDICT_PP = 1.0  # registered L19c pass and kill level
SUBSET_SEED0 = 100


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def parse_splits(s):
    if "-" in s:
        lo, hi = s.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def l16a_band():
    """5th and 95th percentiles of the 48 L16a L2b cut means (200-label tables), read from the saved result at run time."""
    with open(L16A_RESULT) as fh: cuts = json.load(fh)["train_cuts"]["values_pp"]
    assert len(cuts) == 48, len(cuts)
    return float(np.percentile(cuts, 5)), float(np.percentile(cuts, 95)), len(cuts)


def verdict(mean_pp, p5, p95):
    if mean_pp <= VERDICT_PP or mean_pp < p5: return "KILL"
    if mean_pp > p95: return "INCONCLUSIVE"
    return "PASS"


def fit_table(fit_df, k0):
    """Frozen table: posterior error rate per stratum from every labelled prompt of the fit pool. Never refit."""
    g = fit_df.assign(_y=fit_df[LABEL].to_numpy(int)).groupby(STRATUM)["_y"].agg(["sum", "size"]).reset_index()
    g = g.rename(columns={"sum": "errors", "size": "count"})
    g["prior"] = 1 - g["agreement"]
    g["post"] = (g["errors"] + k0 * g["prior"]) / (g["count"] + k0)
    return g


def attach_score(df, table):
    """Frozen score per doc: the table posterior of its stratum. A stratum missing from the table scores its prior."""
    for c in STRATUM: assert not df[c].isna().any(), c
    m = df[STRATUM].merge(table[STRATUM + ["post"]], on=STRATUM, how="left", validate="many_to_one")
    assert len(m) == len(df)
    post = m["post"].to_numpy(float); prior = 1 - df["agreement"].to_numpy(float)
    return df.assign(frozen_score=np.where(np.isnan(post), prior, post), in_table=~np.isnan(post))


def fixed_pick(S, budget, seed):
    rnd = np.random.default_rng(seed).random(S.n)
    return np.lexsort((rnd, -S.nd, S.ag))[:budget]


def frozen_pick(S, score, budget, seed):
    rnd = np.random.default_rng(seed).random(S.n)
    return np.lexsort((rnd, -S.nd, S.ag, -score))[:budget]


def job(args):
    df_in, split, table, frac, seeds = args
    df = attach_score(df_in, table)
    sim, n_groups = L1B.build_sim(df, GROUP, split)
    S = L1B.Sim(sim, LABEL)
    score = sim["frozen_score"].to_numpy(float)
    budget = int(frac * S.n)
    fixed, frozen, spend_bad = [], [], []
    for s in seeds:
        sel_f, sel_z = fixed_pick(S, budget, s), frozen_pick(S, score, budget, s)
        hf = int(S.y[sel_f].sum())
        assert hf == S.fixed(budget, s), "fixed_pick disagrees with L1B.Sim.fixed"
        fixed.append(hf); frozen.append(int(S.y[sel_z].sum()))
        for name, sel in (("fixed", sel_f), ("frozen", sel_z)):
            d = int(len(np.unique(sel)))
            if len(sel) != budget or d != budget:
                spend_bad.append({"policy": name, "seed": int(s), "distinct": d, "expected": budget})
    return {"split": split, "n_sim": S.n, "n_buckets": int(sim[GROUP].nunique()), "n_groups": n_groups,
            "hits_total": int(S.y.sum()), "budget": budget, "fixed_hits": fixed, "frozen_hits": frozen,
            "spend_bad": spend_bad, "in_table_frac": float(sim["in_table"].mean())}


def run_pool(df, table, frac, splits, seeds):
    with Pool(min(len(splits), os.cpu_count())) as p:
        res = p.map(job, [(df, s, table, frac, seeds) for s in splits])
    return {r["split"]: r for r in res}


def pool_summary(R, splits):
    gaps, fixed_pct, frozen_pct = [], [], []
    for s in splits:
        r = R[s]; b = r["budget"]
        f = 100.0 * np.asarray(r["fixed_hits"], float) / b; z = 100.0 * np.asarray(r["frozen_hits"], float) / b
        gaps.append(float((z - f).mean())); fixed_pct.append(float(f.mean())); frozen_pct.append(float(z.mean()))
    st = L1B.stats(gaps)
    return {"n_sim_per_split": [R[s]["n_sim"] for s in splits], "budget_per_split": [R[s]["budget"] for s in splits],
            "gap_pp_per_split": gaps, "fixed_pct_per_split": fixed_pct, "frozen_pct_per_split": frozen_pct,
            "in_table_frac_per_split": [R[s]["in_table_frac"] for s in splits],
            "mean_pp": st["mean_pp"], "sd_pp": st["sd_pp"], "n_positive": st["n_positive"]}


def self_check(first, second):
    """Plain fixed at 10% and the 3,000-label gap on the test rows, from a first and a second pass over the same rows."""
    fx, gp = float(np.mean(first["fixed_pct_per_split"])), float(first["mean_pp"])
    fx2, gp2 = float(np.mean(second["fixed_pct_per_split"])), float(second["mean_pp"])
    d = max(abs(fx - fx2), abs(gp - gp2))
    return {"fixed_pct": fx, "gap_pp": gp, "second_pass_fixed_pct": fx2, "second_pass_gap_pp": gp2,
            "max_abs_diff_pp": float(d), "tol_pp": GATE_TOL_PP, "pass": bool(d <= GATE_TOL_PP and 0.0 < fx < 100.0)}


def spend_gate(pools):
    bad = [dict(pool=name, split=s, **b) for name, R in pools.items() for s in R for b in R[s]["spend_bad"]]
    checks = sum(2 * len(R[next(iter(R))]["fixed_hits"]) * len(R) for R in pools.values())
    return {"checks": checks, "failures": len(bad), "examples": bad[:5], "pass": not bad}


def draw_subset(fit_df, i, size):
    """Row positions of label subset i: size distinct p0 prompts, numpy RandomState(100 + i), without replacement."""
    return np.random.RandomState(SUBSET_SEED0 + i).choice(len(fit_df), size=size, replace=False)


def raw_hits(R, splits):
    return {str(s): {"n_sim": R[s]["n_sim"], "budget": R[s]["budget"], "fixed_hits": R[s]["fixed_hits"],
                     "frozen_hits": R[s]["frozen_hits"]} for s in splits}


def write_json(out_dir, out):
    with open(os.path.join(out_dir, "result.json"), "w") as fh: json.dump(out, fh, indent=1, default=_jsonable)


def write_table(out_dir, r):
    t2, c, g = r["table_200"], r["context_only"], r["gates"]
    t3, rm = c["table_3000_test"], c["table_200_remaining"]
    rp, sg, ref = g["test_row_self_check"], g["spend_gate"], r["reference"]
    p5, p95 = ref["L16a_cut_p5_pp"], ref["L16a_cut_p95_pp"]
    L = ["# L19c: frozen config B on the 1,319 registered test rows, fresh rollouts (sampling seed 2)", "",
         f"Rule: frozen gap is frozen B minus plain fixed at 10%, paired by split seed and tie-break seed, over split seeds 0 to 7 "
         f"({r['tie_seeds']} tie-break seeds each). A 200-label subset mean is the mean of its 8 split gaps. The verdict value is "
         f"the mean of the {t2['n_subsets']} subset means. PASS if above +{VERDICT_PP:.1f} pp and inside [{p5:+.2f}, {p95:+.2f}] pp "
         f"(the 5th and 95th percentiles of the 48 L16a cut means). KILL if at or below +{VERDICT_PP:.1f} pp or below {p5:+.2f} pp. "
         f"INCONCLUSIVE if above {p95:+.2f} pp. Test rows: {r['test_prompts']} prompts, proxy error "
         f"{100 * r['test_error_rate']:.1f}%.", "",
         "## Verdict", "",
         f"- 200-label tables (pids 0 to {r['eval_pid_max']}): **{t2['verdict']}**, mean of {t2['n_subsets']} subset means "
         f"{t2['mean_pp']:+.2f} pp (sd over subset means {t2['sd_pp']:.2f}, {t2['n_subsets_above_verdict']} of {t2['n_subsets']} "
         f"subsets above +{VERDICT_PP:.1f})", "",
         "## Gates", "", "| gate | result |", "|---|---|",
         f"| self-check: plain fixed at 10% and the 3,000-label gap recomputed in a second pass (fixed {rp['fixed_pct']:.4f}%, "
         f"gap {rp['gap_pp']:+.4f} pp) | {'PASS' if rp['pass'] else 'FAIL'} (max abs diff {rp['max_abs_diff_pp']:.6f} pp, "
         f"tol {GATE_TOL_PP}) |",
         f"| spend gate | {'PASS' if sg['pass'] else 'FAIL'} ({sg['checks']} checks, failures {sg['failures']}) |",
         f"| every subset has {r['subset_size']} distinct prompts | {'PASS' if g['subset_distinct_prompts'] else 'FAIL'} |",
         f"| fixed picks identical across all tables | {'PASS' if g['fixed_identical_across_tables'] else 'FAIL'} |",
         f"| test question text found in p0 | {'PASS (0 rows)' if g['eval_question_overlap_zero'] else 'FAIL'} |", "",
         "## Metrics and references", "",
         "| quantity | value |", "|---|---|",
         f"| 200-label, mean of subset means: gain, fixed at 10%, frozen at 10% | {t2['mean_pp']:+.2f} pp, "
         f"{t2['fixed_pct_mean']:.2f}%, {t2['frozen_pct_mean']:.2f}% |",
         f"| 200-label, sd over subset means; min and max subset mean | {t2['sd_pp']:.2f}; {t2['min_subset_mean_pp']:+.2f} and "
         f"{t2['max_subset_mean_pp']:+.2f} pp |",
         f"| 48 L16a cut means, p5 and p95 (the band) | {p5:+.2f} and {p95:+.2f} pp |",
         f"| L6a, same questions, old rollouts, 200-label | {ref['L6a_table_200_test_pp']:+.2f} pp |",
         f"| L6a, same questions, old rollouts, 3,000-label | {ref['L6a_table_3000_test_pp']:+.2f} pp |", "",
         "## Subsets (200-label tables)", "",
         "| subset | strata in table | gain, pids 0 to 1318 (pp) | gain, pids above 1318 (pp) | docs in table (%) |",
         "|---|---|---|---|---|"]
    for s in r["subset_detail"]:
        L.append(f"| {s['subset']} | {s['strata_in_table']} | {s['mean_pp']:+.2f} | {s['remain_mean_pp']:+.2f} | "
                 f"{100 * s['in_table_frac']:.1f} |")
    L += ["", "## Context only (outside the verdict)", "",
          "| set | prompts | proxy error | gain, mean (pp) | sd (pp) | above +1.0 |", "|---|---|---|---|---|---|",
          f"| 3,000-label table, pids 0 to {r['eval_pid_max']} (mean over split seeds 0 to 7) | {r['test_prompts']} | "
          f"{100 * r['test_error_rate']:.1f}% | {t3['mean_pp']:+.2f} | {t3['sd_pp']:.2f} | {t3['n_splits_above_1pp']} of "
          f"{len(r['splits'])} split gaps |",
          f"| 200-label tables, pids above {r['eval_pid_max']} (mean of {rm['n_subsets']} subset means) | {rm['prompts']} | "
          f"{100 * rm['error_rate']:.1f}% | {rm['mean_pp']:+.2f} | {rm['sd_pp']:.2f} | {rm['n_subsets_above_1pp']} of "
          f"{rm['n_subsets']} subsets |", "",
          "## Notes", "",
          "- Test rows are pids 0 to 1318 of runs/L19c/feats.parquet, the same questions as runs/L4b pids 0 to 1318. The rollouts "
          "are fresh: avrl/exp/l4b_rollouts_pool3.py with --G 8 --seed 2.",
          "- Subset i uses numpy RandomState(100 + i).choice(3000, 200, replace=False) over the p0 rows. Each subset table is "
          "fitted on its 200 prompts only and frozen.",
          "- The cell100 split (L1B.build_sim, MIN 3, bucket shuffle seed = split seed) runs on each evaluation frame after its "
          "own filter, as in L6a.",
          "- Per-split gaps, raw hit counts per tie seed, n_sim and budget for every subset, and for the 3,000-label table, are "
          "in result.json.", ""]
    with open(os.path.join(out_dir, "table.md"), "w") as fh: fh.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_feats", default=P0_FEATS)
    ap.add_argument("--eval_feats", default=L19C_FEATS)
    ap.add_argument("--eval_pid_max", type=int, default=TEST_PID_MAX)
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--k0", type=float, default=50.0)
    ap.add_argument("--subsets", type=int, default=20)
    ap.add_argument("--subset_size", type=int, default=200)
    ap.add_argument("--splits", default="0-7")
    ap.add_argument("--tie_seeds", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    splits, seeds = parse_splits(a.splits), list(range(a.tie_seeds))
    p5, p95, n_cuts = l16a_band()
    with open(L6A_RESULT) as fh: l6a = json.load(fh)

    fit_df = pd.read_parquet(a.fit_feats)
    ev_all = pd.read_parquet(a.eval_feats).reset_index(drop=True)
    test = ev_all[ev_all["pid"] <= a.eval_pid_max].reset_index(drop=True)
    remain = ev_all[ev_all["pid"] > a.eval_pid_max].reset_index(drop=True)
    overlap = int(test["question"].isin(set(fit_df["question"])).sum())
    print(f"test rows {len(test)} (pid <= {a.eval_pid_max}), remaining rows {len(remain)}, test proxy error "
          f"{test[LABEL].mean():.3f}, p0 fit prompts {len(fit_df)}, test question overlap with p0 {overlap}", flush=True)
    print(f"L16a band from {n_cuts} cut means: p5 {p5:+.4f} pp, p95 {p95:+.4f} pp", flush=True)

    full_table = fit_table(fit_df, a.k0)
    print(f"gate table: {len(full_table)} strata from {len(fit_df)} p0 prompts, k0 {a.k0:g}, no refit", flush=True)
    full_runs = run_pool(test, full_table, a.budget, splits, seeds)
    full_sum = pool_summary(full_runs, splits)
    again_runs = run_pool(test, full_table, a.budget, splits, seeds)
    again_sum = pool_summary(again_runs, splits)
    repro = self_check(full_sum, again_sum)
    print(f"test-row self-check: fixed at 10% {repro['fixed_pct']:.4f}% (second pass {repro['second_pass_fixed_pct']:.4f}%), "
          f"3,000-label gap {repro['gap_pp']:+.4f} pp (second pass {repro['second_pass_gap_pp']:+.4f}), max abs diff "
          f"{repro['max_abs_diff_pp']:.6f} pp (tol {GATE_TOL_PP}), pass {repro['pass']}", flush=True)
    if not repro["pass"]:
        write_json(a.out, {"experiment": "L19c", "verdict": "GATE_FAILED", "stage": "test-row self-check",
                           "gates": {"test_row_self_check": repro}, "table_3000_test": full_sum})
        print("GATE FAILED at the test-row self-check: no L19c verdict written", flush=True); sys.exit(1)

    subs = []
    for i in range(a.subsets):
        idx = draw_subset(fit_df, i, a.subset_size)
        sub = fit_df.iloc[idx]
        table = fit_table(sub, a.k0)
        R = run_pool(test, table, a.budget, splits, seeds)
        Rr = run_pool(remain, table, a.budget, splits, seeds)
        summ, rsumm = pool_summary(R, splits), pool_summary(Rr, splits)
        subs.append({"subset": i, "row_positions": [int(x) for x in idx], "distinct_pids": int(sub["pid"].nunique()),
                     "strata_in_table": int(len(table)), "runs": R, "summary": summ, "remain_runs": Rr, "remain_summary": rsumm})
        print(f"subset {i:2d}: {len(table)} strata, pids 0 to {a.eval_pid_max} {summ['mean_pp']:+.2f} pp, "
              f"pids above {a.eval_pid_max} {rsumm['mean_pp']:+.2f} pp", flush=True)

    fixed_same = all(x["runs"][s]["fixed_hits"] == full_runs[s]["fixed_hits"] for x in subs for s in splits)
    sg = spend_gate({"test/3000": full_runs, "test/3000_second_pass": again_runs,
                     **{f"test/subset{x['subset']}": x["runs"] for x in subs},
                     **{f"remaining/subset{x['subset']}": x["remain_runs"] for x in subs}})
    subset_ok = all(x["distinct_pids"] == a.subset_size and len(set(x["row_positions"])) == a.subset_size for x in subs)
    overlap_ok = overlap == 0
    gates_ok = bool(sg["pass"] and subset_ok and overlap_ok and fixed_same and repro["pass"])
    gates = {"test_row_self_check": repro, "spend_gate": sg, "subset_distinct_prompts": bool(subset_ok),
             "fixed_identical_across_tables": bool(fixed_same), "eval_question_overlap_zero": bool(overlap_ok),
             "pass": gates_ok}
    print(f"spend gate: {sg['checks']} checks, failures {sg['failures']}; subsets distinct {subset_ok}; "
          f"fixed identical {fixed_same}; test question overlap with p0 {overlap}; gates {'PASS' if gates_ok else 'FAIL'}",
          flush=True)
    if not gates_ok:
        write_json(a.out, {"experiment": "L19c", "verdict": "GATE_FAILED", "gates": gates})
        print("GATE FAILED: no L19c verdict written", flush=True); sys.exit(1)

    means = [x["summary"]["mean_pp"] for x in subs]
    st = L1B.stats(means)
    rem_means = [x["remain_summary"]["mean_pp"] for x in subs]
    rst = L1B.stats(rem_means)
    gaps_all = np.asarray([g_ for x in subs for g_ in x["summary"]["gap_pp_per_split"]], float)
    v = verdict(st["mean_pp"], p5, p95)
    t2 = {"n_subsets": len(means), "subset_mean_pp": means, "mean_pp": st["mean_pp"], "sd_pp": st["sd_pp"],
          "n_subsets_above_verdict": int(sum(m > VERDICT_PP for m in means)),
          "min_subset_mean_pp": float(min(means)), "max_subset_mean_pp": float(max(means)),
          "split_gap_mean_pp": float(gaps_all.mean()), "split_gap_sd_pp": float(gaps_all.std(ddof=1)),
          "fixed_pct_mean": float(np.mean([np.mean(x["summary"]["fixed_pct_per_split"]) for x in subs])),
          "frozen_pct_mean": float(np.mean([np.mean(x["summary"]["frozen_pct_per_split"]) for x in subs])),
          "gap_pp_per_subset": [x["summary"]["gap_pp_per_split"] for x in subs], "verdict": v}
    t3 = {"mean_pp": full_sum["mean_pp"], "sd_pp": full_sum["sd_pp"], "gap_pp_per_split": full_sum["gap_pp_per_split"],
          "n_splits_above_1pp": int(sum(g_ > VERDICT_PP for g_ in full_sum["gap_pp_per_split"])),
          "fixed_pct_mean": float(np.mean(full_sum["fixed_pct_per_split"])),
          "frozen_pct_mean": float(np.mean(full_sum["frozen_pct_per_split"])),
          "in_table_frac_mean": float(np.mean(full_sum["in_table_frac_per_split"]))}
    rm = {"prompts": int(len(remain)), "error_rate": float(remain[LABEL].mean()), "n_subsets": len(rem_means),
          "subset_mean_pp": rem_means, "mean_pp": rst["mean_pp"], "sd_pp": rst["sd_pp"],
          "n_subsets_above_1pp": int(sum(m > VERDICT_PP for m in rem_means))}
    reference = {"L6a_table_200_test_pp": l6a["table_200"]["mean_pp"], "L6a_table_3000_test_pp": l6a["table_3000"]["mean_pp"],
                 "L16a_cut_p5_pp": p5, "L16a_cut_p95_pp": p95, "L16a_cut_n": n_cuts}
    detail = [{"subset": x["subset"], "row_positions": x["row_positions"], "distinct_pids": x["distinct_pids"],
               "strata_in_table": x["strata_in_table"], "mean_pp": x["summary"]["mean_pp"],
               "gap_pp_per_split": x["summary"]["gap_pp_per_split"], "remain_mean_pp": x["remain_summary"]["mean_pp"],
               "in_table_frac": float(np.mean(x["summary"]["in_table_frac_per_split"])), "raw": raw_hits(x["runs"], splits)}
              for x in subs]
    out = {"experiment": "L19c", "metric": "200-label frozen minus plain fixed at 10%, mean of the 20 subset means, pids 0 to "
           f"{a.eval_pid_max}", "verdict": v, "value_pp": st["mean_pp"],
           "verdict_rule": f"PASS if above +{VERDICT_PP:.1f} pp and inside [p5, p95] of the 48 L16a cut means; KILL if at or below "
           f"+{VERDICT_PP:.1f} pp or below p5; INCONCLUSIVE if above p95",
           "verdict_pp": VERDICT_PP, "label": LABEL, "group": GROUP, "stratum": STRATUM, "k0": a.k0, "budget_frac": a.budget,
           "fit_feats": a.fit_feats, "eval_feats": a.eval_feats, "eval_pid_max": a.eval_pid_max,
           "test_prompts": int(len(test)), "test_error_rate": float(test[LABEL].mean()),
           "fit_prompts": int(len(fit_df)), "subsets": a.subsets, "subset_size": a.subset_size,
           "subset_draw": "numpy RandomState(100 + i).choice(n_p0_rows, subset_size, replace=False)",
           "refit": "none (each table is frozen after its fit)", "seeds": seeds, "splits": splits, "sd_ddof": 1,
           "tie_seeds": a.tie_seeds, "eval_question_overlap_with_p0": overlap, "gates": gates,
           "table_200": t2, "context_only": {"table_3000_test": t3, "table_200_remaining": rm},
           "reference": reference, "subset_detail": detail, "raw_3000_test": raw_hits(full_runs, splits)}
    write_json(a.out, out); write_table(a.out, out)
    print(f"200-label tables on test rows: mean of subset means {st['mean_pp']:+.2f} pp, sd {st['sd_pp']:.2f}, verdict {v} "
          f"(band {p5:+.2f} to {p95:+.2f} pp, value rule above +{VERDICT_PP:.1f})", flush=True)
    print(f"context: 3,000-label table on test rows {t3['mean_pp']:+.2f} pp; 200-label tables on pids above {a.eval_pid_max} "
          f"{rst['mean_pp']:+.2f} pp (sd {rst['sd_pp']:.2f})", flush=True)
    print(f"wrote {os.path.join(a.out, 'result.json')} and table.md", flush=True)


if __name__ == "__main__":
    main()
