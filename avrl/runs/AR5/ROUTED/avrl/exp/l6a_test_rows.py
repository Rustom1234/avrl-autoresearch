"""L6a: frozen config B on the registered L4b test rows (pids 0 to 1318, 1,319 prompts). CPU only.

Copied from avrl/exp/l6b_frozen_label_budget.py (L6b is unchanged). Changes:
  - the L4b eval frame is cut to pid <= --eval_pid_max (default 1318) before the cell100 split, as the L4b and L5a critics
    did for the test rows, so the split and its MIN=3 bucket filter run on the test rows only;
  - the reproduction gate is the L5a critic stand-in on the test rows (plain fixed 84.50 percent, 3,000-label gap +3.39 pp,
    runs/L5a_critic/indep.log, line "L4b test only") instead of the full-pool references in runs/L2b and runs/L4b;
  - context only, outside the verdict: the 1,473 remaining-train rows (pid above --eval_pid_max) and 64 random 1,319-prompt
    halves of the full L4b pool, both scored with the 3,000-label table.
The fit, frozen-table, split, tie-break and spend code is the L6b code, unchanged.

Two tables are judged on the test rows, each against plain fixed at 10% on the same split and tie-break seed:
  - the 3,000-label table (fitted once on all p0 prompts): PASS if the mean gap over split seeds 0 to 7 is above +2.0 pp;
  - the 200-label tables (20 subsets of p0, numpy RandomState(100 + i), each frozen and never refit, as in L6b): PASS if the
    mean over the 20 subset means is above +2.0 pp.
A subset mean is the mean over its 8 split gaps. sd is ddof 1. Pass is above +2.0 pp, kill is at or below +2.0 pp.

Gates, before any verdict (a failed gate writes verdict GATE_FAILED and exits 1):
  1. Plain fixed at 10% on the test rows reproduces the L5a critic stand-in: mean fixed hit rate 84.50 percent and
     3,000-label gap +3.39 pp, each within 0.05 pp. Checked before any subset runs.
  2. Spend: every policy takes exactly int(budget x n_sim) distinct docs on every split, seed and table (0 failures).
  3. Each subset has exactly subset_size distinct prompts, fixed picks are identical across all tables, and no test
     question text occurs in p0.

Usage: python -I l6a_test_rows.py [--fit_feats P0] [--eval_feats L4B] [--eval_pid_max 1318] --budget 0.10 --k0 50
           --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8 --out OUTDIR
Writes <out>/result.json and <out>/table.md. A failed gate writes result.json with verdict GATE_FAILED and exits 1.
"""
import os, sys, json, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import l1b_tuned_threshold as L1B

RUNS = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
P0_FEATS, L4B_FEATS = f"{RUNS}/p0/feats.parquet", f"{RUNS}/L4b/feats.parquet"
TEST_PID_MAX = 1318  # registered test rows: pids 0 to 1318, 1,319 prompts
LABEL, GROUP = "proxy_error", "cell100"
STRATUM = ["agreement", "n_distinct", "any_truncated"]
GATE_TOL_PP = 0.05
# L5a critic stand-in on the test rows: plain fixed at 10% (8 splits, 8 tie seeds) and the 3,000-label frozen gap
L5A_TEST_SRC = f"{RUNS}/L5a_critic/indep.log"
L5A_TEST_FIXED_PCT, L5A_TEST_GAP_PP = 84.50, 3.39
SUBSET_SEED0 = 100
HALF_SEED0, N_HALVES, HALF_SIZE = 100, 64, 1319


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def parse_splits(s):
    if "-" in s:
        lo, hi = s.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


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
            "mean_pp": st["mean_pp"], "sd_pp": st["sd_pp"], "n_positive": st["n_positive"],
            "n_above_pass_pp": st["n_above_pass_pp"],
            "verdict": "PASS" if st["mean_pp"] > L1B.PASS_PP else "KILL"}


def test_gate(summary):
    """Plain fixed on the test rows against the L5a critic stand-in: mean fixed hit rate and mean 3,000-label gap."""
    fixed = float(np.mean(summary["fixed_pct_per_split"])); gap = float(summary["mean_pp"])
    d = max(abs(fixed - L5A_TEST_FIXED_PCT), abs(gap - L5A_TEST_GAP_PP))
    return {"ref": L5A_TEST_SRC, "ref_fixed_pct": L5A_TEST_FIXED_PCT, "mine_fixed_pct": fixed,
            "ref_gap_pp": L5A_TEST_GAP_PP, "mine_gap_pp": gap, "max_abs_diff_pp": float(d), "tol_pp": GATE_TOL_PP,
            "pass": bool(d <= GATE_TOL_PP)}


def spend_gate(pools):
    bad = [dict(pool=name, split=s, **b) for name, R in pools.items() for s in R for b in R[s]["spend_bad"]]
    checks = sum(2 * len(R[next(iter(R))]["fixed_hits"]) * len(R) for R in pools.values())
    return {"checks": checks, "failures": len(bad), "examples": bad[:5], "pass": not bad}


def draw_subset(fit_df, i, size):
    """Row positions of label subset i: size distinct p0 prompts, numpy RandomState(100 + i), without replacement."""
    return np.random.RandomState(SUBSET_SEED0 + i).choice(len(fit_df), size=size, replace=False)


def write_json(out_dir, out):
    with open(os.path.join(out_dir, "result.json"), "w") as fh: json.dump(out, fh, indent=1, default=_jsonable)


def write_table(out_dir, r):
    t3, t2, c, g = r["table_3000"], r["table_200"], r["context_only"], r["gates"]
    rp, sg = g["test_row_reproduction"], g["spend_gate"]
    h, rm = c["halves_3000"], c["remaining_train_3000"]
    L = ["# L6a: frozen config B on the 1,319 registered L4b test rows", "",
         f"Rule: frozen gap is frozen B minus plain fixed at 10%, paired by split seed and tie-break seed, over split seeds 0 to 7 "
         f"({r['tie_seeds']} tie-break seeds each). A 200-label subset mean is the mean of its 8 split gaps. A table passes if its "
         f"judged mean is above +{L1B.PASS_PP:.1f} pp, else it is killed. Test rows: {r['test_prompts']} prompts, proxy error "
         f"{100 * r['test_error_rate']:.1f}%.", "",
         "## Verdicts", "",
         f"- 3,000-label table: **{t3['verdict']}**, mean gap {t3['mean_pp']:+.2f} pp (sd {t3['sd_pp']:.2f}, "
         f"{t3['n_splits_above_pass']} of {len(r['splits'])} split gaps above +{L1B.PASS_PP:.1f})",
         f"- 200-label tables: **{t2['verdict']}**, mean of {t2['n_subsets']} subset means {t2['mean_pp']:+.2f} pp "
         f"(sd over subset means {t2['sd_pp']:.2f}, {t2['n_subsets_above_pass']} of {t2['n_subsets']} subsets above "
         f"+{L1B.PASS_PP:.1f})", "",
         "## Gates", "", "| gate | result |", "|---|---|",
         f"| plain fixed at 10% on the test rows reproduces the L5a critic stand-in (fixed {L5A_TEST_FIXED_PCT:.2f}%, "
         f"3,000-label gap {L5A_TEST_GAP_PP:+.2f} pp) | {'PASS' if rp['pass'] else 'FAIL'} (fixed {rp['mine_fixed_pct']:.4f}%, "
         f"gap {rp['mine_gap_pp']:+.4f} pp, max abs diff {rp['max_abs_diff_pp']:.6f} pp, tol {GATE_TOL_PP}) |",
         f"| spend gate | {'PASS' if sg['pass'] else 'FAIL'} ({sg['checks']} checks, failures {sg['failures']}) |",
         f"| every subset has {r['subset_size']} distinct prompts | {'PASS' if g['subset_distinct_prompts'] else 'FAIL'} |",
         f"| fixed picks identical across all tables | {'PASS' if g['fixed_identical_across_tables'] else 'FAIL'} |",
         f"| test question text found in p0 | {'PASS (0 rows)' if g['eval_question_overlap_zero'] else 'FAIL'} |", "",
         "## Verdict metrics", "",
         "| table | mean gap (pp) | sd (pp) | above +2.0 | fixed at 10% (%) | frozen at 10% (%) | verdict |",
         "|---|---|---|---|---|---|---|",
         f"| 3,000-label (mean over split seeds 0 to 7) | {t3['mean_pp']:+.2f} | {t3['sd_pp']:.2f} | "
         f"{t3['n_splits_above_pass']} of {len(r['splits'])} split gaps | {t3['fixed_pct_mean']:.2f} | "
         f"{t3['frozen_pct_mean']:.2f} | **{t3['verdict']}** |",
         f"| 200-label (mean of {t2['n_subsets']} subset means) | {t2['mean_pp']:+.2f} | {t2['sd_pp']:.2f} (over subset means) | "
         f"{t2['n_subsets_above_pass']} of {t2['n_subsets']} subsets | {t2['fixed_pct_mean']:.2f} | "
         f"{t2['frozen_pct_mean']:.2f} | **{t2['verdict']}** |", "",
         "## Subsets (200-label tables)", "",
         "| subset | strata in table | mean gap (pp) | docs in table (%) |", "|---|---|---|---|"]
    for s in r["subset_detail"]:
        L.append(f"| {s['subset']} | {s['strata_in_table']} | {s['mean_pp']:+.2f} | {100 * s['in_table_frac']:.1f} |")
    L += ["", "## Context only (outside the verdict)", "",
          "| set | prompts | proxy error | 3,000-label gap, mean (pp) | sd (pp) | above +2.0 |", "|---|---|---|---|---|---|",
          f"| remaining train, pid 1319 and up | {rm['prompts']} | {100 * rm['error_rate']:.1f}% | {rm['mean_pp']:+.2f} | "
          f"{rm['sd_pp']:.2f} | {rm['n_splits_above_pass']} of {len(rm['gap_pp_per_split'])} split gaps |",
          f"| {h['n_halves']} random 1,319-prompt halves of L4b (mean over halves) | {h['size']} | n/a | {h['mean_pp']:+.2f} | "
          f"{h['sd_pp']:.2f} | {h['n_above_pass']} of {h['n_halves']} halves (min {h['min_pp']:+.2f}, max {h['max_pp']:+.2f}) |", "",
          "## Notes", "",
          "- Test rows are pids 0 to 1318 of runs/L4b/feats.parquet. The cell100 split (L1B.build_sim, MIN 3, bucket shuffle seed "
          "= split seed) runs on the filtered frame, as in the L5a critic.",
          "- Subset i uses numpy RandomState(100 + i).choice(3000, 200, replace=False) over the p0 rows, as in L6b. Each subset "
          "table is fitted on its 200 prompts only and frozen.",
          "- Halves use numpy default_rng(100 + i).permutation over all L4b prompts, keeping the first 1,319 draws in pid order "
          "(the draw used by the L5a critic).",
          "- Per-split gaps for every subset, the 3,000-label table, the remaining train rows and each half are in result.json.", ""]
    with open(os.path.join(out_dir, "table.md"), "w") as fh: fh.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_feats", default=P0_FEATS)
    ap.add_argument("--eval_feats", default=L4B_FEATS)
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

    fit_df = pd.read_parquet(a.fit_feats)
    ev_all = pd.read_parquet(a.eval_feats).reset_index(drop=True)
    test = ev_all[ev_all["pid"] <= a.eval_pid_max].reset_index(drop=True)
    remain = ev_all[ev_all["pid"] > a.eval_pid_max].reset_index(drop=True)
    overlap = int(test["question"].isin(set(fit_df["question"])).sum())
    print(f"test rows {len(test)} (pid <= {a.eval_pid_max}), remaining rows {len(remain)}, test proxy error "
          f"{test[LABEL].mean():.3f}, p0 fit prompts {len(fit_df)}, test question overlap with p0 {overlap}", flush=True)

    full_table = fit_table(fit_df, a.k0)
    print(f"gate table: {len(full_table)} strata from {len(fit_df)} p0 prompts, k0 {a.k0:g}, no refit", flush=True)
    full = run_pool(test, full_table, a.budget, splits, seeds)
    full_sum = pool_summary(full, splits)
    repro = test_gate(full_sum)
    print(f"test-row reproduction: fixed at 10% {repro['mine_fixed_pct']:.4f}% (stand-in {L5A_TEST_FIXED_PCT:.2f}%), "
          f"3,000-label gap {repro['mine_gap_pp']:+.4f} pp (stand-in {L5A_TEST_GAP_PP:+.2f}), max abs diff "
          f"{repro['max_abs_diff_pp']:.6f} pp (tol {GATE_TOL_PP}), pass {repro['pass']}", flush=True)
    if not repro["pass"]:
        write_json(a.out, {"experiment": "L6a", "verdict": "GATE_FAILED", "stage": "test-row reproduction",
                           "gates": {"test_row_reproduction": repro}, "table_3000_test": full_sum})
        print("GATE FAILED at the test-row reproduction: no L6a verdict written", flush=True); sys.exit(1)

    subs = []
    for i in range(a.subsets):
        idx = draw_subset(fit_df, i, a.subset_size)
        sub = fit_df.iloc[idx]
        table = fit_table(sub, a.k0)
        R = run_pool(test, table, a.budget, splits, seeds)
        summ = pool_summary(R, splits)
        subs.append({"subset": i, "row_positions": [int(x) for x in idx], "distinct_pids": int(sub["pid"].nunique()),
                     "strata_in_table": int(len(table)), "runs": R, "summary": summ})
        print(f"subset {i:2d}: {len(table)} strata, test rows {summ['mean_pp']:+.2f} pp", flush=True)

    # Context only: remaining-train rows and random 1,319-prompt halves of the full L4b pool, 3,000-label table
    rem_R = run_pool(remain, full_table, a.budget, splits, seeds); rem_sum = pool_summary(rem_R, splits)
    halves = []
    for i in range(N_HALVES):
        idx = np.random.default_rng(HALF_SEED0 + i).permutation(len(ev_all))[:HALF_SIZE]
        sub = ev_all.iloc[sorted(idx)].reset_index(drop=True)
        R = run_pool(sub, full_table, a.budget, splits, seeds)
        halves.append({"half": i, "runs": R, "summary": pool_summary(R, splits)})
    hv = [float(np.mean(x["summary"]["gap_pp_per_split"])) for x in halves]

    fixed_same = all(x["runs"][s]["fixed_hits"] == full[s]["fixed_hits"] for x in subs for s in splits)
    sg = spend_gate({"test/3000": full, **{f"test/subset{x['subset']}": x["runs"] for x in subs},
                     "remaining/3000": rem_R, **{f"L4b/half{x['half']}": x["runs"] for x in halves}})
    subset_ok = all(x["distinct_pids"] == a.subset_size and len(set(x["row_positions"])) == a.subset_size for x in subs)
    overlap_ok = overlap == 0
    gates_ok = bool(sg["pass"] and subset_ok and overlap_ok and fixed_same and repro["pass"])
    gates = {"test_row_reproduction": repro, "spend_gate": sg, "subset_distinct_prompts": bool(subset_ok),
             "fixed_identical_across_tables": bool(fixed_same), "eval_question_overlap_zero": bool(overlap_ok),
             "pass": gates_ok}
    print(f"spend gate: {sg['checks']} checks, failures {sg['failures']}; subsets distinct {subset_ok}; "
          f"fixed identical {fixed_same}; test question overlap with p0 {overlap}; gates {'PASS' if gates_ok else 'FAIL'}",
          flush=True)
    if not gates_ok:
        write_json(a.out, {"experiment": "L6a", "verdict": "GATE_FAILED", "gates": gates})
        print("GATE FAILED: no L6a verdict written", flush=True); sys.exit(1)

    sm = [x["summary"] for x in subs]
    means = [x["mean_pp"] for x in sm]
    st, hst = L1B.stats(means), L1B.stats(hv)
    gaps_all = np.asarray([g_ for x in sm for g_ in x["gap_pp_per_split"]], float)
    t3 = {"mean_pp": full_sum["mean_pp"], "sd_pp": full_sum["sd_pp"], "gap_pp_per_split": full_sum["gap_pp_per_split"],
          "n_splits_above_pass": int(sum(g_ > L1B.PASS_PP for g_ in full_sum["gap_pp_per_split"])),
          "fixed_pct_mean": float(np.mean(full_sum["fixed_pct_per_split"])),
          "frozen_pct_mean": float(np.mean(full_sum["frozen_pct_per_split"])),
          "in_table_frac_mean": float(np.mean(full_sum["in_table_frac_per_split"])),
          "verdict": "PASS" if full_sum["mean_pp"] > L1B.PASS_PP else "KILL"}
    t2 = {"n_subsets": len(means), "subset_mean_pp": means, "mean_pp": st["mean_pp"], "sd_pp": st["sd_pp"],
          "n_subsets_above_pass": st["n_above_pass_pp"], "n_subsets_at_or_below_pass": len(means) - st["n_above_pass_pp"],
          "min_subset_mean_pp": float(min(means)), "max_subset_mean_pp": float(max(means)),
          "split_gap_mean_pp": float(gaps_all.mean()), "split_gap_sd_pp": float(gaps_all.std(ddof=1)),
          "fixed_pct_mean": float(np.mean([np.mean(x["fixed_pct_per_split"]) for x in sm])),
          "frozen_pct_mean": float(np.mean([np.mean(x["frozen_pct_per_split"]) for x in sm])),
          "gap_pp_per_subset": [x["gap_pp_per_split"] for x in sm],
          "verdict": "PASS" if st["mean_pp"] > L1B.PASS_PP else "KILL"}
    rm = {"prompts": int(len(remain)), "error_rate": float(remain[LABEL].mean()), "mean_pp": rem_sum["mean_pp"],
          "sd_pp": rem_sum["sd_pp"], "gap_pp_per_split": rem_sum["gap_pp_per_split"],
          "n_splits_above_pass": int(sum(g_ > L1B.PASS_PP for g_ in rem_sum["gap_pp_per_split"])),
          "fixed_pct_mean": float(np.mean(rem_sum["fixed_pct_per_split"])),
          "frozen_pct_mean": float(np.mean(rem_sum["frozen_pct_per_split"]))}
    hh = {"n_halves": N_HALVES, "size": HALF_SIZE, "rule": f"numpy default_rng({HALF_SEED0} + i).permutation(all L4b prompts)"
          f"[:{HALF_SIZE}], rows kept in pid order", "gap_mean_pp_per_half": hv, "mean_pp": hst["mean_pp"],
          "sd_pp": hst["sd_pp"], "n_above_pass": hst["n_above_pass_pp"], "min_pp": float(min(hv)), "max_pp": float(max(hv))}
    detail = [{"subset": x["subset"], "row_positions": x["row_positions"], "distinct_pids": x["distinct_pids"],
               "strata_in_table": x["strata_in_table"], "mean_pp": x["summary"]["mean_pp"],
               "gap_pp_per_split": x["summary"]["gap_pp_per_split"],
               "in_table_frac": float(np.mean(x["summary"]["in_table_frac_per_split"]))} for x in subs]
    out = {"experiment": "L6a", "label": LABEL, "group": GROUP, "stratum": STRATUM, "k0": a.k0, "budget_frac": a.budget,
           "fit_feats": a.fit_feats, "eval_feats": a.eval_feats, "eval_pid_max": a.eval_pid_max,
           "test_prompts": int(len(test)), "test_error_rate": float(test[LABEL].mean()),
           "fit_prompts": int(len(fit_df)), "subsets": a.subsets, "subset_size": a.subset_size,
           "subset_draw": "numpy RandomState(100 + i).choice(n_p0_rows, subset_size, replace=False)",
           "refit": "none (each table is frozen after its fit)", "seeds": seeds, "splits": splits, "sd_ddof": 1,
           "pass_threshold_pp": L1B.PASS_PP, "tie_seeds": a.tie_seeds, "eval_question_overlap_with_p0": overlap,
           "gates": gates, "table_3000": t3, "table_200": t2,
           "context_only": {"remaining_train_3000": rm, "halves_3000": hh},
           "verdicts": {"table_3000": t3["verdict"], "table_200": t2["verdict"]}, "subset_detail": detail}
    write_json(a.out, out); write_table(a.out, out)
    print(f"3,000-label table on test rows: mean gap {t3['mean_pp']:+.2f} pp, sd {t3['sd_pp']:.2f}, verdict {t3['verdict']}",
          flush=True)
    print(f"200-label tables on test rows: mean of subset means {t2['mean_pp']:+.2f} pp, sd {t2['sd_pp']:.2f}, above +2.0 "
          f"{t2['n_subsets_above_pass']} of {t2['n_subsets']}, verdict {t2['verdict']}", flush=True)
    print(f"context: remaining train {rm['mean_pp']:+.2f} pp (3,000-label table); {N_HALVES} halves mean {hh['mean_pp']:+.2f} pp, "
          f"above +2.0 {hh['n_above_pass']} of {N_HALVES}; first six halves {[round(x, 2) for x in hv[:6]]}", flush=True)
    print(f"wrote {os.path.join(a.out, 'result.json')} and table.md", flush=True)


if __name__ == "__main__":
    main()
