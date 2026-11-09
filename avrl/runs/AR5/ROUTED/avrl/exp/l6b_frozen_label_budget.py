"""L6b: frozen config B fitted on 200 p0 labels, 20 label subsets. CPU only.

Copied from avrl/exp/l5a_frozen_table.py (L5a is unchanged). Change: the frozen table is fitted on a 200-prompt subset
of p0 instead of all 3,000 prompts, repeated for 20 subsets. Subset i draws 200 distinct p0 prompts without replacement
with numpy RandomState(100 + i). Config B (stratum agreement x n_distinct x any_truncated, k0 50, 36-stratum posterior
with prior 1 - agreement, errors from proxy_error) is fitted on those prompts only, frozen, and never refit.
Each frozen table is evaluated on L2b and L4b at budget 0.10 over split seeds 0 to 7 with 8 tie-break seeds each,
against plain fixed on the same split and tie-break seed. The gap per split is paired by tie-break seed. A subset mean
is the mean over its 8 split gaps. The pool verdict uses the mean over the 20 subset means.
The 3,000-prompt table is also run, as a gate against runs/L5a/result.json.

Gates, before any verdict:
  1. Plain fixed at 10% reproduces runs/L2b/result.json (config B) and runs/L4b/result.json (config B40) per split to
     0.01 pp, with the same calls per split.
  2. The 3,000-prompt frozen table reproduces runs/L5a/result.json gap per split on L2b and L4b to 1e-6 pp.
  3. Spend: every policy takes exactly int(budget * n_sim) distinct docs on every split, seed and table (0 failures).
  4. Each subset has exactly subset_size distinct prompts, fixed picks are identical across all tables, and no eval
     question text occurs in p0.
Verdict: per eval pool, PASS if the mean over the subset means is above +2.0 pp, else KILL. Overall PASS needs both pools.

Usage: python -I l6b_frozen_label_budget.py --fit_feats FEATS --eval_feats FEATS [FEATS ...] --budget 0.10 --k0 50
           --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8 --out OUTDIR
Writes <out>/result.json and <out>/table.md. If a gate fails, result.json gets verdict GATE_FAILED and the exit code is 1.
"""
import os, sys, json, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import l1b_tuned_threshold as L1B

RUNS = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
LABEL, GROUP = "proxy_error", "cell100"
STRATUM = ["agreement", "n_distinct", "any_truncated"]
REPRO_TOL_PP = 0.01
L5A_TOL_PP = 1e-6
SUBSET_SEED0 = 100
# Pool name -> (result.json holding plain fixed rates at 10%, config key of config B in that file)
REPRO_REF = {"L2b": (f"{RUNS}/L2b/result.json", "B"), "L4b": (f"{RUNS}/L4b/result.json", "B40")}
L5A_REF = f"{RUNS}/L5a/result.json"
# Refitted B at refit floor 40, shown beside the frozen result for reference
REFIT40_REF = {"L2b": (f"{RUNS}/L3a/fresh/result.json", "B"), "L4b": (f"{RUNS}/L4b/result.json", "B40")}


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def parse_splits(s):
    if "-" in s:
        lo, hi = s.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def pool_name(path):
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


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
    feats, split, table, frac, seeds = args
    df = attach_score(pd.read_parquet(feats), table)
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


def run_pool(feats, table, frac, splits, seeds):
    with Pool(min(len(splits), os.cpu_count())) as p:
        res = p.map(job, [(feats, s, table, frac, seeds) for s in splits])
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


def repro_gate(name, R, splits):
    path, key = REPRO_REF[name]
    ref = json.load(open(path))["configs"][key]
    ref_pct = [float(x) for x in ref["fixed_hit_pct_per_split"]]; ref_calls = [int(x) for x in ref["calls_per_split"]]
    mine = [float(100.0 * np.mean(R[s]["fixed_hits"]) / R[s]["budget"]) for s in splits]
    calls = [R[s]["budget"] for s in splits]
    d = max(abs(m - r) for m, r in zip(mine, ref_pct))
    return {"ref": path, "ref_config": key, "ref_fixed_pct_per_split": ref_pct, "mine_fixed_pct_per_split": mine,
            "max_abs_diff_pp": float(d), "tol_pp": REPRO_TOL_PP, "calls_match": calls == ref_calls,
            "pass": bool(d <= REPRO_TOL_PP and calls == ref_calls)}


def l5a_gate(name, summary):
    ref = json.load(open(L5A_REF))["pools"][name]
    mine = summary["gap_pp_per_split"]; theirs = [float(x) for x in ref["gap_pp_per_split"]]
    d = max(abs(m - t) for m, t in zip(mine, theirs))
    return {"ref": L5A_REF, "ref_mean_pp": float(ref["mean_pp"]), "mine_mean_pp": summary["mean_pp"],
            "ref_gap_pp_per_split": theirs, "mine_gap_pp_per_split": mine, "max_abs_diff_pp": float(d),
            "tol_pp": L5A_TOL_PP, "pass": bool(d <= L5A_TOL_PP and len(mine) == len(theirs))}


def spend_gate(pools):
    bad = [dict(pool=name, split=s, **b) for name, R in pools.items() for s in R for b in R[s]["spend_bad"]]
    checks = sum(2 * len(R[next(iter(R))]["fixed_hits"]) * len(R) for R in pools.values())
    return {"checks": checks, "failures": len(bad), "examples": bad[:5], "pass": not bad}


def refit_ref(name):
    path, key = REFIT40_REF[name]
    a = json.load(open(path))["configs"][key]["all_splits"]
    return {"source": path, "config": key, "mean_pp": float(a["mean_pp"]), "sd_pp": float(a["sd_pp"]),
            "per_split_pp": [float(x) for x in a["per_split_pp"]]}


def draw_subset(fit_df, i, size):
    """Row positions of label subset i: size distinct p0 prompts, numpy RandomState(100 + i), without replacement."""
    return np.random.RandomState(SUBSET_SEED0 + i).choice(len(fit_df), size=size, replace=False)


def write_json(out_dir, out):
    with open(os.path.join(out_dir, "result.json"), "w") as fh: json.dump(out, fh, indent=1, default=_jsonable)


def write_table(out_dir, r):
    P = r["pools"]; names = list(P)
    L = ["# L6b: frozen config B fitted on 200 p0 labels, 20 label subsets", "",
         f"Rule: frozen-table gap is frozen B minus plain fixed at 10%, paired by split seed and tie-break seed. A subset mean "
         f"is the mean over split seeds 0 to 7 ({r['tie_seeds']} tie-break seeds each). PASS per pool if the mean over the "
         f"{r['subsets']} subset means is above +{L1B.PASS_PP:.1f} pp, else KILL. Overall PASS needs both pools.", "",
         f"Overall verdict: **{r['verdict']}**", "",
         "## Gates", "", "| gate | result |", "|---|---|"]
    for name, g in r["gates"]["fixed_reproduction"].items():
        L.append(f"| plain fixed 10% reproduces runs/{name}/result.json | {'PASS' if g['pass'] else 'FAIL'} "
                 f"(max abs diff {g['max_abs_diff_pp']:.6f} pp, tol {REPRO_TOL_PP}, calls match {g['calls_match']}) |")
    for name, g in r["gates"]["frozen_3000_reproduction_L5a"].items():
        L.append(f"| 3,000-prompt frozen table reproduces runs/L5a/result.json on {name} | {'PASS' if g['pass'] else 'FAIL'} "
                 f"(max abs diff {g['max_abs_diff_pp']:.6f} pp, tol {L5A_TOL_PP}) |")
    sg = r["gates"]["spend_gate"]
    L.append(f"| spend gate | {'PASS' if sg['pass'] else 'FAIL'} ({sg['checks']} checks, failures {sg['failures']}) |")
    L.append(f"| every subset has {r['subset_size']} distinct prompts | {'PASS' if r['gates']['subset_distinct_prompts'] else 'FAIL'} |")
    L.append(f"| fixed picks identical across all tables | {'PASS' if r['gates']['fixed_identical_across_tables'] else 'FAIL'} |")
    L.append(f"| eval question text found in p0 | {'PASS (0 rows)' if r['gates']['eval_question_overlap_zero'] else 'FAIL'} |")
    L += ["", "## Verdict and reference", "",
          "| pool | 200-label frozen gap, mean over subsets (pp) | sd over subset means (pp) | subsets above +2.0 | "
          "subsets at or below +2.0 | min to max subset mean (pp) | refitted B floor 40 (pp) | 3,000-label frozen (pp) | "
          "200-label minus refitted B (pp) | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        p = P[n]
        L.append(f"| {n} | {p['mean_pp']:+.2f} | {p['sd_pp']:.2f} | {p['n_subsets_above_pass']} of {p['n_subsets']} | "
                 f"{p['n_subsets_at_or_below_pass']} of {p['n_subsets']} | {p['min_subset_mean_pp']:+.2f} to "
                 f"{p['max_subset_mean_pp']:+.2f} | {p['refit_b_floor40']['mean_pp']:+.2f} | "
                 f"{p['full_table_3000_mean_pp']:+.2f} | {p['subset_minus_refit_b40_pp']:+.2f} | **{p['verdict']}** |")
    L += ["", "## Subsets: mean gap over split seeds 0 to 7 (pp), strata, share of eval docs labelled by the table", "",
          "| subset | strata in 200-label table | " + " | ".join(f"{n} subset mean (pp)" for n in names) + " | "
          + " | ".join(f"{n} docs in table (%)" for n in names) + " |",
          "|---|---|" + "---|" * len(names) + "---|" * len(names)]
    for s in r["subset_detail"]:
        L.append(f"| {s['subset']} | {s['strata_in_table']} | "
                 + " | ".join(f"{s[n + '_mean_pp']:+.2f}" for n in names) + " | "
                 + " | ".join(f"{100 * s[n + '_in_table_frac']:.1f}" for n in names) + " |")
    L += ["", "## Reference rows", "",
          "| pool | source | mean (pp) | sd (pp) | per split (pp) |", "|---|---|---|---|---|"]
    for n in names:
        rf = P[n]["refit_b_floor40"]
        L.append(f"| {n} | {rf['source'].replace('/homes/rdubash/avrl_live/ROUTED/', '')} | {rf['mean_pp']:+.2f} | {rf['sd_pp']:.2f} | "
                 + " / ".join(f"{x:+.2f}" for x in rf["per_split_pp"]) + " |")
        L.append(f"| {n} | runs/L5a/result.json, 3,000-prompt frozen table | {P[n]['full_table_3000_mean_pp']:+.2f} | - | - |")
    L += ["", "## Notes", "",
          "- Subset i uses numpy RandomState(100 + i).choice(3000, 200, replace=False) over the p0 rows. Row positions are in "
          "result.json under subset_detail.",
          "- Each subset table is fitted on its 200 prompts only, frozen, and never refit. Strata absent from a subset table "
          "score their prior, as in L5a.",
          "- Per-split gaps for every subset and pool are in result.json under pools[pool].gap_pp_per_subset.",
          "- Subset means are the mean over split seeds 0 to 7 of the paired gap. Per-split values are in result.json.", ""]
    with open(os.path.join(out_dir, "table.md"), "w") as fh: fh.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_feats", required=True)
    ap.add_argument("--eval_feats", nargs="+", required=True)
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
    evals = {}
    for f in a.eval_feats:
        name = pool_name(f)
        if name not in REPRO_REF: sys.exit(f"no reproduction reference for pool {name}")
        evals[name] = f
    p0_questions = set(fit_df["question"])
    overlap = {n: int(pd.read_parquet(f)["question"].isin(p0_questions).sum()) for n, f in evals.items()}

    full_table = fit_table(fit_df, a.k0)
    print(f"gate table: {len(full_table)} strata from {len(fit_df)} p0 prompts, k0 {a.k0:g}, no refit", flush=True)
    full = {n: run_pool(f, full_table, a.budget, splits, seeds) for n, f in evals.items()}
    full_sum = {n: pool_summary(full[n], splits) for n in evals}

    subs = []
    for i in range(a.subsets):
        idx = draw_subset(fit_df, i, a.subset_size)
        sub = fit_df.iloc[idx]
        table = fit_table(sub, a.k0)
        runs = {n: run_pool(f, table, a.budget, splits, seeds) for n, f in evals.items()}
        summ = {n: pool_summary(runs[n], splits) for n in evals}
        subs.append({"subset": i, "row_positions": [int(x) for x in idx], "distinct_pids": int(sub["pid"].nunique()),
                     "strata_in_table": int(len(table)), "runs": runs, "summary": summ})
        print(f"subset {i:2d}: {len(table)} strata, " + ", ".join(f"{n} {summ[n]['mean_pp']:+.2f}" for n in evals), flush=True)

    fixed_same = all(R[s]["fixed_hits"] == full[n][s]["fixed_hits"]
                     for x in subs for n, R in x["runs"].items() for s in splits)
    sg = spend_gate({**{f"{n}/gate": full[n] for n in evals},
                     **{f"{n}/subset{x['subset']}": x["runs"][n] for x in subs for n in evals}})
    repro = {n: repro_gate(n, full[n], splits) for n in evals}
    l5a = {n: l5a_gate(n, full_sum[n]) for n in evals}
    subset_ok = all(x["distinct_pids"] == a.subset_size and len(set(x["row_positions"])) == a.subset_size for x in subs)
    overlap_ok = all(v == 0 for v in overlap.values())
    gates_ok = (sg["pass"] and subset_ok and overlap_ok and fixed_same
                and all(g["pass"] for g in repro.values()) and all(g["pass"] for g in l5a.values()))
    out = {"experiment": "L6b", "label": LABEL, "group": GROUP, "stratum": STRATUM, "k0": a.k0, "budget_frac": a.budget,
           "fit_feats": a.fit_feats, "fit_prompts": int(len(fit_df)), "subsets": a.subsets, "subset_size": a.subset_size,
           "subset_draw": "numpy RandomState(100 + i).choice(n_p0_rows, subset_size, replace=False)",
           "refit": "none (each subset table is frozen after its fit)", "seeds": seeds, "splits": splits, "sd_ddof": 1,
           "pass_threshold_pp": L1B.PASS_PP, "tie_seeds": a.tie_seeds, "eval_question_overlap_with_p0": overlap,
           "gates": {"fixed_reproduction": repro, "frozen_3000_reproduction_L5a": l5a, "spend_gate": sg,
                     "fixed_identical_across_tables": bool(fixed_same), "subset_distinct_prompts": bool(subset_ok),
                     "eval_question_overlap_zero": bool(overlap_ok), "pass": bool(gates_ok)}}
    for n, v in repro.items():
        print(f"reproduction {n}: max abs diff {v['max_abs_diff_pp']:.6f} pp (tol {REPRO_TOL_PP}), calls match "
              f"{v['calls_match']}, pass {v['pass']}", flush=True)
    for n, v in l5a.items():
        print(f"L5a reproduction {n}: 3,000-prompt frozen mean {v['mine_mean_pp']:+.2f} (L5a {v['ref_mean_pp']:+.2f}), "
              f"max abs diff {v['max_abs_diff_pp']:.6f} pp (tol {L5A_TOL_PP}), pass {v['pass']}", flush=True)
    print(f"spend gate: {sg['checks']} checks, failures {sg['failures']}; subsets distinct {subset_ok}; "
          f"fixed identical {fixed_same}; eval overlap with p0 {overlap}; gates {'PASS' if gates_ok else 'FAIL'}", flush=True)
    if not gates_ok:
        out["verdict"] = "GATE_FAILED"; write_json(a.out, out)
        print("GATE FAILED: no L6b verdict written", flush=True); sys.exit(1)

    pools = {}
    for n in evals:
        sm = [x["summary"][n] for x in subs]
        means = [x["mean_pp"] for x in sm]
        st = L1B.stats(means)
        gaps_all = np.asarray([g for x in sm for g in x["gap_pp_per_split"]], float)
        ref = refit_ref(n)
        pools[n] = {"feats": evals[n], "mean_pp": st["mean_pp"], "sd_pp": st["sd_pp"], "n_subsets": len(means),
                    "n_subsets_above_pass": st["n_above_pass_pp"],
                    "n_subsets_at_or_below_pass": len(means) - st["n_above_pass_pp"],
                    "min_subset_mean_pp": float(min(means)), "max_subset_mean_pp": float(max(means)),
                    "split_gap_mean_pp": float(gaps_all.mean()), "split_gap_sd_pp": float(gaps_all.std(ddof=1)),
                    "n_split_gaps": int(gaps_all.size), "n_split_gaps_above_pass": int((gaps_all > L1B.PASS_PP).sum()),
                    "subset_mean_pp": means, "gap_pp_per_subset": [x["gap_pp_per_split"] for x in sm],
                    "fixed_pct_mean": float(np.mean([np.mean(x["fixed_pct_per_split"]) for x in sm])),
                    "frozen_pct_mean": float(np.mean([np.mean(x["frozen_pct_per_split"]) for x in sm])),
                    "in_table_frac_per_subset": [float(np.mean(x["in_table_frac_per_split"])) for x in sm],
                    "full_table_3000_mean_pp": full_sum[n]["mean_pp"], "refit_b_floor40": ref,
                    "subset_minus_refit_b40_pp": st["mean_pp"] - ref["mean_pp"],
                    "verdict": "PASS" if st["mean_pp"] > L1B.PASS_PP else "KILL"}
    out["pools"] = pools
    out["verdicts"] = {n: pools[n]["verdict"] for n in evals}
    out["verdict"] = "PASS" if all(v == "PASS" for v in out["verdicts"].values()) else "KILL"
    out["subset_detail"] = [dict({"subset": x["subset"], "row_positions": x["row_positions"],
                                  "distinct_pids": x["distinct_pids"], "strata_in_table": x["strata_in_table"]},
                                 **{f"{n}_mean_pp": x["summary"][n]["mean_pp"] for n in evals},
                                 **{f"{n}_gap_pp_per_split": x["summary"][n]["gap_pp_per_split"] for n in evals},
                                 **{f"{n}_in_table_frac": float(np.mean(x["summary"][n]["in_table_frac_per_split"]))
                                    for n in evals}) for x in subs]
    out["plan_branches"] = {
        "both_pools_pass": out["verdict"] == "PASS",
        "within_1pp_of_refit_b40_both_pools": bool(all(abs(pools[n]["subset_minus_refit_b40_pp"]) <= 1.0 for n in evals)),
        "at_or_below_2pp_either_pool": bool(any(pools[n]["mean_pp"] <= L1B.PASS_PP for n in evals)),
    }
    for n in pools:
        p = pools[n]
        print(f"{n}: 200-label frozen minus fixed at 10%, mean over {p['n_subsets']} subsets {p['mean_pp']:+.2f}, "
              f"sd over subset means {p['sd_pp']:.2f}, subsets above +2.0 {p['n_subsets_above_pass']} of {p['n_subsets']}, "
              f"min {p['min_subset_mean_pp']:+.2f} max {p['max_subset_mean_pp']:+.2f}, refitted B floor 40 "
              f"{p['refit_b_floor40']['mean_pp']:+.2f}, 3,000-label frozen {p['full_table_3000_mean_pp']:+.2f}, "
              f"verdict {p['verdict']}", flush=True)
    print(f"L6b verdict: {out['verdict']} (per pool {out['verdicts']})", flush=True)
    write_json(a.out, out); write_table(a.out, out)
    print(f"wrote {os.path.join(a.out, 'result.json')} and table.md", flush=True)


if __name__ == "__main__":
    main()
