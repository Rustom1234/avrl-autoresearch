"""L7a: learned online allocator against the frozen 200-label table at equal labels. CPU only, no GPU job.

Copied from avrl/exp/l6b_frozen_label_budget.py (L6b is unchanged). Changes: one policy added (greedy_prior), the pass and
kill threshold set to +1.0 pp, and a gate that reproduces the per-subset frozen gaps of runs/L6b/result.json.

greedy_prior is the greedy branch of greedy_online in avrl/exp/g01_label_free_baselines.py: K = 1, refit every
R = max(budget // 8, 40) verified labels, top-scored unseen doc first, cheap features as in G01 (--cheap list). Its initial
scorer is fitted on the same 200 p0 prompts that fit the frozen table, not on random scores. Every refit uses the 200 p0
labels plus all verified eval labels so far. Each fit is StandardScaler plus LogisticRegression(class_weight balanced,
max_iter 1000). Ties in the score are broken by the per-seed random draw.

ASSUMPTION: G01 greedy_online has no p0 fit step and starts from random scores, so the 200-label initialisation is new code.
The refit includes the p0 labels because that is the variant most favourable to the learned arm, so a KILL is the strong reading.

Subset i draws 200 distinct p0 prompts without replacement with numpy RandomState(100 + i), as in L6b. The frozen table is
L6b config B (stratum agreement x n_distinct x any_truncated, k0 50) fitted on those prompts, frozen, never refit.
Splits 0 to 7, 8 tie-break seeds each, budget 10% of the eval sim pool.

Metric: per pool, greedy_prior minus the frozen 200-label table. Per split, the gap is the mean over tie-break seeds of the
hit-rate difference in pp. A subset mean is the mean over split seeds 0 to 7. The pool value is the mean over the 20 subset
means, with sd (ddof 1) and the count of subset means above +1.0 pp. Context: plain fixed, frozen minus plain fixed, and
greedy_prior minus plain fixed.

Gates, before any verdict:
  1. Plain fixed at 10% reproduces runs/L2b/result.json (config B) and runs/L4b/result.json (config B40) per split to
     0.01 pp, with the same calls per split.
  2. The 3,000-prompt frozen table reproduces runs/L5a/result.json gap per split on L2b and L4b to 1e-6 pp.
  3. The per-subset frozen gaps and row positions reproduce runs/L6b/result.json subset_detail to 1e-6 pp.
  4. Spend: every policy takes exactly int(budget * n_sim) distinct docs on every split, seed and subset (0 failures).
  5. Each subset has exactly subset_size distinct prompts, fixed picks are identical across all tables, and no eval
     question text occurs in p0.
Verdict per pool: PASS if the mean over the subset means is above +1.0 pp, else KILL. Overall PASS needs both pools.
Pass and kill are complementary. A gate failure is recorded as GATE_FAILED, not as a verdict.

Usage: one eval pool per call, then the combine step (avrl/exp/l7a_combine.py):
  python -I l7a_learned_equal_labels.py --fit_feats FEATS --eval_feats FEATS --budget 0.10 --k0 50 --subsets 20
         --subset_size 200 --splits 0-7 --tie_seeds 8 --out OUTDIR
Writes <out>/result.json and <out>/table.md. If a gate fails, result.json gets verdict GATE_FAILED and the exit code is 1.
"""
import os, sys, json, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import l1b_tuned_threshold as L1B

RUNS = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
LABEL, GROUP = "proxy_error", "cell100"
STRATUM = ["agreement", "n_distinct", "any_truncated"]
CHEAP = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]  # G01 --cheap list
PASS_PP = 1.0  # L7a threshold (L6b used +2.0)
REPRO_TOL_PP = 0.01
L5A_TOL_PP = 1e-6
L6B_TOL_PP = 1e-6
SUBSET_SEED0 = 100
# Pool name -> (result.json holding plain fixed rates at 10%, config key of config B in that file)
REPRO_REF = {"L2b": (f"{RUNS}/L2b/result.json", "B"), "L4b": (f"{RUNS}/L4b/result.json", "B40")}
L5A_REF = f"{RUNS}/L5a/result.json"
L6B_REF = f"{RUNS}/L6b/result.json"
# Refitted B at refit floor 40, shown beside the results for reference
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


def scorer():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))


def greedy_prior_picks(y, X, Xp, yp, budget, seed):
    """G01 greedy_online (K = 1) with the initial scorer fitted on the p0 labels. Returns the picked row positions in order.
    Refit every R verified labels on the p0 labels plus every verified eval label so far."""
    assert 0 < yp.sum() < len(yp), "p0 subset must hold both classes"
    rnd = np.random.default_rng(seed).random(len(y)); R = max(budget // 8, 40)
    seen = np.zeros(len(y), bool)
    score = scorer().fit(Xp, yp).predict_proba(X)[:, 1]
    picks, next_fit = [], R
    while len(picks) < budget:
        mu = np.flatnonzero(~seen)
        c = mu[np.lexsort((rnd[mu], -score[mu]))[0]]
        seen[c] = True; picks.append(int(c))
        if len(picks) >= next_fit:
            score = scorer().fit(np.vstack([Xp, X[seen]]), np.concatenate([yp, y[seen]])).predict_proba(X)[:, 1]
            next_fit += R
    return np.asarray(picks)


def job_greedy(args):
    feats, split, fit_sub, frac, seeds = args
    sim, _ = L1B.build_sim(pd.read_parquet(feats), GROUP, split)
    y = sim[LABEL].to_numpy(int); X = sim[CHEAP].fillna(0).to_numpy(float)
    Xp = fit_sub[CHEAP].fillna(0).to_numpy(float); yp = fit_sub[LABEL].to_numpy(int)
    budget = int(frac * len(y))
    hits, spend_bad = [], []
    for s in seeds:
        sel = greedy_prior_picks(y, X, Xp, yp, budget, s)
        d = int(len(np.unique(sel)))
        if len(sel) != budget or d != budget:
            spend_bad.append({"policy": "greedy_prior", "seed": int(s), "distinct": d, "expected": budget})
        hits.append(int(y[sel].sum()))
    return {"split": split, "budget": budget, "greedy_hits": hits, "spend_bad": spend_bad}


def run_pool_greedy(feats, fit_sub, frac, splits, seeds):
    with Pool(min(len(splits), os.cpu_count())) as p:
        res = p.map(job_greedy, [(feats, s, fit_sub, frac, seeds) for s in splits])
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
            "n_above_pass_pp": int((np.asarray(gaps) > PASS_PP).sum()),
            "verdict": "PASS" if st["mean_pp"] > PASS_PP else "KILL"}


def greedy_summary(R, G, splits):
    """Per split, pp, mean over tie-break seeds: greedy_prior minus frozen (primary), greedy_prior minus plain fixed (context).
    Also the spread of greedy hits across tie-break seeds, to show whether the tie-break matters at all."""
    vs_frozen, vs_fixed, pct, spread = [], [], [], []
    for s in splits:
        b = R[s]["budget"]; assert G[s]["budget"] == b
        g = 100.0 * np.asarray(G[s]["greedy_hits"], float) / b
        z = 100.0 * np.asarray(R[s]["frozen_hits"], float) / b
        f = 100.0 * np.asarray(R[s]["fixed_hits"], float) / b
        vs_frozen.append(float((g - z).mean())); vs_fixed.append(float((g - f).mean()))
        pct.append(float(g.mean())); spread.append(int(max(G[s]["greedy_hits"]) - min(G[s]["greedy_hits"])))
    return {"gap_vs_frozen_pp_per_split": vs_frozen, "gap_vs_fixed_pp_per_split": vs_fixed, "greedy_pct_per_split": pct,
            "tie_seed_spread_hits_per_split": spread, "mean_vs_frozen_pp": float(np.mean(vs_frozen)),
            "mean_vs_fixed_pp": float(np.mean(vs_fixed))}


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


def l6b_gate(name, subs):
    """Per-subset frozen gaps (plain fixed as base) and row positions against runs/L6b/result.json subset_detail."""
    ref = json.load(open(L6B_REF))["subset_detail"]
    worst, pos_ok = 0.0, len(ref) == len(subs)
    for x in subs:
        r = ref[x["subset"]]
        pos_ok = pos_ok and r["row_positions"] == x["row_positions"] and r["distinct_pids"] == x["distinct_pids"]
        mine = x["summary"][name]["gap_pp_per_split"]; theirs = [float(v) for v in r[f"{name}_gap_pp_per_split"]]
        worst = max(worst, max(abs(m - t) for m, t in zip(mine, theirs)))
    return {"ref": L6B_REF, "subsets": len(subs), "row_positions_match": bool(pos_ok), "max_abs_diff_pp": float(worst),
            "tol_pp": L6B_TOL_PP, "pass": bool(pos_ok and worst <= L6B_TOL_PP)}


def spend_gate(pools):
    bad = [dict(pool=name, split=s, **b) for name, R in pools.items() for s in R for b in R[s]["spend_bad"]]
    checks = sum(2 * len(R[next(iter(R))]["fixed_hits"]) * len(R) for R in pools.values())
    return {"checks": checks, "failures": len(bad), "examples": bad[:5], "pass": not bad}


def greedy_spend(subs, names):
    bad = [dict(pool=n, subset=x["subset"], split=s, **b)
           for x in subs for n in names for s in x["greedy"][n] for b in x["greedy"][n][s]["spend_bad"]]
    checks = sum(len(x["greedy"][n][s]["greedy_hits"]) for x in subs for n in names for s in x["greedy"][n])
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
    P = r["pools"]; names = list(P); G = r["gates"]
    L = ["# L7a: learned online allocator (greedy_prior) against the frozen 200-label table, equal labels", "",
         f"Rule: per pool, greedy_prior minus the frozen 200-label config B table at 10%, paired by split seed and tie-break "
         f"seed. A subset mean is the mean over split seeds 0 to 7 ({r['tie_seeds']} tie-break seeds each). PASS per pool if "
         f"the mean over the {r['subsets']} subset means is above +{PASS_PP:.1f} pp, else KILL. Overall PASS needs both pools.",
         "", f"Overall verdict: **{r['verdict']}**", "", "## Gates", "", "| gate | result |", "|---|---|"]
    for name, g in G["fixed_reproduction"].items():
        L.append(f"| plain fixed 10% reproduces runs/{name}/result.json | {'PASS' if g['pass'] else 'FAIL'} "
                 f"(max abs diff {g['max_abs_diff_pp']:.6f} pp, tol {REPRO_TOL_PP}, calls match {g['calls_match']}) |")
    for name, g in G["frozen_3000_reproduction_L5a"].items():
        L.append(f"| 3,000-prompt frozen table reproduces runs/L5a/result.json on {name} | {'PASS' if g['pass'] else 'FAIL'} "
                 f"(max abs diff {g['max_abs_diff_pp']:.6f} pp, tol {L5A_TOL_PP}) |")
    for name, g in G["frozen_subset_reproduction_L6b"].items():
        L.append(f"| per-subset frozen gaps reproduce runs/L6b/result.json on {name} | {'PASS' if g['pass'] else 'FAIL'} "
                 f"(max abs diff {g['max_abs_diff_pp']:.6f} pp, tol {L6B_TOL_PP}, row positions match {g['row_positions_match']}) |")
    for key, label in (("spend_gate", "spend gate, fixed and frozen"), ("greedy_spend_gate", "spend gate, greedy_prior")):
        sg = G[key]; L.append(f"| {label} | {'PASS' if sg['pass'] else 'FAIL'} ({sg['checks']} checks, failures {sg['failures']}) |")
    L.append(f"| every subset has {r['subset_size']} distinct prompts | {'PASS' if G['subset_distinct_prompts'] else 'FAIL'} |")
    L.append(f"| fixed picks identical across all tables | {'PASS' if G['fixed_identical_across_tables'] else 'FAIL'} |")
    L.append(f"| eval question text found in p0 | {'PASS (0 rows)' if G['eval_question_overlap_zero'] else 'FAIL'} |")
    L += ["", "## Verdict and context", "",
          "| pool | greedy_prior minus frozen 200-label, mean over subsets (pp) | sd over subset means (pp) | subsets above +1.0 | "
          "subsets at or below +1.0 | min to max subset mean (pp) | frozen 200-label minus plain fixed (pp) | "
          "greedy_prior minus plain fixed (pp) | plain fixed hit rate (%) | greedy_prior hit rate (%) | "
          "refitted B floor 40 minus plain fixed (pp) | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        p = P[n]
        L.append(f"| {n} | {p['mean_pp']:+.2f} | {p['sd_pp']:.2f} | {p['n_subsets_above_pass']} of {p['n_subsets']} | "
                 f"{p['n_subsets_at_or_below_pass']} of {p['n_subsets']} | {p['min_subset_mean_pp']:+.2f} to "
                 f"{p['max_subset_mean_pp']:+.2f} | {p['frozen_vs_fixed_mean_pp']:+.2f} | {p['greedy_vs_fixed_mean_pp']:+.2f} | "
                 f"{p['fixed_pct_mean']:.2f} | {p['greedy_pct_mean']:.2f} | {p['refit_b_floor40']['mean_pp']:+.2f} | "
                 f"**{p['verdict']}** |")
    hdr = ["subset", "strata in table", "distinct prompts"] + [f"{n} {c}" for n in names for c in
                                                             ("greedy minus frozen", "frozen minus fixed", "greedy minus fixed")]
    L += ["", "## Subsets: mean gap over split seeds 0 to 7 (pp)", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for s in r["subset_detail"]:
        cells = [str(s["subset"]), str(s["strata_in_table"]), str(s["distinct_pids"])]
        for n in names:
            cells += [f"{s[n + '_greedy_minus_frozen_pp']:+.2f}", f"{s[n + '_frozen_minus_fixed_pp']:+.2f}",
                      f"{s[n + '_greedy_minus_fixed_pp']:+.2f}"]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "## Reference rows", "", "| pool | source | mean (pp) | sd (pp) | per split (pp) |", "|---|---|---|---|---|"]
    for n in names:
        rf = P[n]["refit_b_floor40"]
        L.append(f"| {n} | {rf['source'].replace('/homes/rdubash/avrl_live/ROUTED/', '')} | {rf['mean_pp']:+.2f} | {rf['sd_pp']:.2f} | "
                 + " / ".join(f"{x:+.2f}" for x in rf["per_split_pp"]) + " |")
        L.append(f"| {n} | runs/L5a/result.json, 3,000-prompt frozen table, frozen minus plain fixed | "
                 f"{P[n]['full_table_3000_mean_pp']:+.2f} | - | - |")
    L += ["", "## Notes", "",
          "- greedy_prior is deterministic given the split and the subset: its initial scorer and every refit are fitted without "
          "randomness, so the tie-break seed only changes exact score ties. The spread of greedy hits across tie-break seeds is in "
          "result.json under pools[pool].max_tie_seed_spread_hits.",
          "- Each subset's frozen table is the L6b table for the same subset. Its per-split gaps are reproduced from runs/L6b/result.json.",
          "- Refits in greedy_prior use max_iter 1000. G01 greedy_online refits use max_iter 500.",
          "- Subset means are the mean over split seeds 0 to 7 of the paired gap. The full per-split values are in result.json.", ""]
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
        greedy = {n: run_pool_greedy(f, sub, a.budget, splits, seeds) for n, f in evals.items()}
        summ = {n: pool_summary(runs[n], splits) for n in evals}
        gsum = {n: greedy_summary(runs[n], greedy[n], splits) for n in evals}
        subs.append({"subset": i, "row_positions": [int(x) for x in idx], "distinct_pids": int(sub["pid"].nunique()),
                     "strata_in_table": int(len(table)), "runs": runs, "summary": summ, "greedy": greedy, "greedy_summary": gsum})
        print(f"subset {i:2d}: {len(table)} strata, frozen minus fixed " + ", ".join(f"{n} {summ[n]['mean_pp']:+.2f}" for n in evals)
              + ", greedy_prior minus frozen " + ", ".join(f"{n} {gsum[n]['mean_vs_frozen_pp']:+.2f}" for n in evals), flush=True)

    fixed_same = all(R[s]["fixed_hits"] == full[n][s]["fixed_hits"]
                     for x in subs for n, R in x["runs"].items() for s in splits)
    sg = spend_gate({**{f"{n}/gate": full[n] for n in evals},
                     **{f"{n}/subset{x['subset']}": x["runs"][n] for x in subs for n in evals}})
    gg = greedy_spend(subs, list(evals))
    repro = {n: repro_gate(n, full[n], splits) for n in evals}
    l5a = {n: l5a_gate(n, full_sum[n]) for n in evals}
    l6b = {n: l6b_gate(n, subs) for n in evals}
    subset_ok = all(x["distinct_pids"] == a.subset_size and len(set(x["row_positions"])) == a.subset_size for x in subs)
    overlap_ok = all(v == 0 for v in overlap.values())
    gates_ok = (sg["pass"] and gg["pass"] and subset_ok and overlap_ok and fixed_same
                and all(g["pass"] for g in repro.values()) and all(g["pass"] for g in l5a.values())
                and all(g["pass"] for g in l6b.values()))
    out = {"experiment": "L7a", "label": LABEL, "group": GROUP, "stratum": STRATUM, "cheap": CHEAP, "k0": a.k0,
           "budget_frac": a.budget, "fit_feats": a.fit_feats, "fit_prompts": int(len(fit_df)), "subsets": a.subsets,
           "subset_size": a.subset_size, "subset_draw": "numpy RandomState(100 + i).choice(n_p0_rows, subset_size, replace=False)",
           "frozen_refit": "none (each subset table is frozen after its fit)",
           "greedy_policy": "greedy_prior: G01 greedy_online, K = 1, R = max(budget // 8, 40), initial scorer fitted on the "
                            "subset's 200 p0 labels, refit on the p0 labels plus verified eval labels",
           "seeds": seeds, "splits": splits, "sd_ddof": 1, "pass_threshold_pp": PASS_PP, "tie_seeds": a.tie_seeds,
           "eval_question_overlap_with_p0": overlap,
           "gates": {"fixed_reproduction": repro, "frozen_3000_reproduction_L5a": l5a, "frozen_subset_reproduction_L6b": l6b,
                     "spend_gate": sg, "greedy_spend_gate": gg, "fixed_identical_across_tables": bool(fixed_same),
                     "subset_distinct_prompts": bool(subset_ok), "eval_question_overlap_zero": bool(overlap_ok),
                     "pass": bool(gates_ok)}}
    for n, v in repro.items():
        print(f"reproduction {n}: max abs diff {v['max_abs_diff_pp']:.6f} pp (tol {REPRO_TOL_PP}), calls match "
              f"{v['calls_match']}, pass {v['pass']}", flush=True)
    for n, v in l5a.items():
        print(f"L5a reproduction {n}: 3,000-prompt frozen mean {v['mine_mean_pp']:+.2f} (L5a {v['ref_mean_pp']:+.2f}), "
              f"max abs diff {v['max_abs_diff_pp']:.6f} pp (tol {L5A_TOL_PP}), pass {v['pass']}", flush=True)
    for n, v in l6b.items():
        print(f"L6b per-subset reproduction {n}: max abs diff {v['max_abs_diff_pp']:.2e} pp (tol {L6B_TOL_PP:g}), "
              f"row positions match {v['row_positions_match']}, pass {v['pass']}", flush=True)
    print(f"spend gate: {sg['checks']} checks, failures {sg['failures']}; greedy spend: {gg['checks']} checks, failures "
          f"{gg['failures']}; subsets distinct {subset_ok}; fixed identical {fixed_same}; eval overlap with p0 {overlap}; "
          f"gates {'PASS' if gates_ok else 'FAIL'}", flush=True)
    if not gates_ok:
        out["verdict"] = "GATE_FAILED"; write_json(a.out, out)
        print("GATE FAILED: no L7a verdict written", flush=True); sys.exit(1)

    pools = {}
    for n in evals:
        gs = [x["greedy_summary"][n] for x in subs]; sm = [x["summary"][n] for x in subs]
        means = [g["mean_vs_frozen_pp"] for g in gs]; m = np.asarray(means, float)
        ref = refit_ref(n)
        pools[n] = {"feats": evals[n], "mean_pp": float(m.mean()), "sd_pp": float(m.std(ddof=1)), "n_subsets": len(m),
                    "n_subsets_above_pass": int((m > PASS_PP).sum()),
                    "n_subsets_at_or_below_pass": int((m <= PASS_PP).sum()),
                    "min_subset_mean_pp": float(m.min()), "max_subset_mean_pp": float(m.max()),
                    "subset_mean_pp": means, "gap_pp_per_subset": [g["gap_vs_frozen_pp_per_split"] for g in gs],
                    "n_split_gaps": int(sum(len(g["gap_vs_frozen_pp_per_split"]) for g in gs)),
                    "n_split_gaps_above_pass": int(sum((np.asarray(g["gap_vs_frozen_pp_per_split"]) > PASS_PP).sum() for g in gs)),
                    "max_tie_seed_spread_hits": int(max(max(g["tie_seed_spread_hits_per_split"]) for g in gs)),
                    "greedy_pct_mean": float(np.mean([np.mean(g["greedy_pct_per_split"]) for g in gs])),
                    "fixed_pct_mean": float(np.mean([np.mean(x["fixed_pct_per_split"]) for x in sm])),
                    "frozen_pct_mean": float(np.mean([np.mean(x["frozen_pct_per_split"]) for x in sm])),
                    "frozen_vs_fixed_mean_pp": float(np.mean([x["mean_pp"] for x in sm])),
                    "greedy_vs_fixed_mean_pp": float(np.mean([g["mean_vs_fixed_pp"] for g in gs])),
                    "full_table_3000_mean_pp": full_sum[n]["mean_pp"], "refit_b_floor40": ref,
                    "greedy_minus_refit_b40_pp": float(m.mean() - ref["mean_pp"]),
                    "verdict": "PASS" if m.mean() > PASS_PP else "KILL"}
    out["pools"] = pools
    out["verdicts"] = {n: pools[n]["verdict"] for n in evals}
    out["verdict"] = "PASS" if all(v == "PASS" for v in out["verdicts"].values()) else "KILL"
    out["subset_detail"] = [dict({"subset": x["subset"], "row_positions": x["row_positions"], "distinct_pids": x["distinct_pids"],
                                  "strata_in_table": x["strata_in_table"]},
                                 **{f"{n}_greedy_minus_frozen_pp": x["greedy_summary"][n]["mean_vs_frozen_pp"] for n in evals},
                                 **{f"{n}_greedy_minus_frozen_pp_per_split": x["greedy_summary"][n]["gap_vs_frozen_pp_per_split"]
                                    for n in evals},
                                 **{f"{n}_frozen_minus_fixed_pp": x["summary"][n]["mean_pp"] for n in evals},
                                 **{f"{n}_gap_pp_per_split": x["summary"][n]["gap_pp_per_split"] for n in evals},
                                 **{f"{n}_greedy_minus_fixed_pp": x["greedy_summary"][n]["mean_vs_fixed_pp"] for n in evals})
                           for x in subs]
    for n in pools:
        p = pools[n]
        print(f"{n}: greedy_prior minus frozen 200-label, mean over {p['n_subsets']} subsets {p['mean_pp']:+.2f}, sd over subset "
              f"means {p['sd_pp']:.2f}, subsets above +1.0 {p['n_subsets_above_pass']} of {p['n_subsets']}, min "
              f"{p['min_subset_mean_pp']:+.2f} max {p['max_subset_mean_pp']:+.2f}, frozen minus fixed {p['frozen_vs_fixed_mean_pp']:+.2f}, "
              f"greedy minus fixed {p['greedy_vs_fixed_mean_pp']:+.2f}, verdict {p['verdict']}", flush=True)
    print(f"L7a verdict: {out['verdict']} (per pool {out['verdicts']})", flush=True)
    write_json(a.out, out); write_table(a.out, out)
    print(f"wrote {os.path.join(a.out, 'result.json')} and table.md", flush=True)


if __name__ == "__main__":
    main()
