"""L9b: learned arm at 3,000 p0 labels against the 3,000-label frozen table (L5a). CPU only, no GPU job.

Derived from avrl/exp/l7a_learned_equal_labels.py, which is unchanged. Changes:
  1. The greedy_prior initial scorer and every refit are fitted on all 3,000 p0 rows (one fit, no 200-row subset draw).
     Each refit still adds every verified eval label so far, as in L7a.
  2. The comparator is the L5a frozen table: fitted once on all 3,000 p0 labels with k0 50 and prior 1 - agreement,
     never refit, scored with attach_score and frozen_pick.
  3. There are no subsets, so the L6b per-subset gate does not apply and is not run. The plain fixed gate, the L5a
     reproduction gate and both spend gates are kept.

greedy_prior is the greedy branch of greedy_online in avrl/exp/g01_label_free_baselines.py: K = 1, refit every
R = max(budget // 8, 40) verified labels, top-scored unseen doc first, cheap features as in G01 (--cheap list). Each fit is
StandardScaler plus LogisticRegression(class_weight balanced, max_iter 1000). Ties in the score are broken by the per-seed draw.

Metric: per pool, per split (0 to 7), the mean over 8 tie-break seeds of the greedy_prior hit rate minus the frozen
3,000-label hit rate, in pp. The pool value is the mean over the 8 split gaps, with sd (ddof 1) over splits and the count
of splits above +1.0 pp. Context: frozen minus plain fixed (should match L5a: +5.65 on L2b, +8.38 on L4b) and greedy
minus plain fixed. All 8 split gaps share the one p0 fit, so the sd over splits covers eval-side variation only.

Verdict per pool: PASS if the pool mean is above +1.0 pp, else KILL. Overall PASS needs both pools. A gate failure is
recorded as GATE_FAILED, not as a verdict.

Usage (one call, CPU):
  python -I l9b_learned_3000_labels.py --fit_feats runs/p0/feats.parquet --eval_feats runs/L2b/feats.parquet runs/L4b/feats.parquet
         --budget 0.10 --k0 50 --splits 0-7 --tie_seeds 8 --out runs/L9b
Writes <out>/result.json (including the raw hit counts per split) and <out>/table.md. A gate failure writes verdict
GATE_FAILED and exits 1.
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
PASS_PP = 1.0  # L9b threshold, as L7a
REPRO_TOL_PP = 0.01
L5A_TOL_PP = 1e-6
N_P0 = 3000
# Pool name -> (result.json holding plain fixed rates at 10%, config key of config B in that file)
REPRO_REF = {"L2b": (f"{RUNS}/L2b/result.json", "B"), "L4b": (f"{RUNS}/L4b/result.json", "B40")}
L5A_REF = f"{RUNS}/L5a/result.json"


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
    assert 0 < yp.sum() < len(yp), "p0 fit set must hold both classes"
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
    feats, split, fit_set, frac, seeds = args
    sim, _ = L1B.build_sim(pd.read_parquet(feats), GROUP, split)
    y = sim[LABEL].to_numpy(int); X = sim[CHEAP].fillna(0).to_numpy(float)
    Xp = fit_set[CHEAP].fillna(0).to_numpy(float); yp = fit_set[LABEL].to_numpy(int)
    budget = int(frac * len(y))
    hits, spend_bad = [], []
    for s in seeds:
        sel = greedy_prior_picks(y, X, Xp, yp, budget, s)
        d = int(len(np.unique(sel)))
        if len(sel) != budget or d != budget:
            spend_bad.append({"policy": "greedy_prior", "seed": int(s), "distinct": d, "expected": budget})
        hits.append(int(y[sel].sum()))
    return {"split": split, "budget": budget, "greedy_hits": hits, "spend_bad": spend_bad}


def run_pool_greedy(feats, fit_set, frac, splits, seeds):
    with Pool(min(len(splits), os.cpu_count())) as p:
        res = p.map(job_greedy, [(feats, s, fit_set, frac, seeds) for s in splits])
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


def spend_gate(pools):
    bad = [dict(pool=name, split=s, **b) for name, R in pools.items() for s in R for b in R[s]["spend_bad"]]
    checks = sum(2 * len(R[next(iter(R))]["fixed_hits"]) * len(R) for R in pools.values())
    return {"checks": checks, "failures": len(bad), "examples": bad[:5], "pass": not bad}


def greedy_spend(G):
    bad = [dict(pool=n, split=s, **b) for n in G for s in G[n] for b in G[n][s]["spend_bad"]]
    checks = sum(len(G[n][s]["greedy_hits"]) for n in G for s in G[n])
    return {"checks": checks, "failures": len(bad), "examples": bad[:5], "pass": not bad}


def write_json(out_dir, out):
    with open(os.path.join(out_dir, "result.json"), "w") as fh: json.dump(out, fh, indent=1, default=_jsonable)


def write_table(out_dir, r):
    P = r["pools"]; names = list(P); G = r["gates"]; splits = r["splits"]
    L = ["# L9b: learned arm (greedy_prior) at 3,000 p0 labels against the 3,000-label frozen table (L5a), 10% budget", "",
         f"Rule: per pool, greedy_prior minus the 3,000-label frozen table at 10%, paired by split seed and tie-break seed. "
         f"A split gap is the mean over {r['tie_seeds']} tie-break seeds. PASS per pool if the mean over split seeds "
         f"{splits[0]} to {splits[-1]} is above +{PASS_PP:.1f} pp, else KILL. Overall PASS needs both pools.",
         "", f"Overall verdict: **{r['verdict']}**", "", "## Gates", "", "| gate | result |", "|---|---|"]
    for name, g in G["fixed_reproduction"].items():
        L.append(f"| plain fixed 10% reproduces runs/{name}/result.json | {'PASS' if g['pass'] else 'FAIL'} "
                 f"(max abs diff {g['max_abs_diff_pp']:.6f} pp, tol {REPRO_TOL_PP}, calls match {g['calls_match']}) |")
    for name, g in G["frozen_3000_reproduction_L5a"].items():
        L.append(f"| 3,000-prompt frozen table reproduces runs/L5a/result.json on {name} | {'PASS' if g['pass'] else 'FAIL'} "
                 f"(max abs diff {g['max_abs_diff_pp']:.6f} pp, tol {L5A_TOL_PP}) |")
    L.append(f"| spend gate, fixed and frozen | {'PASS' if G['spend_gate']['pass'] else 'FAIL'} "
             f"({G['spend_gate']['checks']} checks, failures {G['spend_gate']['failures']}) |")
    L.append(f"| spend gate, greedy_prior | {'PASS' if G['greedy_spend_gate']['pass'] else 'FAIL'} "
             f"({G['greedy_spend_gate']['checks']} checks, failures {G['greedy_spend_gate']['failures']}) |")
    L.append(f"| eval question text found in p0 | {'PASS (0 rows)' if G['eval_question_overlap_zero'] else 'FAIL'} |")
    L.append("| L6b per-subset reproduction | not applicable (L9b has no subsets) |")
    L += ["", "## Verdict and context", "",
          "| pool | greedy_prior minus 3,000-label table, mean over splits (pp) | sd over splits (pp) | splits above +1.0 | "
          "min to max split gap (pp) | 3,000-label table minus plain fixed (pp) | greedy_prior minus plain fixed (pp) | "
          "plain fixed hit rate (%) | 3,000-label table hit rate (%) | greedy_prior hit rate (%) | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        p = P[n]
        L.append(f"| {n} | {p['mean_pp']:+.2f} | {p['sd_pp']:.2f} | {p['n_splits_above_pass']} of {p['n_splits']} | "
                 f"{p['min_split_pp']:+.2f} to {p['max_split_pp']:+.2f} | {p['table_minus_fixed_mean_pp']:+.2f} | "
                 f"{p['greedy_minus_fixed_mean_pp']:+.2f} | {p['fixed_pct_mean']:.2f} | {p['table_pct_mean']:.2f} | "
                 f"{p['greedy_pct_mean']:.2f} | **{p['verdict']}** |")
    hdr = ["split"] + [f"{n} {c}" for n in names for c in ("greedy minus table", "table minus fixed", "greedy minus fixed")]
    L += ["", "## Splits: gap per split seed (pp)", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for i, s in enumerate(splits):
        cells = [str(s)]
        for n in names:
            cells += [f"{P[n]['greedy_minus_table_pp_per_split'][i]:+.2f}", f"{P[n]['table_minus_fixed_pp_per_split'][i]:+.2f}",
                      f"{P[n]['greedy_minus_fixed_pp_per_split'][i]:+.2f}"]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "## Notes", "",
          "- Reference: runs/L5a/result.json gives the 3,000-label table minus plain fixed at +5.65 (L2b) and +8.38 (L4b). "
          "Those are the context values this run must reproduce.",
          "- greedy_prior is deterministic given the split: its initial scorer and every refit are fitted without randomness, "
          "so the tie-break seed only changes exact score ties. max_tie_seed_spread_hits per pool is in result.json.",
          "- All 8 split gaps share the one p0 fit, so the sd over splits covers eval-side variation only. The p0-label "
          "sampling variance that L7a covered with 20 subsets is not in this sd.",
          "- Per-split hit counts used in the independent check are under raw_hits in result.json.", ""]
    with open(os.path.join(out_dir, "table.md"), "w") as fh: fh.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_feats", required=True)
    ap.add_argument("--eval_feats", nargs="+", required=True)
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--k0", type=float, default=50.0)
    ap.add_argument("--splits", default="0-7")
    ap.add_argument("--tie_seeds", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    splits, seeds = parse_splits(a.splits), list(range(a.tie_seeds))

    fit_df = pd.read_parquet(a.fit_feats)
    if len(fit_df) != N_P0: sys.exit(f"fit set has {len(fit_df)} rows, expected {N_P0}")
    evals = {}
    for f in a.eval_feats:
        name = pool_name(f)
        if name not in REPRO_REF: sys.exit(f"no reproduction reference for pool {name}")
        evals[name] = f
    p0_questions = set(fit_df["question"])
    overlap = {n: int(pd.read_parquet(f)["question"].isin(p0_questions).sum()) for n, f in evals.items()}

    table = fit_table(fit_df, a.k0)
    print(f"comparator table: {len(table)} strata from {len(fit_df)} p0 prompts, k0 {a.k0:g}, no refit", flush=True)
    runs = {n: run_pool(f, table, a.budget, splits, seeds) for n, f in evals.items()}
    greedy = {n: run_pool_greedy(f, fit_df, a.budget, splits, seeds) for n, f in evals.items()}
    summ = {n: pool_summary(runs[n], splits) for n in evals}
    gsum = {n: greedy_summary(runs[n], greedy[n], splits) for n in evals}
    for n in evals:
        for i, s in enumerate(splits):
            print(f"{n} split {s}: greedy_prior minus table {gsum[n]['gap_vs_frozen_pp_per_split'][i]:+.2f}, table minus fixed "
                  f"{summ[n]['gap_pp_per_split'][i]:+.2f}, greedy_prior minus fixed {gsum[n]['gap_vs_fixed_pp_per_split'][i]:+.2f}",
                  flush=True)

    sg = spend_gate({n: runs[n] for n in evals})
    gg = greedy_spend(greedy)
    repro = {n: repro_gate(n, runs[n], splits) for n in evals}
    l5a = {n: l5a_gate(n, summ[n]) for n in evals}
    overlap_ok = all(v == 0 for v in overlap.values())
    gates_ok = (sg["pass"] and gg["pass"] and overlap_ok
                and all(g["pass"] for g in repro.values()) and all(g["pass"] for g in l5a.values()))
    gates = {"fixed_reproduction": repro, "frozen_3000_reproduction_L5a": l5a, "spend_gate": sg, "greedy_spend_gate": gg,
             "eval_question_overlap_zero": bool(overlap_ok), "l6b_subset_gate": "not applicable (no subsets in L9b)",
             "pass": bool(gates_ok)}
    for n, v in repro.items():
        print(f"reproduction {n}: max abs diff {v['max_abs_diff_pp']:.6f} pp (tol {REPRO_TOL_PP}), calls match "
              f"{v['calls_match']}, pass {v['pass']}", flush=True)
    for n, v in l5a.items():
        print(f"L5a reproduction {n}: 3,000-prompt frozen mean {v['mine_mean_pp']:+.2f} (L5a {v['ref_mean_pp']:+.2f}), "
              f"max abs diff {v['max_abs_diff_pp']:.6f} pp (tol {L5A_TOL_PP}), pass {v['pass']}", flush=True)
    print(f"spend gate: {sg['checks']} checks, failures {sg['failures']}; greedy spend: {gg['checks']} checks, failures "
          f"{gg['failures']}; eval overlap with p0 {overlap}; gates {'PASS' if gates_ok else 'FAIL'}", flush=True)
    if not gates_ok:
        write_json(a.out, {"experiment": "L9b", "verdict": "GATE_FAILED", "gates": gates})
        print("GATE FAILED: no L9b verdict written", flush=True); sys.exit(1)

    pools = {}
    for n in evals:
        m = np.asarray(gsum[n]["gap_vs_frozen_pp_per_split"], float)
        pools[n] = {"feats": evals[n], "n_splits": len(splits), "greedy_minus_table_pp_per_split": [float(x) for x in m],
                    "mean_pp": float(m.mean()), "sd_pp": float(m.std(ddof=1)), "n_splits_above_pass": int((m > PASS_PP).sum()),
                    "min_split_pp": float(m.min()), "max_split_pp": float(m.max()),
                    "table_minus_fixed_pp_per_split": summ[n]["gap_pp_per_split"],
                    "greedy_minus_fixed_pp_per_split": gsum[n]["gap_vs_fixed_pp_per_split"],
                    "table_minus_fixed_mean_pp": summ[n]["mean_pp"], "greedy_minus_fixed_mean_pp": gsum[n]["mean_vs_fixed_pp"],
                    "max_tie_seed_spread_hits": int(max(gsum[n]["tie_seed_spread_hits_per_split"])),
                    "fixed_pct_mean": float(np.mean(summ[n]["fixed_pct_per_split"])),
                    "table_pct_mean": float(np.mean(summ[n]["frozen_pct_per_split"])),
                    "greedy_pct_mean": float(np.mean(gsum[n]["greedy_pct_per_split"])),
                    "verdict": "PASS" if m.mean() > PASS_PP else "KILL"}
    raw = {n: {str(s): {"n_sim": int(runs[n][s]["n_sim"]), "budget": int(runs[n][s]["budget"]),
                        "fixed_hits": runs[n][s]["fixed_hits"], "frozen_hits": runs[n][s]["frozen_hits"],
                        "greedy_hits": greedy[n][s]["greedy_hits"]} for s in splits} for n in evals}
    out = {"experiment": "L9b", "label": LABEL, "group": GROUP, "stratum": STRATUM, "cheap": CHEAP, "k0": a.k0,
           "budget_frac": a.budget, "fit_feats": a.fit_feats, "fit_prompts": int(len(fit_df)),
           "comparator": "L5a 3,000-label frozen table: fit_table on all p0 rows, k0, prior 1 - agreement, never refit, frozen_pick",
           "greedy_policy": "greedy_prior: G01 greedy_online, K = 1, R = max(budget // 8, 40), initial scorer fitted on all 3,000 "
                            "p0 rows, refit on all p0 rows plus verified eval labels",
           "seeds": seeds, "splits": splits, "sd_ddof": 1, "pass_threshold_pp": PASS_PP, "tie_seeds": a.tie_seeds,
           "eval_question_overlap_with_p0": overlap, "gates": gates, "pools": pools,
           "verdicts": {n: pools[n]["verdict"] for n in evals}, "raw_hits": raw}
    out["verdict"] = "PASS" if all(v == "PASS" for v in out["verdicts"].values()) else "KILL"
    for n in pools:
        p = pools[n]
        print(f"{n}: greedy_prior minus 3,000-label table, mean over {p['n_splits']} splits {p['mean_pp']:+.2f}, sd over splits "
              f"{p['sd_pp']:.2f}, splits above +1.0 {p['n_splits_above_pass']} of {p['n_splits']}, min {p['min_split_pp']:+.2f} "
              f"max {p['max_split_pp']:+.2f}, table minus fixed {p['table_minus_fixed_mean_pp']:+.2f}, greedy minus fixed "
              f"{p['greedy_minus_fixed_mean_pp']:+.2f}, verdict {p['verdict']}", flush=True)
    print(f"L9b verdict: {out['verdict']} (per pool {out['verdicts']})", flush=True)
    write_json(a.out, out); write_table(a.out, out)
    print(f"wrote {os.path.join(a.out, 'result.json')} and table.md", flush=True)


if __name__ == "__main__":
    main()
