"""L10b: matched-precision control for L9a, at equal labels. CPU only, no GPU job.

Copied from avrl/exp/l9a_table_init_learned.py (L9a is unchanged). Change: the learned arm's eval-row posterior is built at the
same precision as the p0 out-of-fold posterior. Each eval row draws one fold f in 0 to 4 and is scored by the 160-label fold
table fitted without fold f, instead of the 200-label subset table. The frozen comparator stays the 200-label subset table.
The question: does the L9a KILL come from the 160-vs-200-label posterior mismatch, or from the refit design itself?

Arms per subset and pool, 10% budget, splits 0 to 7, 8 tie-break seeds each:
  frozen        the 200-label subset table, fixed ranking, never refit (L6b, unchanged).
  table_zero    greedy from the matched fold-table posterior with refits switched off. Gate 4 checks that it equals frozen_pick
                on that same score (replaces L9a's table_zero equals frozen, which does not hold by design here).
  table_refit   greedy from the matched fold-table posterior, refit every R = max(budget // 8, 40) verified labels: StandardScaler
                plus LogisticRegression(class_weight balanced, max_iter 1000) on the six cheap features plus the posterior,
                fitted on the 200 p0 labels plus every verified eval label so far.
  fixed         plain fixed threshold, as in runs/L2b and runs/L4b config B (unchanged).

Refit features. The 200 p0 rows get an out-of-fold table posterior: five balanced folds inside the subset, each row scored by
the table fitted on the other four folds (same k0, prior for missing strata), as in L9a. Each eval row gets the posterior of the
160-label fold table fitted without its drawn fold f. The draw is numpy RandomState(400 + i).randint(0, 5, size = eval pool rows),
one fold per eval row in file order, the same for every split and tie seed. No refit feature is built from the label it is
fitted to: p0 rows never see their own fold, and eval rows use no eval label.

ASSUMPTIONS (not in the spec, chosen here): the eval fold draw is RandomState(400 + i).randint(0, 5) per eval row, with one draw
per pool; the OOF fold seed is RandomState(300 + i) as in L9a; the zero-refit gate is replaced by the matched gate 4 below,
because the L9a zero-refit gate cannot hold once the init is the fold table. The refit is skipped after the last pick, since it
could not be used; this changes no pick.

Subset i draws 200 distinct p0 prompts without replacement with numpy RandomState(100 + i), as in L9a and L6b.
Metric: per pool, table_refit minus frozen at 10%. Per split, the mean over tie-break seeds of the hit-rate difference in pp.
A subset mean is the mean over split seeds 0 to 7. The pool value is the mean over the 20 subset means, with sd (ddof 1) and
the count of subset means above +1.0 pp. Context: table_zero minus frozen (matched precision, no refit), frozen minus plain
fixed, table_refit minus plain fixed, the L9a values, and the refitted B floor 40 reference.

Gates, before any verdict:
  1. Plain fixed at 10% reproduces runs/L2b/result.json (config B) and runs/L4b/result.json (config B40) per split to 0.01 pp.
  2. The 3,000-prompt frozen table reproduces runs/L5a/result.json gap per split on L2b and L4b to 1e-6 pp.
  3. The per-subset frozen gaps and row positions reproduce runs/L6b/result.json subset_detail to 1e-6 pp.
  4. Matched zero-refit: table_zero hits equal the frozen_pick order on the matched fold-table score, on every subset, split and
     tie seed (the greedy ranking with refits off is the frozen ranking on that score).
  5. Spend: every arm takes exactly int(budget * n_sim) distinct docs on every split, seed and subset (0 failures).
  6. Each subset has exactly subset_size distinct prompts, fixed picks are identical across all tables, and no eval question
     text occurs in p0.
Verdict per pool: PASS if the mean over the subset means is above +1.0 pp, else KILL. Overall PASS needs both pools.
Pass and kill are complementary. A gate failure is recorded as GATE_FAILED, not as a verdict.

Usage: one eval pool per call, then the combine step (avrl/exp/l10b_combine.py):
  python -I l10b_matched_precision.py --fit_feats FEATS --eval_feats FEATS --budget 0.10 --k0 50 --subsets 20
         --subset_size 200 --splits 0-7 --tie_seeds 8 --out OUTDIR
Writes <out>/result.json. If a gate fails, result.json gets verdict GATE_FAILED and the exit code is 1.
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
PASS_PP = 1.0  # L10b threshold, the same +1.0 pp as L9a (L6b used +2.0, L7a used +1.0)
REPRO_TOL_PP = 0.01
L5A_TOL_PP = 1e-6
L6B_TOL_PP = 1e-6
SUBSET_SEED0 = 100
OOF_SEED0 = 300
DRAW_SEED0 = 400  # eval-row fold draw, L10b only (assumption, not in the spec)
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


def p0_folds(n, i):
    """Five balanced fold labels for the n p0 rows of subset i, from RandomState(OOF_SEED0 + i) (unchanged from L9a)."""
    fold = np.empty(n, int)
    fold[np.random.RandomState(OOF_SEED0 + i).permutation(n)] = np.arange(n) % 5
    return fold


def fold_tables(sub, k0, i):
    """The five 160-label fold tables of subset i: entry f is fitted on the 200 p0 rows outside fold f."""
    fold = p0_folds(len(sub), i)
    return [fit_table(sub[fold != f], k0) for f in range(5)]


def oof_posterior(sub, k0, i):
    """Out-of-fold table posterior for the 200 p0 rows of subset i: five balanced folds inside the subset, each row scored by
    the table fitted on the other four folds. Same values as L9a."""
    fold = p0_folds(len(sub), i); tabs = fold_tables(sub, k0, i)
    out = np.empty(len(sub))
    for f in range(5):
        out[fold == f] = attach_score(sub[fold == f], tabs[f])["frozen_score"].to_numpy(float)
    return out


def matched_score(raw, tabs, draw):
    """Eval-row posterior at matched precision (L10b): row r is scored by the 160-label fold table tabs[draw[r]], where draw is
    one fold per eval row. No eval label is used. A missing stratum scores its prior, as in attach_score."""
    out = np.empty(len(raw))
    for f in range(5):
        m = draw == f
        out[m] = attach_score(raw[m], tabs[f])["frozen_score"].to_numpy(float)
    return out


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


def table_picks(y, ag, nd, X, init, Xp, yp, budget, seed, refit_every):
    """Greedy online allocator (G01 greedy_online, K = 1) started from the frozen table score init. Each step takes the top
    unseen doc in the frozen_pick key order (-score, agreement, -n_distinct, seeded draw). refit_every None means no refit, so
    the ranking stays the frozen table. Otherwise the scorer is refitted after every refit_every verified labels, on the 200 p0
    rows (out-of-fold posterior) plus every verified eval label so far (subset table posterior). Returns row positions in order."""
    rnd = np.random.default_rng(seed).random(len(y))
    seen = np.zeros(len(y), bool); score = init
    picks, next_fit = [], refit_every
    while len(picks) < budget:
        mu = np.flatnonzero(~seen)
        c = mu[np.lexsort((rnd[mu], -nd[mu], ag[mu], -score[mu]))[0]]
        seen[c] = True; picks.append(int(c))
        if refit_every and len(picks) >= next_fit and len(picks) < budget:
            score = scorer().fit(np.vstack([Xp, X[seen]]), np.concatenate([yp, y[seen]])).predict_proba(X)[:, 1]
            next_fit += refit_every
    return np.asarray(picks)


def job_table(args):
    feats, split, tabs, draw, Xp, yp, frac, seeds = args
    assert 0 < yp.sum() < len(yp), "p0 subset must hold both classes"
    raw = pd.read_parquet(feats)
    sim, _ = L1B.build_sim(raw.assign(matched_score=matched_score(raw, tabs, draw)), GROUP, split)
    y = sim[LABEL].to_numpy(int); ag = sim["agreement"].values; nd = sim["n_distinct"].values
    init = sim["matched_score"].to_numpy(float)
    X = np.column_stack([sim[CHEAP].fillna(0).to_numpy(float), init])
    S = L1B.Sim(sim, LABEL)
    budget = int(frac * len(y)); R = max(budget // 8, 40)
    zero, learned, matched, spend_bad = [], [], [], []
    for s in seeds:
        sel_z = table_picks(y, ag, nd, X, init, Xp, yp, budget, s, None)
        sel_l = table_picks(y, ag, nd, X, init, Xp, yp, budget, s, R)
        sel_m = frozen_pick(S, init, budget, s)
        zero.append(int(y[sel_z].sum())); learned.append(int(y[sel_l].sum())); matched.append(int(y[sel_m].sum()))
        for name, sel in (("table_zero", sel_z), ("table_refit", sel_l)):
            d = int(len(np.unique(sel)))
            if len(sel) != budget or d != budget:
                spend_bad.append({"policy": name, "seed": int(s), "distinct": d, "expected": budget})
    return {"split": split, "budget": budget, "zero_hits": zero, "learned_hits": learned, "matched_pick_hits": matched,
            "spend_bad": spend_bad}


def run_pool_table(feats, tabs, draw, Xp, yp, frac, splits, seeds):
    with Pool(min(len(splits), os.cpu_count())) as p:
        res = p.map(job_table, [(feats, s, tabs, draw, Xp, yp, frac, seeds) for s in splits])
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


def table_summary(R, T, splits):
    """Per split, pp, mean over tie-break seeds: table_refit minus frozen (primary), table_refit minus plain fixed (context),
    table_zero minus frozen (context: matched precision with refits off), and the spread of table_refit hits across tie seeds."""
    vs_frozen, vs_fixed, zero_vs_frozen, pct, spread = [], [], [], [], []
    for s in splits:
        b = R[s]["budget"]; assert T[s]["budget"] == b
        l = 100.0 * np.asarray(T[s]["learned_hits"], float) / b
        q = 100.0 * np.asarray(T[s]["zero_hits"], float) / b
        z = 100.0 * np.asarray(R[s]["frozen_hits"], float) / b
        f = 100.0 * np.asarray(R[s]["fixed_hits"], float) / b
        vs_frozen.append(float((l - z).mean())); vs_fixed.append(float((l - f).mean()))
        zero_vs_frozen.append(float((q - z).mean()))
        pct.append(float(l.mean())); spread.append(int(max(T[s]["learned_hits"]) - min(T[s]["learned_hits"])))
    return {"gap_vs_frozen_pp_per_split": vs_frozen, "gap_vs_fixed_pp_per_split": vs_fixed, "learned_pct_per_split": pct,
            "zero_minus_frozen_pp_per_split": zero_vs_frozen, "mean_zero_minus_frozen_pp": float(np.mean(zero_vs_frozen)),
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


def matched_zero_gate(name, subs, splits):
    """Matched zero-refit gate (replaces L9a's zero-refit gate, which cannot hold with a fold-table init): the greedy ranking with
    refits off must equal the frozen_pick order on the matched fold-table score, so table_zero hits equal matched_pick_hits on
    every subset, split and tie-break seed."""
    same = all(x["tab"][name][s]["zero_hits"] == x["tab"][name][s]["matched_pick_hits"] for x in subs for s in splits)
    return {"subsets": len(subs), "splits": len(splits), "hits_equal_frozen_pick_on_matched_score": bool(same), "pass": bool(same)}


def spend_gate(pools):
    bad = [dict(pool=name, split=s, **b) for name, R in pools.items() for s in R for b in R[s]["spend_bad"]]
    checks = sum(2 * len(R[next(iter(R))]["fixed_hits"]) * len(R) for R in pools.values())
    return {"checks": checks, "failures": len(bad), "examples": bad[:5], "pass": not bad}


def table_spend(subs, names):
    bad = [dict(pool=n, subset=x["subset"], split=s, **b)
           for x in subs for n in names for s in x["tab"][n] for b in x["tab"][n][s]["spend_bad"]]
    checks = sum(2 * len(x["tab"][n][s]["learned_hits"]) for x in subs for n in names for s in x["tab"][n])
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
    n_eval = {n: len(pd.read_parquet(f)) for n, f in evals.items()}

    full_table = fit_table(fit_df, a.k0)
    print(f"gate table: {len(full_table)} strata from {len(fit_df)} p0 prompts, k0 {a.k0:g}, no refit", flush=True)
    full = {n: run_pool(f, full_table, a.budget, splits, seeds) for n, f in evals.items()}
    full_sum = {n: pool_summary(full[n], splits) for n in evals}

    subs = []
    for i in range(a.subsets):
        idx = draw_subset(fit_df, i, a.subset_size)
        sub = fit_df.iloc[idx]
        table = fit_table(sub, a.k0)
        tabs = fold_tables(sub, a.k0, i)  # the five 160-label fold tables, used for eval rows only (matched precision)
        draw = {n: np.random.RandomState(DRAW_SEED0 + i).randint(0, 5, size=n_eval[n]) for n in evals}
        Xp = np.column_stack([sub[CHEAP].fillna(0).to_numpy(float), oof_posterior(sub, a.k0, i)])
        yp = sub[LABEL].to_numpy(int)
        runs = {n: run_pool(f, table, a.budget, splits, seeds) for n, f in evals.items()}
        tab = {n: run_pool_table(f, tabs, draw[n], Xp, yp, a.budget, splits, seeds) for n, f in evals.items()}
        summ = {n: pool_summary(runs[n], splits) for n in evals}
        tsum = {n: table_summary(runs[n], tab[n], splits) for n in evals}
        subs.append({"subset": i, "row_positions": [int(x) for x in idx], "distinct_pids": int(sub["pid"].nunique()),
                     "strata_in_table": int(len(table)), "eval_fold_counts": {n: np.bincount(draw[n], minlength=5).tolist()
                                                                              for n in evals},
                     "runs": runs, "summary": summ, "tab": tab, "tab_summary": tsum})
        print(f"subset {i:2d}: {len(table)} strata, frozen minus fixed " + ", ".join(f"{n} {summ[n]['mean_pp']:+.2f}" for n in evals)
              + ", table_refit minus frozen " + ", ".join(f"{n} {tsum[n]['mean_vs_frozen_pp']:+.2f}" for n in evals)
              + ", table_zero minus frozen " + ", ".join(f"{n} {tsum[n]['mean_zero_minus_frozen_pp']:+.2f}" for n in evals), flush=True)

    fixed_same = all(R[s]["fixed_hits"] == full[n][s]["fixed_hits"]
                     for x in subs for n, R in x["runs"].items() for s in splits)
    sg = spend_gate({**{f"{n}/gate": full[n] for n in evals},
                     **{f"{n}/subset{x['subset']}": x["runs"][n] for x in subs for n in evals}})
    tg = table_spend(subs, list(evals))
    repro = {n: repro_gate(n, full[n], splits) for n in evals}
    l5a = {n: l5a_gate(n, full_sum[n]) for n in evals}
    l6b = {n: l6b_gate(n, subs) for n in evals}
    zg = {n: matched_zero_gate(n, subs, splits) for n in evals}
    subset_ok = all(x["distinct_pids"] == a.subset_size and len(set(x["row_positions"])) == a.subset_size for x in subs)
    overlap_ok = all(v == 0 for v in overlap.values())
    gates_ok = (sg["pass"] and tg["pass"] and subset_ok and overlap_ok and fixed_same
                and all(g["pass"] for g in repro.values()) and all(g["pass"] for g in l5a.values())
                and all(g["pass"] for g in l6b.values()) and all(g["pass"] for g in zg.values()))
    out = {"experiment": "L10b", "label": LABEL, "group": GROUP, "stratum": STRATUM, "cheap": CHEAP,
           "refit_features": CHEAP + ["table_posterior"], "k0": a.k0,
           "budget_frac": a.budget, "fit_feats": a.fit_feats, "fit_prompts": int(len(fit_df)), "subsets": a.subsets,
           "subset_size": a.subset_size, "subset_draw": "numpy RandomState(100 + i).choice(n_p0_rows, subset_size, replace=False)",
           "oof_folds": "5 balanced folds inside each 200-label subset, numpy RandomState(300 + i).permutation (as L9a)",
           "eval_fold_draw": "ASSUMPTION (not in spec): per eval pool, one fold f in 0..4 per eval row, numpy "
                             "RandomState(400 + i).randint(0, 5, size = eval pool rows), file row order, same for all splits and seeds",
           "eval_posterior": "eval rows: posterior of the 160-label fold table fitted without the drawn fold (matched precision); "
                             "p0 rows: out-of-fold posterior as in L9a; frozen arm: 200-label subset table (as L9a)",
           "frozen_refit": "none (each subset table is frozen after its fit)",
           "table_policy": "table_refit: G01 greedy_online, K = 1, initial score = matched fold-table posterior of each eval row, key "
                           "(-score, agreement, -n_distinct, seeded draw), R = max(budget // 8, 40), refit = StandardScaler plus "
                           "LogisticRegression(class_weight balanced, max_iter 1000) on six cheap features plus posterior, "
                           "fitted on the subset's 200 p0 labels (out-of-fold posterior) plus verified eval labels (matched posterior)",
           "seeds": seeds, "splits": splits, "sd_ddof": 1, "pass_threshold_pp": PASS_PP, "tie_seeds": a.tie_seeds,
           "eval_question_overlap_with_p0": overlap,
           "gates": {"fixed_reproduction": repro, "frozen_3000_reproduction_L5a": l5a, "frozen_subset_reproduction_L6b": l6b,
                     "matched_zero_gate": zg, "spend_gate": sg, "table_spend_gate": tg,
                     "fixed_identical_across_tables": bool(fixed_same),
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
    for n, v in zg.items():
        print(f"matched zero-refit gate {n}: table_zero equals frozen_pick on the matched score {v['hits_equal_frozen_pick_on_matched_score']}, "
              f"pass {v['pass']}", flush=True)
    print(f"spend gate: {sg['checks']} checks, failures {sg['failures']}; table spend: {tg['checks']} checks, failures "
          f"{tg['failures']}; subsets distinct {subset_ok}; fixed identical {fixed_same}; eval overlap with p0 {overlap}; "
          f"gates {'PASS' if gates_ok else 'FAIL'}", flush=True)
    if not gates_ok:
        out["verdict"] = "GATE_FAILED"; write_json(a.out, out)
        print("GATE FAILED: no L10b verdict written", flush=True); sys.exit(1)

    pools = {}
    for n in evals:
        ts = [x["tab_summary"][n] for x in subs]; sm = [x["summary"][n] for x in subs]
        means = [t["mean_vs_frozen_pp"] for t in ts]; m = np.asarray(means, float)
        ref = refit_ref(n)
        pools[n] = {"feats": evals[n], "mean_pp": float(m.mean()), "sd_pp": float(m.std(ddof=1)), "n_subsets": len(m),
                    "n_subsets_above_pass": int((m > PASS_PP).sum()),
                    "n_subsets_at_or_below_pass": int((m <= PASS_PP).sum()),
                    "min_subset_mean_pp": float(m.min()), "max_subset_mean_pp": float(m.max()),
                    "subset_mean_pp": means, "gap_pp_per_subset": [t["gap_vs_frozen_pp_per_split"] for t in ts],
                    "n_split_gaps": int(sum(len(t["gap_vs_frozen_pp_per_split"]) for t in ts)),
                    "n_split_gaps_above_pass": int(sum((np.asarray(t["gap_vs_frozen_pp_per_split"]) > PASS_PP).sum() for t in ts)),
                    "max_tie_seed_spread_hits": int(max(max(t["tie_seed_spread_hits_per_split"]) for t in ts)),
                    "learned_pct_mean": float(np.mean([np.mean(t["learned_pct_per_split"]) for t in ts])),
                    "fixed_pct_mean": float(np.mean([np.mean(x["fixed_pct_per_split"]) for x in sm])),
                    "frozen_pct_mean": float(np.mean([np.mean(x["frozen_pct_per_split"]) for x in sm])),
                    "frozen_vs_fixed_mean_pp": float(np.mean([x["mean_pp"] for x in sm])),
                    "learned_vs_fixed_mean_pp": float(np.mean([t["mean_vs_fixed_pp"] for t in ts])),
                    "zero_minus_frozen_mean_pp": float(np.mean([t["mean_zero_minus_frozen_pp"] for t in ts])),
                    "zero_minus_frozen_pp_per_split": [t["zero_minus_frozen_pp_per_split"] for t in ts],
                    "full_table_3000_mean_pp": full_sum[n]["mean_pp"], "refit_b_floor40": ref,
                    "learned_minus_refit_b40_pp": float(m.mean() - ref["mean_pp"]),
                    "verdict": "PASS" if m.mean() > PASS_PP else "KILL"}
    out["pools"] = pools
    out["verdicts"] = {n: pools[n]["verdict"] for n in evals}
    out["verdict"] = "PASS" if all(v == "PASS" for v in out["verdicts"].values()) else "KILL"
    out["subset_detail"] = [dict({"subset": x["subset"], "row_positions": x["row_positions"], "distinct_pids": x["distinct_pids"],
                                  "strata_in_table": x["strata_in_table"]},
                                 **{f"{n}_table_refit_minus_frozen_pp": x["tab_summary"][n]["mean_vs_frozen_pp"] for n in evals},
                                 **{f"{n}_table_refit_minus_frozen_pp_per_split": x["tab_summary"][n]["gap_vs_frozen_pp_per_split"]
                                    for n in evals},
                                 **{f"{n}_frozen_minus_fixed_pp": x["summary"][n]["mean_pp"] for n in evals},
                                 **{f"{n}_gap_pp_per_split": x["summary"][n]["gap_pp_per_split"] for n in evals},
                                 **{f"{n}_table_refit_minus_fixed_pp": x["tab_summary"][n]["mean_vs_fixed_pp"] for n in evals},
                                 **{f"{n}_eval_fold_counts": x["eval_fold_counts"][n] for n in evals})
                           for x in subs]
    for n in pools:
        p = pools[n]
        print(f"{n}: table_refit minus frozen 200-label, mean over {p['n_subsets']} subsets {p['mean_pp']:+.2f}, sd over subset "
              f"means {p['sd_pp']:.2f}, subsets above +1.0 {p['n_subsets_above_pass']} of {p['n_subsets']}, min "
              f"{p['min_subset_mean_pp']:+.2f} max {p['max_subset_mean_pp']:+.2f}, frozen minus fixed {p['frozen_vs_fixed_mean_pp']:+.2f}, "
              f"table_refit minus fixed {p['learned_vs_fixed_mean_pp']:+.2f}, table_zero minus frozen {p['zero_minus_frozen_mean_pp']:+.2f}, "
              f"verdict {p['verdict']}", flush=True)
    print(f"L10b verdict: {out['verdict']} (per pool {out['verdicts']})", flush=True)
    write_json(a.out, out)
    print(f"wrote {os.path.join(a.out, 'result.json')}", flush=True)


if __name__ == "__main__":
    main()
