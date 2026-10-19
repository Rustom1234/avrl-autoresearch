"""L11a: L10a table-pooled stratum posterior on the 1,319 registered L4b test rows (pid <= 1318), at equal labels. CPU only, no GPU job.

Copied from avrl/exp/l10a_table_pooled_posterior.py (L10a is unchanged). Changes: eval rows are cut to pid <= --eval_pid_max
(default 1318) before any cell100 split, as in avrl/exp/l6a_test_rows.py, and the reproduction gates reference runs/L6a (the
same 1,319 rows) instead of the full-frame runs L2b, L4b, L5a and L6b. The learned arm has no logistic model and no p0
refit. It is a per-stratum posterior that pools the frozen 200-label table posterior as prior counts with the verified eval
labels as counts:

    score(doc) = (e + k0 * table_post) / (v + k0)

where table_post is the frozen 200-label posterior of the doc's stratum (1 - agreement when the stratum is absent from the
table), e and v are the verified eval errors and count in that stratum so far, and k0 = 50 is the pseudo-count strength on the
table prior mean. The p0 labels enter only through table_post, so no out-of-fold posterior is needed.

Arms per subset and pool, 10% budget, splits 0 to 7, 8 tie-break seeds each:
  frozen        the 200-label table, fixed ranking, never refit (L6b, unchanged).
  table_zero    greedy from the table with score updates switched off. Must equal frozen exactly (gate).
  table_pooled  greedy online from the table: each step takes the top unseen doc in the key order (-score, agreement,
                -n_distinct, seeded draw). The stratum scores are recomputed every R = max(budget // 8, 40) = 40 verified labels.
  fixed         plain fixed threshold, as in runs/L2b and runs/L4b config B (unchanged).

ASSUMPTIONS: the score update is applied after every R verified labels (the cadence is taken from the spec, the update rule
from the formula above). The update is skipped after the last pick, since it could not be used; this changes no pick. The
zero-update arm is the same greedy code with updates off, not a separate code path, so the gate tests that the greedy
ranking equals frozen_pick.

Subset i draws 200 distinct p0 prompts without replacement with numpy RandomState(100 + i), as in L6b, L7a and L9a.
Metric: per pool, table_pooled minus frozen at 10%. Per split, the mean over tie-break seeds of the hit-rate difference in pp.
A subset mean is the mean over split seeds 0 to 7. The pool value is the mean over the 20 subset means, with sd (ddof 1) and
the count of subset means above +1.0 pp. Context: frozen minus plain fixed, table_pooled minus plain fixed, and the count of
split-level gaps above +1.0 pp out of 160 (20 subsets times 8 splits).

Gates, before any verdict:
  1. The 3,000-prompt frozen table on the 1,319 test rows reproduces runs/L6a/result.json: per-split gap and mean plain
     fixed rate, to 1e-6.
  2. The per-subset frozen gaps and row positions reproduce runs/L6a/result.json subset_detail to 1e-6 pp.
  3. Zero-update: table_zero hits equal frozen hits on every subset, split and tie seed, and the per-subset gap over plain fixed
     reproduces runs/L6a/result.json to 1e-6 pp.
  4. Spend: every arm takes exactly int(budget * n_sim) distinct docs on every split, seed and subset (0 failures).
  5. Each subset has exactly subset_size distinct prompts, fixed picks are identical across all tables, and no eval question
     text occurs in p0.
Verdict: PASS if the mean over the subset means is above +1.0 pp, else KILL.
Pass and kill are complementary. A gate failure is recorded as GATE_FAILED, not as a verdict.

Usage (one L4b pool, no combine step):
  python -I l11a_table_pooled_test_rows.py --fit_feats P0_FEATS --eval_feats L4B_FEATS --eval_pid_max 1318 --budget 0.10
         --k0 50 --subsets 20 --subset_size 200 --splits 0-7 --tie_seeds 8 --out OUTDIR
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
PASS_PP = 1.0  # L11a threshold (same as L10a, L9a and L7a)
TEST_PID_MAX = 1318  # registered test rows: pids 0 to 1318, 1,319 prompts
REPRO_POOL = "L4b"  # the only pool in L11a
L6A_TOL_PP = 1e-6
SUBSET_SEED0 = 100
# Reproduction reference for the test rows: runs/L6a holds the frozen comparator on the same 1,319 rows and the same subset draws
L6A_REF = f"{RUNS}/L6a/result.json"


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def parse_splits(s):
    if "-" in s:
        lo, hi = s.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def pool_name(path):
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


def load_eval(path, pid_max):
    """Registered test rows: eval pids 0 to pid_max, cut before any cell100 split (as in runs/L6a)."""
    d = pd.read_parquet(path)
    return d[d["pid"] <= pid_max].reset_index(drop=True)


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


def run_pool(df_in, table, frac, splits, seeds):
    with Pool(min(len(splits), os.cpu_count())) as p:
        res = p.map(job, [(df_in, s, table, frac, seeds) for s in splits])
    return {r["split"]: r for r in res}


def pooled_scores(code, init, seen, y, k0):
    """Stratum posterior for every doc: k0 prior counts at the frozen table posterior of its stratum (init), plus the verified
    eval errors (e) and count (v) of its stratum so far. code holds the stratum index of each doc."""
    K = int(code.max()) + 1
    e = np.bincount(code[seen], weights=y[seen].astype(float), minlength=K)
    v = np.bincount(code[seen], minlength=K).astype(float)
    return (e[code] + k0 * init) / (v[code] + k0)


def table_picks(y, ag, nd, code, init, budget, seed, refit_every, k0):
    """Greedy online allocator (G01 greedy_online, K = 1) started from the frozen table score init. Each step takes the top
    unseen doc in the frozen_pick key order (-score, agreement, -n_distinct, seeded draw). refit_every None means no update, so
    the ranking stays the frozen table. Otherwise the stratum scores are recomputed after every refit_every verified labels.
    Returns row positions in pick order."""
    rnd = np.random.default_rng(seed).random(len(y))
    seen = np.zeros(len(y), bool); score = init
    picks, next_fit = [], refit_every
    while len(picks) < budget:
        mu = np.flatnonzero(~seen)
        c = mu[np.lexsort((rnd[mu], -nd[mu], ag[mu], -score[mu]))[0]]
        seen[c] = True; picks.append(int(c))
        if refit_every and len(picks) >= next_fit and len(picks) < budget:
            score = pooled_scores(code, init, seen, y, k0)
            next_fit += refit_every
    return np.asarray(picks)


def job_table(args):
    df_in, split, table, frac, seeds, k0 = args
    df = attach_score(df_in, table)
    sim, _ = L1B.build_sim(df, GROUP, split)
    y = sim[LABEL].to_numpy(int); ag = sim["agreement"].values; nd = sim["n_distinct"].values
    init = sim["frozen_score"].to_numpy(float)
    code = sim.groupby(STRATUM).ngroup().to_numpy(int)
    budget = int(frac * len(y)); R = max(budget // 8, 40)
    zero, pooled, spend_bad = [], [], []
    for s in seeds:
        sel_z = table_picks(y, ag, nd, code, init, budget, s, None, k0)
        sel_p = table_picks(y, ag, nd, code, init, budget, s, R, k0)
        zero.append(int(y[sel_z].sum())); pooled.append(int(y[sel_p].sum()))
        for name, sel in (("table_zero", sel_z), ("table_pooled", sel_p)):
            d = int(len(np.unique(sel)))
            if len(sel) != budget or d != budget:
                spend_bad.append({"policy": name, "seed": int(s), "distinct": d, "expected": budget})
    return {"split": split, "budget": budget, "zero_hits": zero, "pooled_hits": pooled, "spend_bad": spend_bad}


def run_pool_table(df_in, table, frac, splits, seeds, k0):
    with Pool(min(len(splits), os.cpu_count())) as p:
        res = p.map(job_table, [(df_in, s, table, frac, seeds, k0) for s in splits])
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
    """Per split, pp, mean over tie-break seeds: table_pooled minus frozen (primary), table_pooled minus plain fixed (context),
    and the spread of table_pooled hits across tie-break seeds."""
    vs_frozen, vs_fixed, pct, spread = [], [], [], []
    for s in splits:
        b = R[s]["budget"]; assert T[s]["budget"] == b
        pl = 100.0 * np.asarray(T[s]["pooled_hits"], float) / b
        z = 100.0 * np.asarray(R[s]["frozen_hits"], float) / b
        f = 100.0 * np.asarray(R[s]["fixed_hits"], float) / b
        vs_frozen.append(float((pl - z).mean())); vs_fixed.append(float((pl - f).mean()))
        pct.append(float(pl.mean())); spread.append(int(max(T[s]["pooled_hits"]) - min(T[s]["pooled_hits"])))
    return {"gap_vs_frozen_pp_per_split": vs_frozen, "gap_vs_fixed_pp_per_split": vs_fixed, "pooled_pct_per_split": pct,
            "tie_seed_spread_hits_per_split": spread, "mean_vs_frozen_pp": float(np.mean(vs_frozen)),
            "mean_vs_fixed_pp": float(np.mean(vs_fixed))}


def l6a_split_gate(summary):
    """3,000-prompt frozen table on the test rows against runs/L6a: per-split gap and the mean plain fixed rate."""
    ref = json.load(open(L6A_REF))["table_3000"]
    mine = summary["gap_pp_per_split"]; theirs = [float(x) for x in ref["gap_pp_per_split"]]
    fixed = float(np.mean(summary["fixed_pct_per_split"])); ref_fixed = float(ref["fixed_pct_mean"])
    d = max(max(abs(m - t) for m, t in zip(mine, theirs)), abs(fixed - ref_fixed))
    return {"ref": L6A_REF, "ref_fixed_pct_mean": ref_fixed, "mine_fixed_pct_mean": fixed, "ref_gap_pp_per_split": theirs,
            "mine_gap_pp_per_split": mine, "max_abs_diff": float(d), "tol": L6A_TOL_PP,
            "pass": bool(d <= L6A_TOL_PP and len(mine) == len(theirs))}


def l6a_subset_gate(name, subs):
    """Per-subset frozen gaps (plain fixed as base) and row positions against runs/L6a/result.json subset_detail."""
    ref = json.load(open(L6A_REF))["subset_detail"]
    worst, pos_ok = 0.0, len(ref) == len(subs)
    for x in subs:
        r = ref[x["subset"]]
        pos_ok = pos_ok and r["row_positions"] == x["row_positions"] and r["distinct_pids"] == x["distinct_pids"]
        mine = x["summary"][name]["gap_pp_per_split"]; theirs = [float(v) for v in r["gap_pp_per_split"]]
        worst = max(worst, max(abs(m - t) for m, t in zip(mine, theirs)))
    return {"ref": L6A_REF, "subsets": len(subs), "row_positions_match": bool(pos_ok), "max_abs_diff_pp": float(worst),
            "tol_pp": L6A_TOL_PP, "pass": bool(pos_ok and worst <= L6A_TOL_PP)}


def zero_refit_gate(name, subs, splits):
    """The zero-update arm (greedy from the table, no update) must equal the frozen table: same hits on every split and tie-break
    seed, and the per-subset gap over plain fixed must match runs/L6a subset_detail to 1e-6 pp."""
    ref = json.load(open(L6A_REF))["subset_detail"]
    worst, same, pos_ok = 0.0, True, len(ref) == len(subs)
    for x in subs:
        r = ref[x["subset"]]; R, T = x["runs"][name], x["tab"][name]
        gaps = []
        for s in splits:
            same = same and T[s]["zero_hits"] == R[s]["frozen_hits"]
            b = R[s]["budget"]
            gaps.append(float(np.mean(100.0 * np.asarray(T[s]["zero_hits"], float) / b
                                      - 100.0 * np.asarray(R[s]["fixed_hits"], float) / b)))
        theirs = [float(v) for v in r["gap_pp_per_split"]]
        worst = max(worst, max(abs(m - t) for m, t in zip(gaps, theirs)))
    return {"ref": L6A_REF, "subsets": len(subs), "hits_equal_frozen": bool(same), "max_abs_diff_pp": float(worst),
            "tol_pp": L6A_TOL_PP, "pass": bool(same and pos_ok and worst <= L6A_TOL_PP)}


def spend_gate(pools):
    bad = [dict(pool=name, split=s, **b) for name, R in pools.items() for s in R for b in R[s]["spend_bad"]]
    checks = sum(2 * len(R[next(iter(R))]["fixed_hits"]) * len(R) for R in pools.values())
    return {"checks": checks, "failures": len(bad), "examples": bad[:5], "pass": not bad}


def table_spend(subs, names):
    bad = [dict(pool=n, subset=x["subset"], split=s, **b)
           for x in subs for n in names for s in x["tab"][n] for b in x["tab"][n][s]["spend_bad"]]
    checks = sum(2 * len(x["tab"][n][s]["pooled_hits"]) for x in subs for n in names for s in x["tab"][n])
    return {"checks": checks, "failures": len(bad), "examples": bad[:5], "pass": not bad}


def draw_subset(fit_df, i, size):
    """Row positions of label subset i: size distinct p0 prompts, numpy RandomState(100 + i), without replacement."""
    return np.random.RandomState(SUBSET_SEED0 + i).choice(len(fit_df), size=size, replace=False)


def write_json(out_dir, out):
    with open(os.path.join(out_dir, "result.json"), "w") as fh: json.dump(out, fh, indent=1, default=_jsonable)


def write_table(out_dir, r):
    P = r["pools"]; names = list(P); G = r["gates"]
    L = ["# L11a: table-pooled stratum posterior on the 1,319 registered L4b test rows, equal labels", "",
         f"Rule: per pool, table_pooled minus the frozen 200-label config B table at 10%, paired by split seed and tie-break "
         f"seed. A subset mean is the mean over split seeds 0 to 7 ({r['tie_seeds']} tie-break seeds each). PASS per pool if "
         f"the mean over the {r['subsets']} subset means is above +{PASS_PP:.1f} pp, else KILL.",
         "", f"Overall verdict: **{r['verdict']}**", "", "## Gates", "", "| gate | result |", "|---|---|"]
    for name, g in G["test_rows_reproduction_L6a"].items():
        L.append(f"| 3,000-prompt frozen table on the test rows reproduces runs/L6a/result.json on {name} | "
                 f"{'PASS' if g['pass'] else 'FAIL'} (max abs diff {g['max_abs_diff']:.6f}, tol {L6A_TOL_PP:g}; mean plain fixed "
                 f"{g['mine_fixed_pct_mean']:.4f}% vs {g['ref_fixed_pct_mean']:.4f}%) |")
    for name, g in G["frozen_subset_reproduction_L6a"].items():
        L.append(f"| per-subset frozen gaps reproduce runs/L6a/result.json subset_detail on {name} | {'PASS' if g['pass'] else 'FAIL'} "
                 f"(max abs diff {g['max_abs_diff_pp']:.6f} pp, tol {L6A_TOL_PP:g}, row positions match {g['row_positions_match']}) |")
    for name, g in G["zero_refit_reproduction"].items():
        L.append(f"| zero-update table arm equals frozen table on {name} | {'PASS' if g['pass'] else 'FAIL'} "
                 f"(hits equal frozen {g['hits_equal_frozen']}, max abs diff vs runs/L6a/result.json {g['max_abs_diff_pp']:.6f} pp, "
                 f"tol {L6A_TOL_PP:g}) |")
    for key, label in (("spend_gate", "spend gate, fixed and frozen"), ("table_spend_gate", "spend gate, table_zero and table_pooled")):
        sg = G[key]; L.append(f"| {label} | {'PASS' if sg['pass'] else 'FAIL'} ({sg['checks']} checks, failures {sg['failures']}) |")
    L.append(f"| every subset has {r['subset_size']} distinct prompts | {'PASS' if G['subset_distinct_prompts'] else 'FAIL'} |")
    L.append(f"| fixed picks identical across all tables | {'PASS' if G['fixed_identical_across_tables'] else 'FAIL'} |")
    L.append(f"| eval question text found in p0 | {'PASS (0 rows)' if G['eval_question_overlap_zero'] else 'FAIL'} |")
    L += ["", "## Verdict and context", "",
          "| pool | table_pooled minus frozen 200-label, mean over subsets (pp) | sd over subset means (pp) | subsets above +1.0 | "
          "subsets at or below +1.0 | min to max subset mean (pp) | split gaps above +1.0 | frozen 200-label minus plain fixed (pp) | "
          "table_pooled minus plain fixed (pp) | plain fixed hit rate (%) | table_pooled hit rate (%) | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        p = P[n]
        L.append(f"| {n} | {p['mean_pp']:+.2f} | {p['sd_pp']:.2f} | {p['n_subsets_above_pass']} of {p['n_subsets']} | "
                 f"{p['n_subsets_at_or_below_pass']} of {p['n_subsets']} | {p['min_subset_mean_pp']:+.2f} to "
                 f"{p['max_subset_mean_pp']:+.2f} | {p['n_split_gaps_above_pass']} of {p['n_split_gaps']} | "
                 f"{p['frozen_vs_fixed_mean_pp']:+.2f} | {p['pooled_vs_fixed_mean_pp']:+.2f} | "
                 f"{p['fixed_pct_mean']:.2f} | {p['pooled_pct_mean']:.2f} | **{p['verdict']}** |")
    hdr = ["subset", "strata in table", "distinct prompts"] + [f"{n} {c}" for n in names for c in
                                                             ("table_pooled minus frozen", "frozen minus fixed", "table_pooled minus fixed")]
    L += ["", "## Subsets: mean gap over split seeds 0 to 7 (pp)", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for s in r["subset_detail"]:
        cells = [str(s["subset"]), str(s["strata_in_table"]), str(s["distinct_pids"])]
        for n in names:
            cells += [f"{s[n + '_table_pooled_minus_frozen_pp']:+.2f}", f"{s[n + '_frozen_minus_fixed_pp']:+.2f}",
                      f"{s[n + '_table_pooled_minus_fixed_pp']:+.2f}"]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "## Reference rows", "", "| pool | source | value (pp) |", "|---|---|---|"]
    for n in names:
        L.append(f"| {n} | 3,000-prompt frozen table on the test rows (runs/L6a/result.json table_3000), frozen minus plain fixed | "
                 f"{P[n]['full_table_3000_mean_pp']:+.2f} |")
    L += ["", "## Notes", "",
          "- table_pooled starts from the frozen table score. Until the first update, the tie-break seed orders docs inside each "
          "stratum (their scores tie). After updates, it matters only for exact score ties. Spread across tie-break seeds is in "
          "result.json under pools[pool].max_tie_seed_spread_hits.",
          "- table_zero (updates off) equals the frozen table on every split and seed; this is gate 3.",
          "- Each subset's frozen table is the L6a table for the same subset (same draw). Its per-split gaps are reproduced from runs/L6a/result.json.",
          "- table_pooled score per stratum is (e + k0 * table posterior) / (v + k0), with e and v the verified eval errors and count in "
          "that stratum so far, k0 = 50, recomputed every R = 40 verified labels. No logistic model and no p0 refit.",
          "- Subset means are the mean over split seeds 0 to 7 of the paired gap. The full per-split values are in result.json.", ""]
    with open(os.path.join(out_dir, "table.md"), "w") as fh: fh.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_feats", required=True)
    ap.add_argument("--eval_feats", nargs="+", required=True)
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
    evals = {}
    for f in a.eval_feats:
        name = pool_name(f)
        if name != REPRO_POOL: sys.exit(f"L11a reproduces only pool {REPRO_POOL} (runs/L6a); got {name}")
        evals[name] = f
    p0_questions = set(fit_df["question"])
    frames = {n: load_eval(f, a.eval_pid_max) for n, f in evals.items()}
    overlap = {n: int(frames[n]["question"].isin(p0_questions).sum()) for n in evals}
    test_prompts = {n: int(len(frames[n])) for n in evals}
    test_err = {n: float(frames[n][LABEL].mean()) for n in evals}
    print(f"test rows {test_prompts} (pid <= {a.eval_pid_max}), test proxy error {test_err}, p0 fit prompts {len(fit_df)}, "
          f"test question overlap with p0 {overlap}", flush=True)

    full_table = fit_table(fit_df, a.k0)
    print(f"gate table: {len(full_table)} strata from {len(fit_df)} p0 prompts, k0 {a.k0:g}, no update", flush=True)
    full = {n: run_pool(frames[n], full_table, a.budget, splits, seeds) for n in evals}
    full_sum = {n: pool_summary(full[n], splits) for n in evals}

    subs = []
    for i in range(a.subsets):
        idx = draw_subset(fit_df, i, a.subset_size)
        sub = fit_df.iloc[idx]
        table = fit_table(sub, a.k0)
        runs = {n: run_pool(f, table, a.budget, splits, seeds) for n, f in frames.items()}
        tab = {n: run_pool_table(f, table, a.budget, splits, seeds, a.k0) for n, f in frames.items()}
        summ = {n: pool_summary(runs[n], splits) for n in evals}
        tsum = {n: table_summary(runs[n], tab[n], splits) for n in evals}
        subs.append({"subset": i, "row_positions": [int(x) for x in idx], "distinct_pids": int(sub["pid"].nunique()),
                     "strata_in_table": int(len(table)), "runs": runs, "summary": summ, "tab": tab, "tab_summary": tsum})
        print(f"subset {i:2d}: {len(table)} strata, frozen minus fixed " + ", ".join(f"{n} {summ[n]['mean_pp']:+.2f}" for n in evals)
              + ", table_pooled minus frozen " + ", ".join(f"{n} {tsum[n]['mean_vs_frozen_pp']:+.2f}" for n in evals), flush=True)

    fixed_same = all(R[s]["fixed_hits"] == full[n][s]["fixed_hits"]
                     for x in subs for n, R in x["runs"].items() for s in splits)
    sg = spend_gate({**{f"{n}/gate": full[n] for n in evals},
                     **{f"{n}/subset{x['subset']}": x["runs"][n] for x in subs for n in evals}})
    tg = table_spend(subs, list(evals))
    l6a_full = {n: l6a_split_gate(full_sum[n]) for n in evals}
    l6a_subs = {n: l6a_subset_gate(n, subs) for n in evals}
    zg = {n: zero_refit_gate(n, subs, splits) for n in evals}
    subset_ok = all(x["distinct_pids"] == a.subset_size and len(set(x["row_positions"])) == a.subset_size for x in subs)
    overlap_ok = all(v == 0 for v in overlap.values())
    gates_ok = (sg["pass"] and tg["pass"] and subset_ok and overlap_ok and fixed_same
                and all(g["pass"] for g in l6a_full.values()) and all(g["pass"] for g in l6a_subs.values())
                and all(g["pass"] for g in zg.values()))
    out = {"experiment": "L11a", "label": LABEL, "group": GROUP, "stratum": STRATUM, "eval_pid_max": a.eval_pid_max,
           "test_prompts": test_prompts, "test_proxy_error_rate": test_err,
           "pooled_score": "(e + k0 * table_post) / (v + k0) per stratum; e, v = verified eval errors and count in the stratum; "
                           "table_post = frozen 200-label posterior (1 - agreement when absent); recomputed every R labels",
           "k0": a.k0, "budget_frac": a.budget, "fit_feats": a.fit_feats, "fit_prompts": int(len(fit_df)), "subsets": a.subsets,
           "subset_size": a.subset_size, "subset_draw": "numpy RandomState(100 + i).choice(n_p0_rows, subset_size, replace=False)",
           "frozen_refit": "none (each subset table is frozen after its fit)",
           "table_policy": "table_pooled: G01 greedy_online, K = 1, initial score = frozen 200-label table posterior, key "
                           "(-score, agreement, -n_distinct, seeded draw), scores recomputed every R = max(budget // 8, 40) verified "
                           "labels as (e + k0 * table posterior) / (v + k0) per stratum; no logistic model, no p0 refit",
           "seeds": seeds, "splits": splits, "sd_ddof": 1, "pass_threshold_pp": PASS_PP, "tie_seeds": a.tie_seeds,
           "eval_question_overlap_with_p0": overlap,
           "gates": {"test_rows_reproduction_L6a": l6a_full, "frozen_subset_reproduction_L6a": l6a_subs,
                     "zero_refit_reproduction": zg, "spend_gate": sg, "table_spend_gate": tg,
                     "fixed_identical_across_tables": bool(fixed_same),
                     "subset_distinct_prompts": bool(subset_ok), "eval_question_overlap_zero": bool(overlap_ok),
                     "pass": bool(gates_ok)}}
    for n, v in l6a_full.items():
        print(f"L6a test-row reproduction {n}: 3,000-prompt frozen gaps and mean fixed rate, max abs diff {v['max_abs_diff']:.2e} "
              f"(tol {L6A_TOL_PP:g}), pass {v['pass']}", flush=True)
    for n, v in l6a_subs.items():
        print(f"L6a per-subset reproduction {n}: max abs diff {v['max_abs_diff_pp']:.2e} pp (tol {L6A_TOL_PP:g}), "
              f"row positions match {v['row_positions_match']}, pass {v['pass']}", flush=True)
    for n, v in zg.items():
        print(f"zero-update gate {n}: hits equal frozen {v['hits_equal_frozen']}, max abs diff vs L6a {v['max_abs_diff_pp']:.2e} pp, "
              f"pass {v['pass']}", flush=True)
    print(f"spend gate: {sg['checks']} checks, failures {sg['failures']}; table spend: {tg['checks']} checks, failures "
          f"{tg['failures']}; subsets distinct {subset_ok}; fixed identical {fixed_same}; eval overlap with p0 {overlap}; "
          f"gates {'PASS' if gates_ok else 'FAIL'}", flush=True)
    if not gates_ok:
        out["verdict"] = "GATE_FAILED"; write_json(a.out, out)
        print("GATE FAILED: no L11a verdict written", flush=True); sys.exit(1)

    pools = {}
    for n in evals:
        ts = [x["tab_summary"][n] for x in subs]; sm = [x["summary"][n] for x in subs]
        means = [t["mean_vs_frozen_pp"] for t in ts]; m = np.asarray(means, float)
        pools[n] = {"feats": evals[n], "mean_pp": float(m.mean()), "sd_pp": float(m.std(ddof=1)), "n_subsets": len(m),
                    "n_subsets_above_pass": int((m > PASS_PP).sum()),
                    "n_subsets_at_or_below_pass": int((m <= PASS_PP).sum()),
                    "min_subset_mean_pp": float(m.min()), "max_subset_mean_pp": float(m.max()),
                    "subset_mean_pp": means, "gap_pp_per_subset": [t["gap_vs_frozen_pp_per_split"] for t in ts],
                    "n_split_gaps": int(sum(len(t["gap_vs_frozen_pp_per_split"]) for t in ts)),
                    "n_split_gaps_above_pass": int(sum((np.asarray(t["gap_vs_frozen_pp_per_split"]) > PASS_PP).sum() for t in ts)),
                    "max_tie_seed_spread_hits": int(max(max(t["tie_seed_spread_hits_per_split"]) for t in ts)),
                    "pooled_pct_mean": float(np.mean([np.mean(t["pooled_pct_per_split"]) for t in ts])),
                    "fixed_pct_mean": float(np.mean([np.mean(x["fixed_pct_per_split"]) for x in sm])),
                    "frozen_pct_mean": float(np.mean([np.mean(x["frozen_pct_per_split"]) for x in sm])),
                    "frozen_vs_fixed_mean_pp": float(np.mean([x["mean_pp"] for x in sm])),
                    "pooled_vs_fixed_mean_pp": float(np.mean([t["mean_vs_fixed_pp"] for t in ts])),
                    "full_table_3000_mean_pp": full_sum[n]["mean_pp"],
                    "verdict": "PASS" if m.mean() > PASS_PP else "KILL"}
    out["pools"] = pools
    out["verdicts"] = {n: pools[n]["verdict"] for n in evals}
    out["verdict"] = "PASS" if all(v == "PASS" for v in out["verdicts"].values()) else "KILL"
    out["subset_detail"] = [dict({"subset": x["subset"], "row_positions": x["row_positions"], "distinct_pids": x["distinct_pids"],
                                  "strata_in_table": x["strata_in_table"]},
                                 **{f"{n}_table_pooled_minus_frozen_pp": x["tab_summary"][n]["mean_vs_frozen_pp"] for n in evals},
                                 **{f"{n}_table_pooled_minus_frozen_pp_per_split": x["tab_summary"][n]["gap_vs_frozen_pp_per_split"]
                                    for n in evals},
                                 **{f"{n}_frozen_minus_fixed_pp": x["summary"][n]["mean_pp"] for n in evals},
                                 **{f"{n}_gap_pp_per_split": x["summary"][n]["gap_pp_per_split"] for n in evals},
                                 **{f"{n}_table_pooled_minus_fixed_pp": x["tab_summary"][n]["mean_vs_fixed_pp"] for n in evals})
                           for x in subs]
    for n in pools:
        p = pools[n]
        print(f"{n}: table_pooled minus frozen 200-label, mean over {p['n_subsets']} subsets {p['mean_pp']:+.2f}, sd over subset "
              f"means {p['sd_pp']:.2f}, subsets above +1.0 {p['n_subsets_above_pass']} of {p['n_subsets']}, split gaps above +1.0 "
              f"{p['n_split_gaps_above_pass']} of {p['n_split_gaps']}, min {p['min_subset_mean_pp']:+.2f} max "
              f"{p['max_subset_mean_pp']:+.2f}, frozen minus fixed {p['frozen_vs_fixed_mean_pp']:+.2f}, "
              f"table_pooled minus fixed {p['pooled_vs_fixed_mean_pp']:+.2f}, verdict {p['verdict']}", flush=True)
    print(f"L11a verdict: {out['verdict']} (per pool {out['verdicts']})", flush=True)
    write_json(a.out, out); write_table(a.out, out)
    print(f"wrote {os.path.join(a.out, 'result.json')} and table.md", flush=True)


if __name__ == "__main__":
    main()
