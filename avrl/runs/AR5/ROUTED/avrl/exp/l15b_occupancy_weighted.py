"""L15b: occupancy-weighted remaining-row cut, the test of whether row mix explains the L13a frozen-table gap. CPU only.
No GPU job, no new rollouts, no edits to existing scripts.

Copied from avrl/exp/l13a_shrunk_refit.py. Kept: the 200-label frozen table (fit_table on the 200 p0 labels of subset i, numpy
RandomState(100 + i), k0 = 50, prior 1 - agreement, scored per pool row by attach_score) and the plain fixed threshold (fixed_pick,
frozen_pick, job and the spend checks exactly as in L13a). Dropped: table_zero, table_shrunk and the offset-ridge refit, the five
matched fold tables, the eval fold draw, and the L5a, L6b and L10b gates. None of them feeds the frozen or the fixed arm.

Stratum key: agreement x n_distinct x any_truncated (STRATUM). Occupancy from runs/L4b/feats.parquet: test rows pid <= 1318
(n_test = 1319) and remaining rows pid > 1318 (n_rem = 1473). A remaining doc in stratum s gets w_s = (test_s / n_test) /
(remaining_s / n_rem). A stratum with no remaining docs is dropped and listed. A stratum with no test docs gets w_s = 0, so its
remaining docs are never drawn. The constant from normalising the weights does not change the draw, since p is normalised.

Weighted cut (--weighted, cut seed c): numpy RandomState(c).choice over the remaining rows, replace=False, p proportional to w_s,
size n_test = 1319. The drawn rows are sorted into file order before scoring, as in L13a's cut frame.
Unregistered sensitivity (--min_stratum 5): strata with fewer than 5 test or fewer than 5 remaining docs are dropped from both
sides, and the draw size is the number of test rows in the kept strata.

Modes:
  unweighted reference cut (--eval_pid_min 1319, or --eval_pid_max 1318): must reproduce runs/L13a/L4b_pid_gt1318 (+8.579 pp) or
    runs/L13a/L4b_pid_le1318 (+1.222 pp), per subset and per split, before any weighted number is reported.
  weighted cut (--weighted --eval_pid_min 1319 --cut_seeds 100-147): one result per cut seed in <out>/cut_<seed>/result.json.
  combine (--combine --out DIR --ref_dir RUN --cut_seeds 100-147): reads the cut results, writes DIR/result.json. It reports no
    metric unless every gate passes.

Per run: 20 subsets, split seeds 0 to 7, 8 tie-break seeds, budget int(0.10 x n_sim) as in L13a.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import l1b_tuned_threshold as L1B

RUNS = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
LABEL, GROUP = "proxy_error", "cell100"
STRATUM = ["agreement", "n_distinct", "any_truncated"]
PID_TEST_MAX = 1318  # registered L4b test rows are pid 0 to 1318; remaining rows are pid 1319 and up
SUBSET_SEED0 = 100
TOL_PP = 1e-6
ABOVE_PP = 2.0
PASS_AT_OR_BELOW_PP, KILL_AT_OR_ABOVE_PP = 3.2, 5.0
L13A_REF = {"rem": f"{RUNS}/L13a/L4b_pid_gt1318/result.json", "test": f"{RUNS}/L13a/L4b_pid_le1318/result.json"}


def parse_seeds(s):
    if "-" in s:
        lo, hi = s.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def fit_table(fit_df, k0):
    """Frozen table: posterior error rate per stratum from every labelled prompt of the fit subset. Never refit."""
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


def draw_subset(fit_df, i, size):
    """Row positions of label subset i: size distinct p0 prompts, numpy RandomState(100 + i), without replacement."""
    return np.random.RandomState(SUBSET_SEED0 + i).choice(len(fit_df), size=size, replace=False)


def job(df, split, frac, seeds):
    """One split of one subset: the L13a job for the fixed and frozen arms."""
    sim, _ = L1B.build_sim(df, GROUP, split)
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
    return {"split": split, "n_sim": S.n, "budget": budget, "fixed_hits": fixed, "frozen_hits": frozen, "spend_bad": spend_bad}


def evaluate(pool, fit_df, a, splits, seeds, keep_rows):
    """For subsets i = 0 .. a.subsets - 1: the 200-label table, the frozen score of every pool row, then fixed and frozen picks
    on every split. Returns per-subset summaries, the count of pool questions that occur in p0, and whether the fixed picks agree
    across subsets (they must, since the fixed threshold does not read the table)."""
    overlap = int(pool["question"].isin(set(fit_df["question"])).sum())
    subs, fixed_ref, fixed_same = [], None, True
    for i in range(a.subsets):
        idx = draw_subset(fit_df, i, a.subset_size)
        sub = fit_df.iloc[idx]
        table = fit_table(sub, a.k0)
        scored = attach_score(pool, table)
        recs = [job(scored, s, a.budget, seeds) for s in splits]
        gaps, fixed_pct, frozen_pct = [], [], []
        for r in recs:
            b = r["budget"]
            f = 100.0 * np.asarray(r["fixed_hits"], float) / b
            z = 100.0 * np.asarray(r["frozen_hits"], float) / b
            gaps.append(float((z - f).mean())); fixed_pct.append(float(f.mean())); frozen_pct.append(float(z.mean()))
        hits = [r["fixed_hits"] for r in recs]
        if fixed_ref is None: fixed_ref = hits
        fixed_same = fixed_same and hits == fixed_ref
        subs.append({"subset": i, "distinct_pids": int(sub["pid"].nunique()), "strata_in_table": int(len(table)),
                     "gap_pp_per_split": gaps, "mean_pp": float(np.mean(gaps)), "fixed_pct_mean": float(np.mean(fixed_pct)),
                     "frozen_pct_mean": float(np.mean(frozen_pct)), "n_sim_per_split": [r["n_sim"] for r in recs],
                     "budget_per_split": [r["budget"] for r in recs], "spend_bad": [b for r in recs for b in r["spend_bad"]],
                     "row_unique": len(set(int(x) for x in idx)),
                     "row_positions": [int(x) for x in idx] if keep_rows else None})
        print(f"subset {i:2d}: {len(table)} strata, frozen minus fixed {subs[-1]['mean_pp']:+.3f} pp", flush=True)
    return subs, overlap, bool(fixed_same)


def gate_block(subs, overlap, fixed_same, side_ok, size_ok, a):
    spend_failures = sum(len(x["spend_bad"]) for x in subs)
    subset_ok = all(x["distinct_pids"] == a.subset_size and x["row_unique"] == a.subset_size for x in subs)
    g = {"spend_failures": spend_failures, "subset_distinct_prompts": bool(subset_ok), "fixed_identical_across_subsets": fixed_same,
         "eval_question_overlap_with_p0": overlap, "pool_side_ok": bool(side_ok), "pool_size_ok": bool(size_ok)}
    g["pass"] = bool(spend_failures == 0 and subset_ok and fixed_same and overlap == 0 and side_ok and size_ok)
    return g


def reference_gate(key, subs, cut_mean):
    """Unweighted reference cut against runs/L13a: the mean, the row positions, and every per-subset per-split gap."""
    ref = json.load(open(L13A_REF[key]))
    det = ref["subset_detail"]
    ref_mean = float(ref["pools"]["L4b"]["frozen_vs_fixed_mean_pp"])
    pos_ok = len(det) == len(subs) and all(det[x["subset"]]["row_positions"] == x["row_positions"]
                                           and det[x["subset"]]["distinct_pids"] == x["distinct_pids"] for x in subs)
    worst = max(abs(m - float(t)) for x in subs for m, t in zip(x["gap_pp_per_split"], det[x["subset"]]["L4b_gap_pp_per_split"]))
    d_mean = abs(cut_mean - ref_mean)
    return {"ref": L13A_REF[key], "ref_mean_pp": ref_mean, "mine_mean_pp": cut_mean, "abs_diff_mean_pp": float(d_mean),
            "row_positions_match": bool(pos_ok), "max_abs_diff_split_gap_pp": float(worst), "tol_pp": TOL_PP,
            "pass": bool(pos_ok and worst <= TOL_PP and d_mean <= TOL_PP)}


def cut_frame(ev, pid_min, pid_max):
    keep = np.ones(len(ev), bool)
    if pid_min is not None: keep &= ev["pid"].to_numpy() >= pid_min
    if pid_max is not None: keep &= ev["pid"].to_numpy() <= pid_max
    return ev[keep].reset_index(drop=True)


def occupancy(ev):
    """Stratum codes over the full eval frame, and the test and remaining counts of every stratum."""
    for c in STRATUM: assert not ev[c].isna().any(), c
    keys = list(ev.groupby(STRATUM, sort=True).size().index)
    code = ev.groupby(STRATUM, sort=True).ngroup().to_numpy()
    assert code.max() + 1 == len(keys)
    is_test = ev["pid"].to_numpy() <= PID_TEST_MAX
    ct = np.bincount(code[is_test], minlength=len(keys)); cr = np.bincount(code[~is_test], minlength=len(keys))
    return keys, code, is_test, ct, cr


def weighted_draw(ev, seed, min_stratum):
    """Remaining-row pool for one cut seed, drawn with probability proportional to w_s. Returns the drawn row positions of ev in
    file order, and the stratum table with the draw counts."""
    keys, code, is_test, ct, cr = occupancy(ev)
    n_test, n_rem = int(is_test.sum()), int((~is_test).sum())
    keep = cr > 0
    if min_stratum: keep = keep & (ct >= min_stratum) & (cr >= min_stratum)
    w = np.where(keep & (ct > 0), (ct / n_test) / (np.where(cr > 0, cr, 1) / n_rem), 0.0)
    size = int(ct[keep].sum()) if min_stratum else n_test
    elig = np.flatnonzero((~is_test) & keep[code])
    p = w[code[elig]]
    assert (p > 0).sum() >= size, f"{int((p > 0).sum())} weighted remaining rows for a draw of {size}"
    pick = np.random.RandomState(seed).choice(len(elig), size=size, replace=False, p=p / p.sum())
    chosen = np.sort(elig[pick])
    drawn = np.bincount(code[chosen], minlength=len(keys))
    tgt = np.where(keep, ct, 0).astype(float); tgt = tgt / tgt.sum()
    dev = np.abs(drawn / drawn.sum() - tgt)
    strata = [{"stratum": [float(k[0]), int(k[1]), int(k[2])], "test": int(ct[j]), "remaining": int(cr[j]), "kept": bool(keep[j]),
               "w": float(w[j]), "drawn": int(drawn[j])} for j, k in enumerate(keys)]
    return chosen, {"size": size, "n_test": n_test, "n_rem": n_rem, "eligible": int(len(elig)), "tv": float(0.5 * dev.sum()),
                    "max_share_dev": float(dev.max()), "strata": strata}


def write_json(out_dir, obj):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def combine(a):
    seeds = parse_seeds(a.cut_seeds)
    cuts = [json.load(open(os.path.join(a.out, f"cut_{c}", "result.json"))) for c in seeds]
    refs = {k: json.load(open(os.path.join(a.ref_dir, f"ref_{k}", "result.json"))) for k in ("rem", "test")} if a.ref_dir else {}
    gates_ok = bool(refs) and all(c["pass"] for c in cuts) and all(r["pass"] for r in refs.values())
    out = {"experiment": "L15b", "registered": a.min_stratum == 0,
           "scope": "occupancy-weighted remaining rows (pid > 1318), frozen minus plain fixed at 10%",
           "cut_seeds": seeds, "n_cuts": len(cuts), "min_stratum": a.min_stratum,
           "thresholds_pp": {"pass_at_or_below": PASS_AT_OR_BELOW_PP, "kill_at_or_above": KILL_AT_OR_ABOVE_PP},
           "references_pp": {"unweighted_remaining_L13a": 8.579411680395452, "registered_test_rows_L13a": 1.2222872985463904},
           "gates": {"references": {k: r["reference_gate"] for k, r in refs.items()},
                     "reference_runs_pass": {k: r["pass"] for k, r in refs.items()},
                     "cuts_all_pass": all(c["pass"] for c in cuts), "gates_ok": gates_ok}}
    if not gates_ok:
        out["verdict"] = "GATE_FAILED: no metric reported"
        write_json(a.out, out); print("GATE FAILED, no metric written", flush=True); return
    means = np.array([c["cut_mean_pp"] for c in cuts], float)
    mean, sd = float(means.mean()), float(means.std(ddof=1))
    if mean <= PASS_AT_OR_BELOW_PP: verdict = "PASS: row mix explains the gap (mean at or below +3.2 pp)"
    elif mean >= KILL_AT_OR_ABOVE_PP: verdict = "KILL: row mix does not explain the gap (mean at or above +5.0 pp)"
    else: verdict = "INCONCLUSIVE: mean between +3.2 and +5.0 pp"
    st = cuts[0]["strata"]
    mean_share = np.mean([[s["drawn"] / c["pool_rows"] for s in c["strata"]] for c in cuts], axis=0)
    tgt = np.array([s["test"] if s["kept"] else 0 for s in st], float); tgt = tgt / tgt.sum()
    size, n_rem = cuts[0]["draw"]["size"], cuts[0]["draw"]["n_rem"]
    cr = np.array([s["remaining"] for s in st], float)
    # mix diagnostics: TV from the target shares of (a) the unweighted remaining rows, (b) the realised weighted draw, and
    # (c) the feasibility floor, the smallest TV any draw of the size from the remaining rows can reach (a stratum cannot supply
    # more docs than it has, so the floor is sum_s max(size x target_s - remaining_s, 0) / size)
    tv_floor = float(np.maximum(size * tgt - cr, 0).sum() / size)
    out.update({
        "mix_tv": {"unweighted_remaining_vs_target": float(0.5 * np.abs(cr / n_rem - tgt).sum()),
                   "weighted_realised_mean": float(np.mean([c["draw"]["tv"] for c in cuts])),
                   "feasibility_floor": tv_floor},
        "metric": "frozen minus plain fixed at 10%, pp of hit rate: mean over cuts of the mean over 20 subsets x split seeds 0-7",
        "mean_pp": mean, "min_pp": float(means.min()), "max_pp": float(means.max()), "sd_pp": sd,
        "n_cuts_above_2pp": int((means > ABOVE_PP).sum()), "cut_means_pp": [float(x) for x in means],
        "verdict": verdict,
        "unweighted_remaining_mine_pp": refs["rem"]["cut_mean_pp"], "unweighted_test_mine_pp": refs["test"]["cut_mean_pp"],
        "pool": {"rows_per_cut": sorted({c["pool_rows"] for c in cuts}), "tv_mean": float(np.mean([c["draw"]["tv"] for c in cuts])),
                 "max_share_dev_worst": float(max(c["draw"]["max_share_dev"] for c in cuts))},
        "mix": [{"stratum": s["stratum"], "kept": s["kept"], "test": s["test"], "remaining": s["remaining"], "w": s["w"],
                 "target_share": float(t), "mean_drawn_share": float(m)} for s, t, m in zip(st, tgt, mean_share)],
        "per_cut": [{"cut_seed": c["cut_seed"], "mean_pp": c["cut_mean_pp"], "fixed_pct_mean": c["cut_fixed_pct_mean"],
                      "frozen_pct_mean": c["cut_frozen_pct_mean"], "n_subsets_above_2pp": c["n_subsets_above_2pp"],
                      "tv": c["draw"]["tv"]} for c in cuts]})
    write_json(a.out, out)
    print(f"L15b ({'registered' if out['registered'] else 'UNREGISTERED sensitivity'}): mean {mean:+.3f}, sd over cut means {sd:.3f}, "
          f"min {means.min():+.3f}, max {means.max():+.3f}, cuts above +2.0: {out['n_cuts_above_2pp']} of {len(means)}; {verdict}",
          flush=True)
    print(f"wrote {os.path.join(a.out, 'result.json')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_feats", required=True)
    ap.add_argument("--eval_feats", required=True)
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--k0", type=float, default=50.0)
    ap.add_argument("--subsets", type=int, default=20)
    ap.add_argument("--subset_size", type=int, default=200)
    ap.add_argument("--splits", default="0-7")
    ap.add_argument("--tie_seeds", type=int, default=8)
    ap.add_argument("--eval_pid_min", type=int, default=None)
    ap.add_argument("--eval_pid_max", type=int, default=None)
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--cut_seeds", default=None)
    ap.add_argument("--min_stratum", type=int, default=0)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--ref_dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.combine:
        combine(a); return
    os.makedirs(a.out, exist_ok=True)
    splits, seeds = parse_seeds(a.splits), list(range(a.tie_seeds))
    fit_df = pd.read_parquet(a.fit_feats); ev = pd.read_parquet(a.eval_feats)

    if a.weighted:
        if not (a.eval_pid_min == PID_TEST_MAX + 1 and a.eval_pid_max is None and a.cut_seeds):
            sys.exit("--weighted needs --eval_pid_min 1319 and --cut_seeds")
        for c in parse_seeds(a.cut_seeds):
            chosen, draw = weighted_draw(ev, c, a.min_stratum)
            pool = ev.iloc[chosen].reset_index(drop=True)
            subs, overlap, fixed_same = evaluate(pool, fit_df, a, splits, seeds, keep_rows=False)
            cut_mean = float(np.mean([x["mean_pp"] for x in subs]))
            side_ok = bool((pool["pid"].to_numpy() > PID_TEST_MAX).all())
            size_ok = len(pool) == draw["size"] and pool["pid"].nunique() == len(pool)
            g = gate_block(subs, overlap, fixed_same, side_ok, size_ok, a)
            res = {"kind": "weighted_cut", "cut_seed": c, "min_stratum": a.min_stratum, "pool_rows": int(len(pool)),
                   "pool_pids": [int(x) for x in pool["pid"].to_numpy()], "draw": {k: v for k, v in draw.items() if k != "strata"},
                   "strata": draw["strata"], "cut_mean_pp": cut_mean,
                   "cut_fixed_pct_mean": float(np.mean([x["fixed_pct_mean"] for x in subs])),
                   "cut_frozen_pct_mean": float(np.mean([x["frozen_pct_mean"] for x in subs])),
                   "n_subsets_above_2pp": int(sum(x["mean_pp"] > ABOVE_PP for x in subs)), "gates": g, "pass": g["pass"],
                   "subsets": subs}
            write_json(os.path.join(a.out, f"cut_{c}"), res)
            print(f"cut {c}: pool {len(pool)} rows, tv {draw['tv']:.4f}, cut mean {cut_mean:+.3f} pp, gates "
                  f"{'PASS' if g['pass'] else 'FAIL'}", flush=True)
        return

    if (a.eval_pid_min, a.eval_pid_max) == (PID_TEST_MAX + 1, None): key, side = "rem", (lambda p: p > PID_TEST_MAX)
    elif (a.eval_pid_min, a.eval_pid_max) == (None, PID_TEST_MAX): key, side = "test", (lambda p: p <= PID_TEST_MAX)
    else: sys.exit("unweighted reference run needs --eval_pid_min 1319 or --eval_pid_max 1318")
    pool = cut_frame(ev, a.eval_pid_min, a.eval_pid_max)
    subs, overlap, fixed_same = evaluate(pool, fit_df, a, splits, seeds, keep_rows=True)
    cut_mean = float(np.mean([x["mean_pp"] for x in subs]))
    g = gate_block(subs, overlap, fixed_same, bool(side(pool["pid"].to_numpy()).all()), len(pool) > 0, a)
    ref = reference_gate(key, subs, cut_mean)
    res = {"kind": "unweighted_reference", "reference": key, "pool_rows": int(len(pool)), "cut_mean_pp": cut_mean,
           "cut_fixed_pct_mean": float(np.mean([x["fixed_pct_mean"] for x in subs])),
           "cut_frozen_pct_mean": float(np.mean([x["frozen_pct_mean"] for x in subs])), "gates": g, "reference_gate": ref,
           "pass": bool(g["pass"] and ref["pass"]), "subsets": subs}
    write_json(a.out, res)
    print(f"reference {key}: pool {len(pool)} rows, frozen minus fixed {cut_mean:+.6f} pp (L13a {ref['ref_mean_pp']:+.6f}), "
          f"row positions match {ref['row_positions_match']}, max split gap diff {ref['max_abs_diff_split_gap_pp']:.2e}, "
          f"pass {res['pass']}", flush=True)


if __name__ == "__main__":
    main()
