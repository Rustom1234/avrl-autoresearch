"""L5a: frozen-table test of config B at 10%. CPU only. The table is fitted once and never refit.

Structure follows avrl/exp/l4b_replay.py. Changes:
  Config B's stratum table (stratum agreement x n_distinct x any_truncated, k0 50, prior mean 1 - agreement) is fitted
  once on the L1B pool (runs/p0/feats.parquet, all 3,000 prompts) as the posterior error rate per stratum,
  (errors + k0 * prior) / (count + k0), and then frozen. Each eval pool ranks its sim docs by that frozen score with the
  tie-break order of the refit (score descending, agreement ascending, n_distinct descending, per-seed random draw) and
  takes the top int(budget * n_sim) docs. A stratum missing from the p0 table scores its prior, as an unlabelled stratum
  does in recal.
  Comparator: plain fixed threshold (L1B.FIXED) on the same sim split and tie-break seed.
  Gates, before any verdict:
    1. Plain fixed at 10% reproduces runs/L2b/result.json (config B) and runs/L4b/result.json (config B40) per split to
       0.01 pp, with the same calls per split.
    2. Spend gate: every policy takes exactly int(budget * n_sim) distinct docs on every split and seed (0 failures).
  Verdict: per eval pool, PASS if the mean gap over split seeds 0 to 7 is above +2.0 pp, else KILL. Overall PASS needs
  both eval pools. The p0 in-sample gap is reference only, because the table was fitted on p0.
  --floor is accepted for command-shape compatibility. The table is never refit, so no refit floor applies.

Usage: python -I l5a_frozen_table.py --fit_feats FEATS --eval_feats FEATS [FEATS ...] --budget 0.10 --k0 50 --floor 40
           --splits 0-7 --tie_seeds 8 --out OUTDIR
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
# Pool name -> (result.json holding plain fixed rates at 10%, config key of config B in that file)
REPRO_REF = {"L2b": (f"{RUNS}/L2b/result.json", "B"), "L4b": (f"{RUNS}/L4b/result.json", "B40")}
# Refitted B at refit floor 40, shown beside the frozen result for reference
REFIT40_REF = {"L2b": (f"{RUNS}/L3a/fresh/result.json", "B"), "L4b": (f"{RUNS}/L4b/result.json", "B40"),
               "p0": (f"{RUNS}/L3a/l1b_pool/result.json", "B")}


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
            "verdict": "PASS" if st["mean_pp"] > L1B.PASS_PP else "KILL",
            "splits_info": {str(s): {"n_sim": R[s]["n_sim"], "n_buckets": R[s]["n_buckets"], "n_groups": R[s]["n_groups"],
                                     "hits_total": R[s]["hits_total"], "budget": R[s]["budget"]} for s in splits}}


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


def spend_gate(pools):
    bad = [dict(pool=name, split=s, **b) for name, R in pools.items() for s in R for b in R[s]["spend_bad"]]
    checks = sum(2 * len(R[next(iter(R))]["fixed_hits"]) * len(R) for R in pools.values())
    return {"checks": checks, "failures": len(bad), "examples": bad[:5], "pass": not bad}


def refit_ref(name):
    path, key = REFIT40_REF[name]
    a = json.load(open(path))["configs"][key]["all_splits"]
    return {"source": path, "config": key, "mean_pp": float(a["mean_pp"]), "sd_pp": float(a["sd_pp"]),
            "per_split_pp": [float(x) for x in a["per_split_pp"]]}


def write_json(out_dir, out):
    with open(os.path.join(out_dir, "result.json"), "w") as fh: json.dump(out, fh, indent=1, default=_jsonable)


def write_table(out_dir, r):
    P = r["pools"]; names = list(P)
    L = ["# L5a: frozen-table config B at 10%, fitted once on p0", "",
         f"Rule: frozen-table gap is frozen B minus plain fixed at 10%, mean over split seeds 0 to 7 (paired by tie-break seed, "
         f"{r['tie_seeds']} per split). PASS if the mean is above +{L1B.PASS_PP:.1f} pp on both eval pools, else KILL. "
         "The p0 in-sample gap is reference only.", "",
         f"Overall verdict: **{r['verdict']}**", "",
         "## Gates", "", "| gate | result |", "|---|---|"]
    for name, g in r["gates"]["fixed_reproduction"].items():
        L.append(f"| plain fixed 10% reproduces runs/{name}/result.json | {'PASS' if g['pass'] else 'FAIL'} "
                 f"(max abs diff {g['max_abs_diff_pp']:.6f} pp, tol {REPRO_TOL_PP}, calls match {g['calls_match']}) |")
    sg = r["gates"]["spend_gate"]
    L.append(f"| spend gate | {'PASS' if sg['pass'] else 'FAIL'} ({sg['checks']} checks, failures {sg['failures']}) |")
    L += ["", "## Verdict and reference", "",
          "| pool | role | sim docs per split | frozen gap mean (pp) | sd (pp) | splits above 0 | splits above +2.0 | "
          "fixed 10% mean (%) | frozen 10% mean (%) | docs in p0 table (%) | refitted B floor 40 mean (pp) | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        p = P[n]; role = "reference only (fitted on p0)" if n == "p0" else "verdict"
        rf = r["refit_floor40"][n]
        nsim = p["n_sim_per_split"]; v = "reference" if n == "p0" else p["verdict"]
        L.append(f"| {n} | {role} | {min(nsim)} to {max(nsim)} | {p['mean_pp']:+.2f} | {p['sd_pp']:.2f} | {p['n_positive']} of 8 | "
                 f"{p['n_above_pass_pp']} of 8 | {np.mean(p['fixed_pct_per_split']):.2f} | {np.mean(p['frozen_pct_per_split']):.2f} | "
                 f"{100 * np.mean(p['in_table_frac_per_split']):.1f} | {rf['mean_pp']:+.2f} | **{v}** |")
    L += ["", "## Gap per split (pp), frozen minus fixed at 10%", "",
          "| split | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for i, s in enumerate(r["splits"]):
        L.append(f"| {s} | " + " | ".join(f"{P[n]['gap_pp_per_split'][i]:+.2f}" for n in names) + " |")
    L += ["", "## Refitted B at floor 40, reference (per split pp in the source files)", "",
          "| pool | source | mean (pp) | sd (pp) | per split (pp) |", "|---|---|---|---|---|"]
    for n in names:
        rf = r["refit_floor40"][n]
        L.append(f"| {n} | {rf['source'].replace('/homes/rdubash/avrl_live/ROUTED/', '')} | {rf['mean_pp']:+.2f} | {rf['sd_pp']:.2f} | "
                 + " / ".join(f"{x:+.2f}" for x in rf["per_split_pp"]) + " |")
    L += ["", "## Frozen table (fitted on p0, k0 " + f"{r['k0']:g}" + ")", "",
          "| agreement | n_distinct | any_truncated | p0 errors | p0 count | prior (1 - agreement) | frozen posterior |",
          "|---|---|---|---|---|---|---|"]
    for row in sorted(r["table"], key=lambda x: -x["post"]):
        L.append(f"| {row['agreement']:.3f} | {row['n_distinct']} | {row['any_truncated']} | {row['errors']} | {row['count']} | "
                 f"{row['prior']:.3f} | {row['post']:.4f} |")
    L += ["", "## Notes", "",
          "- runs/L2b/result.json holds config B at refit floor 10 only. Its floor 40 reference is runs/L3a/fresh/result.json, "
          "which replays the same L2b feats file.",
          "- Strata absent from the p0 table score their prior. The in-table share column shows how much of each eval pool "
          "the frozen table actually labels.",
          "- The table is fitted on p0 and never refit. The floor argument is accepted for command-shape compatibility only.", ""]
    with open(os.path.join(out_dir, "table.md"), "w") as fh: fh.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_feats", required=True)
    ap.add_argument("--eval_feats", nargs="+", required=True)
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--k0", type=float, default=50.0)
    ap.add_argument("--floor", type=int, default=40, help="accepted for command-shape compatibility; the table is never refit")
    ap.add_argument("--splits", default="0-7")
    ap.add_argument("--tie_seeds", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    splits, seeds = parse_splits(a.splits), list(range(a.tie_seeds))

    fit_df = pd.read_parquet(a.fit_feats)
    table = fit_table(fit_df, a.k0)
    print(f"frozen table: {len(table)} strata from {len(fit_df)} prompts of {a.fit_feats}, k0 {a.k0:g}, no refit", flush=True)

    evals = {}
    for f in a.eval_feats:
        name = pool_name(f)
        if name not in REPRO_REF: sys.exit(f"no reproduction reference for pool {name}")
        evals[name] = (f, run_pool(f, table, a.budget, splits, seeds))
    p0 = run_pool(a.fit_feats, table, a.budget, splits, seeds)

    sg = spend_gate({**{n: R for n, (_, R) in evals.items()}, "p0": p0})
    repro = {n: repro_gate(n, R, splits) for n, (_, R) in evals.items()}
    gates_ok = sg["pass"] and all(v["pass"] for v in repro.values())
    out = {"experiment": "L5a", "label": LABEL, "group": GROUP, "stratum": STRATUM, "k0": a.k0, "budget_frac": a.budget,
           "fit_feats": a.fit_feats, "fit_prompts": int(len(fit_df)), "fit_errors": int(fit_df[LABEL].sum()),
           "refit": "none (table frozen after the p0 fit)", "refit_floor_arg_unused": a.floor,
           "seeds": seeds, "splits": splits, "sd_ddof": 1, "pass_threshold_pp": L1B.PASS_PP, "tie_seeds": a.tie_seeds,
           "table": [{k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()} for row in table.to_dict("records")],
           "gates": {"fixed_reproduction": repro, "spend_gate": sg, "pass": gates_ok}}
    for n, v in repro.items():
        print(f"reproduction {n}: max abs diff {v['max_abs_diff_pp']:.6f} pp (tol {REPRO_TOL_PP}), calls match {v['calls_match']}, "
              f"pass {v['pass']}", flush=True)
    print(f"spend gate: {sg['checks']} checks, failures {sg['failures']}; gates {'PASS' if gates_ok else 'FAIL'}", flush=True)
    if not gates_ok:
        out["verdict"] = "GATE_FAILED"; write_json(a.out, out)
        print("GATE FAILED: no L5a verdict written", flush=True); sys.exit(1)

    out["pools"] = {n: dict(pool_summary(R, splits), feats=f) for n, (f, R) in evals.items()}
    out["pools"]["p0"] = dict(pool_summary(p0, splits), feats=a.fit_feats)
    out["refit_floor40"] = {n: refit_ref(n) for n in out["pools"]}
    out["verdicts"] = {n: out["pools"][n]["verdict"] for n in evals}
    out["verdict"] = "PASS" if all(v == "PASS" for v in out["verdicts"].values()) else "KILL"
    out["tie_seeds"] = a.tie_seeds
    for n in out["pools"]:
        p = out["pools"][n]
        print(f"{n}{' (reference only)' if n == 'p0' else ''}: frozen minus fixed at 10%, gap per split "
              f"{[round(g, 2) for g in p['gap_pp_per_split']]} mean {p['mean_pp']:+.2f} sd {p['sd_pp']:.2f} "
              f"above +2.0 {p['n_above_pass_pp']} of 8, fixed 10% mean {np.mean(p['fixed_pct_per_split']):.2f} "
              f"frozen 10% mean {np.mean(p['frozen_pct_per_split']):.2f}, refitted B floor 40 "
              f"{out['refit_floor40'][n]['mean_pp']:+.2f}", flush=True)
    print(f"L5a verdict: {out['verdict']} (per pool {out['verdicts']})", flush=True)
    write_json(a.out, out); write_table(a.out, out)
    print(f"wrote {os.path.join(a.out, 'result.json')} and table.md", flush=True)


if __name__ == "__main__":
    main()
