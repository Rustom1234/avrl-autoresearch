"""L1b: strongest label-free rule as the bar for claim 2 (CPU only, no GPU job).

Copied from avrl/exp/g01_label_free_baselines.py (G01) and cut down to what L1b needs. Two policies are kept as in G01:
  fixed_agreement_threshold: agreement ascending, then n_distinct descending, per-seed random tie-break. No labels.
  reference recal: the G01 recal_threshold (k0 = 2, stratum agreement x n_distinct, R = max(budget // 8, 40)).
The sweep replaces the constant 2 in table = (e + k0 * prior) / (v + k0) with k0 in {2, 10, 50}, refits every R verified
labels with R = max(budget // d, 10) for d in {16, 8, 4}, and uses the stratum key agreement, agreement x n_distinct or
agreement x n_distinct x any_truncated. Ordering is G01's: score, then agreement ascending, then n_distinct descending,
then the per-seed random draw. K = 1 only (the critic showed K = 4 gives identical per-seed hits on the E01 split).

"Label-free" means no trained scorer. recal and every swept rule read verified gold labels through per-stratum counts;
only fixed_agreement_threshold uses no labels. The cheap features are accepted for command-shape compatibility, but no
scorer is fit, so they are not read.

Protocol:
  split seed s: buckets (cell100, MIN = 3 docs) shuffled with default_rng(s); the first quarter of buckets is the fit
  split and is dropped. Tie-break seeds 0 to 7 inside each split.
  Gate, split seed 0: fixed matches runs/G01 within 0.05 pp at each budget; reference recal matches runs/G01 per seed.
  Tune, split seeds 0 to 3: per budget, the sweep config with the highest mean hit rate over splits and seeds.
  Held out, split seeds 4 to 7: per seed, selected config minus fixed, averaged per split; mean and sd over the 4 splits.

Usage (CPU only):
  /homes/rdubash/active-audit/.venv/bin/python -I l1b_tuned_threshold.py --feats FEATS --label proxy_error \
      --group cell100 --cheap agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen \
      --budgets 0.05,0.10,0.25,0.50 --tune-splits 0,1,2,3 --heldout-splits 4,5,6,7 --seeds 8 --ks 1 --out OUTDIR
"""
import os, sys, json, argparse, time, itertools
from multiprocessing import Pool
import numpy as np, pandas as pd

MIN = 3
REF_G01 = "/homes/rdubash/avrl_live/ROUTED/avrl/runs/G01/result.json"
FIXED = "fixed_agreement_threshold"
REF = "reference_recal"
K0S = (2, 10, 50)
RDIVS = (16, 8, 4)
STRATA = {"agreement": ["agreement"], "agreement_x_nd": ["agreement", "n_distinct"],
          "agreement_x_nd_x_trunc": ["agreement", "n_distinct", "any_truncated"]}
PASS_PP = 2.0
GATE_PP = 0.05


def build_specs():
    """(name, spec). spec None = plain fixed threshold, 'ref' = G01 reference recal, else (k0, R divisor, stratum)."""
    specs = [(FIXED, None), (REF, "ref")]
    specs += [(f"k0{k0}_Rdiv{d}_{st}", (k0, d, st)) for k0, d, st in itertools.product(K0S, RDIVS, STRATA)]
    return specs


SPECS = build_specs()
SWEEP = [name for name, spec in SPECS if isinstance(spec, tuple)]


def build_sim(df, G, split_seed):
    """Sim split as in G01, with the bucket shuffle seed set to the split seed."""
    cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
    groups = df[G].unique(); np.random.default_rng(split_seed).shuffle(groups)
    fit_g = set(groups[: len(groups) // 4])
    return df[~df[G].isin(fit_g)].reset_index(drop=True), len(groups)


class Sim:
    def __init__(self, sim, label):
        self.y = sim[label].values.astype(int); self.n = len(self.y)
        self.ag = sim["agreement"].values; self.nd = sim["n_distinct"].values
        self.codes, self.prior = {}, {}
        for name, keys in STRATA.items():
            c = sim.groupby(keys).ngroup().values
            ag_s = np.zeros(int(c.max()) + 1); ag_s[c] = self.ag
            self.codes[name] = c; self.prior[name] = 1 - ag_s  # the fixed threshold's error rate, the Beta prior mean

    def fixed(self, budget, seed):
        rng = np.random.default_rng(seed)
        order = np.lexsort((rng.random(self.n), -self.nd, self.ag))
        return int(self.y[order[:budget]].sum())

    def recal(self, budget, seed, k0, R, stratum):
        """G01 recal_threshold with prior strength k0, refit cadence R and stratum key as parameters (K = 1)."""
        codes, prior = self.codes[stratum], self.prior[stratum]; n_s = len(prior)
        rnd = np.random.default_rng(seed).random(self.n)
        seen = np.zeros(self.n, bool); table = prior.copy()  # no verified labels yet: table = prior
        hits = spent = 0; next_fit = R
        while spent < budget:
            order = np.lexsort((rnd, -self.nd, self.ag, -table[codes]))
            c = order[~seen[order]][:1]; seen[c] = True
            hits += int(self.y[c].sum()); spent += 1
            if spent >= next_fit:
                v = np.bincount(codes[seen], minlength=n_s)
                e = np.bincount(codes[seen], weights=self.y[seen], minlength=n_s)
                table = (e + k0 * prior) / (v + k0); next_fit += R
        return hits


def hits_for(S, spec, budget, seed):
    if spec is None: return S.fixed(budget, seed)
    if spec == "ref": return S.recal(budget, seed, 2, max(budget // 8, 40), "agreement_x_nd")
    k0, d, st = spec
    return S.recal(budget, seed, k0, max(budget // d, 10), st)


def run_split(job):
    """One fit/sim split: every policy, every budget, every tie-break seed. Runs in its own process."""
    feats, label, group, fracs, seeds, split = job
    df = pd.read_parquet(feats)
    sim, n_groups = build_sim(df, group, split)
    S = Sim(sim, label)
    out = {"split": split, "n_sim": S.n, "n_buckets": int(sim[group].nunique()), "n_groups": n_groups,
           "hits_total": int(S.y.sum()), "budgets": {f"{f:.2f}": int(f * S.n) for f in fracs}, "hits": {}}
    for name, spec in SPECS:
        out["hits"][name] = {f"{f:.2f}": [int(hits_for(S, spec, int(f * S.n), s)) for s in seeds] for f in fracs}
    return out


def rates(R, split, name, key):
    """Per-seed hit rate (%) of one policy on one split at one budget."""
    h = np.asarray(R[split]["hits"][name][key], float)
    return 100.0 * h / R[split]["budgets"][key]


def gate(R, ref_path, fracs):
    with open(ref_path) as fh: ref_json = json.load(fh)
    ref = ref_json["by_k"]["1"]["rows"]
    out = {"ref": ref_path, "split": 0, "n_sim_match": R[0]["n_sim"] == ref_json["n_sim"], "rows": {}, "pass": True}
    out["pass"] = bool(out["n_sim_match"])
    for f in fracs:
        key = f"{f:.2f}"; g = ref[key]
        mine_f = rates(R, 0, FIXED, key); mine_r = R[0]["hits"][REF][key]
        row = {"budget": R[0]["budgets"][key], "g01_budget": g[FIXED]["budget"],
               "fixed_mine_pct": float(mine_f.mean()), "fixed_g01_pct": g[FIXED]["hit_rate_pct"],
               "fixed_diff_pp": float(mine_f.mean() - g[FIXED]["hit_rate_pct"]),
               "fixed_per_seed_equal": bool(R[0]["hits"][FIXED][key] == g[FIXED]["hits_per_seed"]),
               "recal_per_seed_mine": mine_r, "recal_per_seed_g01": g["recal_threshold"]["hits_per_seed"]}
        row["ok"] = bool(abs(row["fixed_diff_pp"]) <= GATE_PP and mine_r == g["recal_threshold"]["hits_per_seed"]
                         and row["budget"] == g[FIXED]["budget"])
        out["rows"][key] = row; out["pass"] = out["pass"] and row["ok"]
    return out


def stats(gaps):
    g = np.asarray(gaps, float)
    return {"per_split_pp": [float(x) for x in g], "mean_pp": float(g.mean()), "sd_pp": float(g.std(ddof=1)),
            "n_positive": int((g > 0).sum()), "n_above_pass_pp": int((g > PASS_PP).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True); ap.add_argument("--label", default="proxy_error")
    ap.add_argument("--group", default="cell100"); ap.add_argument("--cheap", required=True)
    ap.add_argument("--budgets", default="0.05,0.10,0.25,0.50")
    ap.add_argument("--tune-splits", default="0,1,2,3"); ap.add_argument("--heldout-splits", default="4,5,6,7")
    ap.add_argument("--seeds", type=int, default=8); ap.add_argument("--ks", default="1")
    ap.add_argument("--ref", default=REF_G01); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert [int(k) for k in a.ks.split(",")] == [1], "L1b runs K = 1 only"
    fracs = [float(b) for b in a.budgets.split(",")]; seeds = list(range(a.seeds))
    tune = [int(x) for x in a.tune_splits.split(",")]; held = [int(x) for x in a.heldout_splits.split(",")]
    os.makedirs(a.out, exist_ok=True)
    jobs = [(a.feats, a.label, a.group, fracs, seeds, s) for s in sorted(set(tune + held))]
    t0 = time.time()
    with Pool(min(len(jobs), os.cpu_count())) as pool:
        res = pool.map(run_split, jobs)
    R = {r["split"]: r for r in res}
    print(f"ran {len(SPECS)} policies x {len(fracs)} budgets x {len(seeds)} seeds on splits {sorted(R)} in {time.time() - t0:.1f}s", flush=True)
    for s in sorted(R):
        print(f"split {s}: sim docs {R[s]['n_sim']} buckets {R[s]['n_buckets']} (groups before fit cut {R[s]['n_groups']}) "
              f"proxy errors {R[s]['hits_total']} calls {R[s]['budgets']}", flush=True)

    g = gate(R, a.ref, fracs)
    for key, row in g["rows"].items():
        print(f"gate budget {key}: fixed mine {row['fixed_mine_pct']:.4f} G01 {row['fixed_g01_pct']:.4f} diff {row['fixed_diff_pp']:+.4f} pp "
              f"(per seed equal {row['fixed_per_seed_equal']}) | recal mine {row['recal_per_seed_mine']} G01 {row['recal_per_seed_g01']} "
              f"match {row['recal_per_seed_mine'] == row['recal_per_seed_g01']} | ok {row['ok']}", flush=True)
    print(f"gate n_sim match {g['n_sim_match']}; reproduction gate {'PASS' if g['pass'] else 'FAIL'}", flush=True)
    if not g["pass"]:
        with open(os.path.join(a.out, "result.json"), "w") as fh:
            json.dump({"verdict": "GATE_FAILED", "gate": g}, fh, indent=1)
        print("GATE FAILED: tuning and held-out steps not run", flush=True)
        sys.exit(3)

    # Tuning on split seeds 0 to 3 only.
    sel, tune_score, per_budget = {}, {}, {}
    for f in fracs:
        key = f"{f:.2f}"
        tune_score[key] = {n: float(np.mean([rates(R, s, n, key).mean() for s in tune])) for n in [FIXED, REF] + SWEEP}
        sel[key] = max(SWEEP, key=lambda n: tune_score[key][n])  # ties go to the earlier config in sweep order

    # Held-out on split seeds 4 to 7: paired by tie-break seed, then averaged per split.
    gap_all = {}
    for f in fracs:
        key = f"{f:.2f}"
        gap_all[key] = {n: stats([(rates(R, s, n, key) - rates(R, s, FIXED, key)).mean() for s in held])
                        for n in SWEEP + [REF]}
        best = max(SWEEP, key=lambda n: gap_all[key][n]["mean_pp"])  # diagnostic only, not used for the verdict
        st = gap_all[key][sel[key]]; ref_st = gap_all[key][REF]
        per_budget[key] = {
            "calls": R[held[0]]["budgets"][key], "selected": sel[key],
            "tune_hit_pct": {"selected": tune_score[key][sel[key]], "fixed": tune_score[key][FIXED], "reference_recal": tune_score[key][REF]},
            "heldout_hit_pct": {
                "fixed": float(np.mean([rates(R, s, FIXED, key).mean() for s in held])),
                "reference_recal": float(np.mean([rates(R, s, REF, key).mean() for s in held])),
                "selected": float(np.mean([rates(R, s, sel[key], key).mean() for s in held]))},
            "heldout_selected_minus_fixed": st, "heldout_reference_minus_fixed": ref_st,
            "pass": bool(st["mean_pp"] > PASS_PP),
            "fragile_pass": bool(st["mean_pp"] > PASS_PP and st["n_positive"] == 1),
            "hindsight_best_diagnostic": {"config": best, "mean_pp": gap_all[key][best]["mean_pp"]},
        }
    verdict = "PASS" if any(v["pass"] for v in per_budget.values()) else "KILL"

    result = {"experiment": "L1b", "label": a.label, "group": a.group, "cheap_accepted_unused": a.cheap.split(","),
              "budgets": fracs, "seeds": seeds, "tune_splits": tune, "heldout_splits": held, "K": 1,
              "pass_threshold_pp": PASS_PP, "sd_ddof": 1, "n_policies": len(SPECS), "n_sweep_configs": len(SWEEP),
              "splits": {str(s): {k: R[s][k] for k in ("n_sim", "n_buckets", "n_groups", "hits_total", "budgets")} for s in sorted(R)},
              "gate": g, "per_budget": per_budget, "tune_hit_pct": tune_score, "heldout_gap_pp": gap_all,
              "verdict": verdict}
    with open(os.path.join(a.out, "result.json"), "w") as fh: json.dump(result, fh, indent=1)

    for key, row in per_budget.items():
        st = row["heldout_selected_minus_fixed"]
        print(f"budget {float(key):.0%} ({row['calls']} calls): selected {row['selected']} | held-out gap {st['mean_pp']:+.2f} pp "
              f"sd {st['sd_pp']:.2f} per split {[round(x, 2) for x in st['per_split_pp']]} | fixed {row['heldout_hit_pct']['fixed']:.2f} "
              f"recal {row['heldout_hit_pct']['reference_recal']:.2f} selected {row['heldout_hit_pct']['selected']:.2f} | "
              f"pass {row['pass']} fragile {row['fragile_pass']} | hindsight {row['hindsight_best_diagnostic']['config']} "
              f"{row['hindsight_best_diagnostic']['mean_pp']:+.2f}", flush=True)
    print(f"L1b verdict: {verdict}", flush=True)
    write_table(a.out, result)
    print(f"wrote {os.path.join(a.out, 'result.json')} and table.md", flush=True)


def write_table(out, r):
    L = ["# L1b: tuned label-free threshold vs plain fixed threshold", "",
         f"Verdict: **{r['verdict']}** (PASS if any budget has a held-out mean gap above +{PASS_PP:.1f} pp, tuned minus plain fixed).", "",
         "Hit rate (%) = proxy errors caught / verifier calls. Held-out gaps are paired by tie-break seed (8 per split), averaged per split; "
         "sd is over the 4 held-out splits (ddof 1).", "",
         "## Sim splits", "", "| split | role | sim docs | buckets | proxy errors | calls 5% | calls 10% | calls 25% | calls 50% |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s, v in r["splits"].items():
        role = "tune" if int(s) in r["tune_splits"] else "held-out"
        L.append(f"| {s} | {role} | {v['n_sim']} | {v['n_buckets']} | {v['hits_total']} | " +
                 " | ".join(str(v["budgets"][k]) for k in ("0.05", "0.10", "0.25", "0.50")) + " |")
    g = r["gate"]
    L += ["", f"## Reproduction gate (split 0, vs runs/G01): {'PASS' if g['pass'] else 'FAIL'}", "",
          "| budget | fixed mine (%) | fixed G01 (%) | diff (pp) | recal per seed matches G01 |", "|---|---|---|---|---|"]
    for k, row in g["rows"].items():
        L.append(f"| {float(k):.0%} | {row['fixed_mine_pct']:.2f} | {row['fixed_g01_pct']:.2f} | {row['fixed_diff_pp']:+.4f} | "
                 f"{row['recal_per_seed_mine'] == row['recal_per_seed_g01']} |")
    L += ["", "## Primary: held-out gap, selected tuned config minus plain fixed threshold (split seeds 4 to 7)", "",
          "| budget | selected config | fixed (%) | reference recal (%) | selected (%) | gap mean (pp) | sd over splits | gap per split 4 / 5 / 6 / 7 | pass | fragile |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, row in r["per_budget"].items():
        st = row["heldout_selected_minus_fixed"]; h = row["heldout_hit_pct"]
        L.append(f"| {float(k):.0%} | {row['selected']} | {h['fixed']:.2f} | {h['reference_recal']:.2f} | {h['selected']:.2f} | "
                 f"{st['mean_pp']:+.2f} | {st['sd_pp']:.2f} | {' / '.join(f'{x:+.2f}' for x in st['per_split_pp'])} | {row['pass']} | {row['fragile_pass']} |")
    L += ["", "## Reference recal (G01 rule, k0 = 2, agreement x n_distinct, R = max(budget/8, 40)) minus fixed, held-out", "",
          "| budget | gap mean (pp) | sd over splits | gap per split 4 / 5 / 6 / 7 |", "|---|---|---|---|"]
    for k, row in r["per_budget"].items():
        st = row["heldout_reference_minus_fixed"]
        L.append(f"| {float(k):.0%} | {st['mean_pp']:+.2f} | {st['sd_pp']:.2f} | {' / '.join(f'{x:+.2f}' for x in st['per_split_pp'])} |")
    L += ["", "## Diagnostic only: best config in hindsight on held-out (not used for the verdict)", "",
          "| budget | config | held-out gap mean (pp) |", "|---|---|---|"]
    for k, row in r["per_budget"].items():
        L.append(f"| {float(k):.0%} | {row['hindsight_best_diagnostic']['config']} | {row['hindsight_best_diagnostic']['mean_pp']:+.2f} |")
    L += ["", "## Full sweep (27 configs): tune hit rate (%, splits 0 to 3) and held-out gap vs fixed (pp, splits 4 to 7)", "",
          "| config | tune 5% | tune 10% | tune 25% | tune 50% | gap 5% | gap 10% | gap 25% | gap 50% |", "|---|---|---|---|---|---|---|---|---|"]
    keys = list(r["per_budget"].keys())
    for n in SWEEP:
        L.append(f"| {n} | " + " | ".join(f"{r['tune_hit_pct'][k][n]:.2f}" for k in keys) + " | " +
                 " | ".join(f"{r['heldout_gap_pp'][k][n]['mean_pp']:+.2f}" for k in keys) + " |")
    L += ["", "Tune hit rate for the plain fixed threshold and reference recal:", "",
          "| budget | fixed | reference recal |", "|---|---|---|"]
    for k in keys:
        L.append(f"| {float(k):.0%} | {r['tune_hit_pct'][k][FIXED]:.2f} | {r['tune_hit_pct'][k][REF]:.2f} |")
    L.append("")
    with open(os.path.join(out, "table.md"), "w") as fh: fh.write("\n".join(L))


if __name__ == "__main__":
    main()
