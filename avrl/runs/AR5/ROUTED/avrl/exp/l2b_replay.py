"""L2b CPU replay on the fresh pool. The two configs and the plain fixed rule were fixed from L1b before this run.
There is no tuning here: no argmax, no selection, and no PASS or KILL text from the L1b script.

Reuses the unchanged L1b module (avrl/exp/l1b_tuned_threshold.py) for build_sim, Sim, SPECS, hits_for, rates and
stats. One deliberate change: the reproduction gate against runs/G01 is not run. That reference is the E01 pool, so
it cannot match a fresh pool. In its place, every policy at every budget on every split must spend exactly
int(f * n_sim) verifier calls. The check replays each split with all labels set to 1, so hits equal calls.

Metric: per split, the held-out gap is the mean over 8 tie-break seeds of (config hit rate minus plain fixed hit
rate), in pp. Split seeds 0 to 7 are all evaluated. Config A: k0 = 2, R = max(budget // 8, 10), agreement x n_distinct,
scored at 5%. Config B: k0 = 50, R = max(budget // 8, 10), agreement x n_distinct x any_truncated, scored at 10%.
PASS if either mean gap is above +2.0 pp. The pass and kill rules as written are complementary (kill means both
means are at or below +2.0), so no separate inconclusive outcome is produced.

Writes <out>/replay.json, and <result_dir>/result.json and table.md.
"""
import os, sys, json, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import l1b_tuned_threshold as L1B

ROOT = "/homes/rdubash/avrl_live/ROUTED"
LABEL, GROUP = "proxy_error", "cell100"
SEEDS = list(range(8)); SPLITS = list(range(8)); FRACS = (0.05, 0.10)
CONFIGS = {"A": ("k02_Rdiv8_agreement_x_nd", "0.05"), "B": ("k050_Rdiv8_agreement_x_nd_x_trunc", "0.10")}
NAMES = dict(L1B.SPECS)
assert all(name in NAMES for name, _ in CONFIGS.values()) and L1B.FIXED in NAMES


def job(args):
    feats, split = args
    df = pd.read_parquet(feats)
    sim, n_groups = L1B.build_sim(df, GROUP, split)
    S = L1B.Sim(sim, LABEL)
    ones = L1B.Sim(sim.assign(**{LABEL: 1}), LABEL)  # hits with all labels 1 equal the verifier calls spent
    budgets = {f"{f:.2f}": int(f * S.n) for f in FRACS}
    hits, bad = {}, []
    for name, spec in L1B.SPECS:
        hits[name] = {}
        for key, b in budgets.items():
            hits[name][key] = [int(L1B.hits_for(S, spec, b, s)) for s in SEEDS]
            spent = [int(L1B.hits_for(ones, spec, b, s)) for s in SEEDS]
            if any(x != b for x in spent): bad.append({"policy": name, "budget": key, "calls": spent, "expected": b})
    return {"split": split, "n_sim": S.n, "n_buckets": int(sim[GROUP].nunique()), "n_groups": n_groups,
            "hits_total": int(S.y.sum()), "budgets": budgets, "hits": hits, "spend_bad": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=os.path.join(ROOT, "avrl", "runs", "L2b", "feats.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "avrl", "runs", "L2b", "replay"))
    ap.add_argument("--result_dir", default=os.path.join(ROOT, "avrl", "runs", "L2b"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    with Pool(min(len(SPLITS), os.cpu_count())) as pool:
        res = pool.map(job, [(a.feats, s) for s in SPLITS])
    R = {r["split"]: r for r in res}
    n_checks = len(SPLITS) * len(L1B.SPECS) * len(FRACS) * len(SEEDS)
    bad = [dict(split=s, **b) for s in SPLITS for b in R[s]["spend_bad"]]
    print(f"spend gate: {n_checks} policy x budget x seed checks on splits {SPLITS}, failures {len(bad)}", flush=True)
    for s in SPLITS:
        print(f"split {s}: sim docs {R[s]['n_sim']} buckets {R[s]['n_buckets']} proxy errors {R[s]['hits_total']} "
              f"calls {R[s]['budgets']}", flush=True)
    out = {"experiment": "L2b", "feats": a.feats, "label": LABEL, "group": GROUP, "seeds": SEEDS, "splits": SPLITS,
           "budgets": list(FRACS), "pass_threshold_pp": L1B.PASS_PP, "sd_ddof": 1,
           "spend_gate": {"checks": n_checks, "failures": len(bad), "examples": bad[:5], "pass": not bad},
           "splits_info": {str(s): {k: R[s][k] for k in ("n_sim", "n_buckets", "n_groups", "hits_total", "budgets")} for s in SPLITS},
           "configs": {}}
    if bad:
        out["verdict"] = "GATE_FAILED"
        with open(os.path.join(a.out, "replay.json"), "w") as fh: json.dump(out, fh, indent=1)
        print("SPEND GATE FAILED", flush=True); sys.exit(1)
    for tag, (name, key) in CONFIGS.items():
        gaps = [float((L1B.rates(R, s, name, key) - L1B.rates(R, s, L1B.FIXED, key)).mean()) for s in SPLITS]
        st = L1B.stats(gaps)
        out["configs"][tag] = {
            "config": name, "budget_frac": key, "calls_per_split": [R[s]["budgets"][key] for s in SPLITS],
            "fixed_hit_pct_per_split": [float(L1B.rates(R, s, L1B.FIXED, key).mean()) for s in SPLITS],
            "config_hit_pct_per_split": [float(L1B.rates(R, s, name, key).mean()) for s in SPLITS],
            "gap_pp_per_split": gaps, "mean_pp": st["mean_pp"], "sd_pp": st["sd_pp"],
            "n_splits_above_0": st["n_positive"], "n_splits_above_pass": st["n_above_pass_pp"],
            "pass": bool(st["mean_pp"] > L1B.PASS_PP)}
        print(f"config {tag} {name} at {float(key):.0%} ({R[0]['budgets'][key]} calls per split): gap per split "
              f"{[round(g, 2) for g in gaps]} mean {st['mean_pp']:+.2f} sd {st['sd_pp']:.2f} above 0 {st['n_positive']} "
              f"above +2.0 {st['n_above_pass_pp']}", flush=True)
    out["verdict"] = "PASS" if any(c["pass"] for c in out["configs"].values()) else "KILL"
    with open(os.path.join(a.out, "replay.json"), "w") as fh: json.dump(out, fh, indent=1)
    with open(os.path.join(a.result_dir, "result.json"), "w") as fh: json.dump(out, fh, indent=1)
    write_table(a.result_dir, out)
    print(f"L2b verdict: {out['verdict']}", flush=True)
    print(f"wrote {os.path.join(a.out, 'replay.json')}, {os.path.join(a.result_dir, 'result.json')} and table.md", flush=True)


def write_table(out_dir, r):
    A, B = r["configs"]["A"], r["configs"]["B"]
    L = ["# L2b: fresh-pool replication of the L1b label-free gains", "",
         f"Verdict: **{r['verdict']}** (PASS if the mean held-out gap is above +{L1B.PASS_PP:.1f} pp for config A at 5% or config B at 10%).", "",
         "Configs were fixed from L1b and not retuned on this pool. Plain fixed threshold is the baseline. Gap = config minus fixed, "
         "paired by tie-break seed (8 per split), averaged per split. Mean and sd are over the 8 split seeds (sd ddof 1).", "",
         "## Pool and sim splits", "", "| split | sim docs | buckets | proxy errors | calls 5% | calls 10% |", "|---|---|---|---|---|---|"]
    for s, v in r["splits_info"].items():
        L.append(f"| {s} | {v['n_sim']} | {v['n_buckets']} | {v['hits_total']} | {v['budgets']['0.05']} | {v['budgets']['0.10']} |")
    L += ["", f"Spend gate: {r['spend_gate']['checks']} policy x budget x seed checks, failures {r['spend_gate']['failures']}.", "",
          "## Held-out gap per split (pp)", "",
          "| split | A gap 5% | B gap 10% | fixed 5% (%) | A 5% (%) | fixed 10% (%) | B 10% (%) |", "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(r["splits"]):
        L.append(f"| {s} | {A['gap_pp_per_split'][i]:+.2f} | {B['gap_pp_per_split'][i]:+.2f} | {A['fixed_hit_pct_per_split'][i]:.2f} | "
                 f"{A['config_hit_pct_per_split'][i]:.2f} | {B['fixed_hit_pct_per_split'][i]:.2f} | {B['config_hit_pct_per_split'][i]:.2f} |")
    L += ["", "## Summary", "", "| config | budget | mean gap (pp) | sd over 8 splits | splits above 0 | splits above +2.0 | pass (mean > +2.0) |",
          "|---|---|---|---|---|---|---|"]
    for tag, c in (("A", A), ("B", B)):
        L.append(f"| {tag}: {c['config']} | {float(c['budget_frac']):.0%} | {c['mean_pp']:+.2f} | {c['sd_pp']:.2f} | "
                 f"{c['n_splits_above_0']} of 8 | {c['n_splits_above_pass']} of 8 | {c['pass']} |")
    L += ["", "Pass and kill rules as written are complementary (kill is both means at or below +2.0), so no inconclusive outcome is produced.", ""]
    with open(os.path.join(out_dir, "table.md"), "w") as fh: fh.write("\n".join(L))


if __name__ == "__main__":
    main()
