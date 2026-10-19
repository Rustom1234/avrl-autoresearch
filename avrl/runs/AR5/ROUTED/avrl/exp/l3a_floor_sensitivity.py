"""L3a: refit-floor sensitivity of the L2b label-free gains. CPU replay only: no GPU, no retuning, no argmax.

Copied from avrl/exp/l2b_replay.py. The one change is the refit floor: R = max(budget // 8, 40) for both configs,
where L2b used 10. The policies are:
  plain fixed threshold (L1B.FIXED), the baseline;
  config A at 5%: k0 2, stratum agreement x n_distinct (the L1B reference_recal branch, the G01 rule);
  config B at 10%: k0 50, stratum agreement x n_distinct x any_truncated.
Split seeds 0 to 7, 8 tie-break seeds, per-seed pairing and the spend gate are as in L2b.

Gate B (reproduction) runs only when --feats is the L1B pool (avrl/runs/p0/feats.parquet): config A at 5% on splits
4 to 7 must match avrl/runs/L1b/result.json heldout_gap_pp["0.05"]["reference_recal"]["per_split_pp"] to 0.01 pp.

Verdict on the fresh pool: PASS if config A at 5% has a mean gap over split seeds 0 to 7 above +2.0 pp, else KILL.
Config B at 10% is a secondary read against the same bar and does not decide the verdict. On the L1B pool the
outcome is CONTEXT_ONLY and the numbers sit beside the floor-10 values.

Writes <out>/replay.json, and <result_dir>/result.json and table.md. Both paths are required so that no earlier run
(L2b, L1b) is overwritten by a default.
"""
import os, sys, json, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import l1b_tuned_threshold as L1B

ROOT = "/homes/rdubash/avrl_live/ROUTED"
L1B_POOL = os.path.join(ROOT, "avrl", "runs", "p0", "feats.parquet")
L1B_RESULT = os.path.join(ROOT, "avrl", "runs", "L1b", "result.json")
L2B_RESULT = os.path.join(ROOT, "avrl", "runs", "L2b", "result.json")
LABEL, GROUP = "proxy_error", "cell100"
SEEDS = list(range(8)); SPLITS = list(range(8)); HELD = [4, 5, 6, 7]; FRACS = (0.05, 0.10)
FLOOR = 40; REPRO_TOL_PP = 0.01
A, B = "A", "B"
CONFIGS = {A: (2, "agreement_x_nd", "0.05"), B: (50, "agreement_x_nd_x_trunc", "0.10")}  # k0, stratum, scored budget
POLICIES = [L1B.FIXED, A, B]


def hits_for(S, pol, budget, seed):
    if pol == L1B.FIXED: return S.fixed(budget, seed)
    k0, stratum, _ = CONFIGS[pol]
    return S.recal(budget, seed, k0, max(budget // 8, FLOOR), stratum)


def job(args):
    feats, split = args
    df = pd.read_parquet(feats)
    sim, n_groups = L1B.build_sim(df, GROUP, split)
    S = L1B.Sim(sim, LABEL)
    ones = L1B.Sim(sim.assign(**{LABEL: 1}), LABEL)  # all labels 1, so hits equal the verifier calls spent
    budgets = {f"{f:.2f}": int(f * S.n) for f in FRACS}
    hits, bad = {}, []
    for pol in POLICIES:
        hits[pol] = {}
        for key, b in budgets.items():
            hits[pol][key] = [int(hits_for(S, pol, b, s)) for s in SEEDS]
            spent = [int(hits_for(ones, pol, b, s)) for s in SEEDS]
            if any(x != b for x in spent): bad.append({"policy": pol, "budget": key, "calls": spent, "expected": b})
    return {"split": split, "n_sim": S.n, "n_buckets": int(sim[GROUP].nunique()), "n_groups": n_groups,
            "hits_total": int(S.y.sum()), "budgets": budgets, "hits": hits, "spend_bad": bad}


def gap(R, tag, key, s):
    """Per-seed gap in pp (config minus plain fixed) on split s. The split gap is its mean over the 8 tie-break seeds."""
    return L1B.rates(R, s, tag, key) - L1B.rates(R, s, L1B.FIXED, key)


def fail(a, out, msg):
    out["verdict"] = "GATE_FAILED"; out["gate_message"] = msg
    with open(os.path.join(a.out, "replay.json"), "w") as fh: json.dump(out, fh, indent=1)
    with open(os.path.join(a.result_dir, "result.json"), "w") as fh: json.dump(out, fh, indent=1)
    print(msg, flush=True); sys.exit(1)


def normalise(c):
    return {"gap_pp_per_split": [float(x) for x in c["gap_pp_per_split"]], "mean_pp": float(c["mean_pp"]), "sd_pp": float(c["sd_pp"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True); ap.add_argument("--out", required=True); ap.add_argument("--result_dir", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs(a.result_dir, exist_ok=True)
    pool = "l1b_pool" if os.path.abspath(a.feats) == L1B_POOL else "fresh"
    with Pool(min(len(SPLITS), os.cpu_count())) as p:
        res = p.map(job, [(a.feats, s) for s in SPLITS])
    R = {r["split"]: r for r in res}
    n_checks = len(SPLITS) * len(POLICIES) * len(FRACS) * len(SEEDS)
    bad = [dict(split=s, **b) for s in SPLITS for b in R[s]["spend_bad"]]
    print(f"spend gate: {n_checks} policy x budget x seed checks on splits {SPLITS}, failures {len(bad)}", flush=True)
    for s in SPLITS:
        print(f"split {s}: sim docs {R[s]['n_sim']} buckets {R[s]['n_buckets']} proxy errors {R[s]['hits_total']} "
              f"calls {R[s]['budgets']}", flush=True)
    out = {"experiment": "L3a", "pool": pool, "feats": a.feats, "label": LABEL, "group": GROUP, "seeds": SEEDS,
           "splits": SPLITS, "held_splits": HELD, "budgets": list(FRACS), "refit_floor": FLOOR,
           "pass_threshold_pp": L1B.PASS_PP, "sd_ddof": 1,
           "spend_gate": {"checks": n_checks, "failures": len(bad), "examples": bad[:5], "pass": not bad},
           "splits_info": {str(s): {k: R[s][k] for k in ("n_sim", "n_buckets", "n_groups", "hits_total", "budgets")} for s in SPLITS},
           "repro_gate": {"applies": pool == "l1b_pool", "pass": None}, "configs": {}}
    if bad: fail(a, out, "SPEND GATE FAILED")
    if pool == "l1b_pool":
        ref = json.load(open(L1B_RESULT))["heldout_gap_pp"]["0.05"]["reference_recal"]["per_split_pp"]
        mine = [float(gap(R, A, "0.05", s).mean()) for s in HELD]
        max_diff = max(abs(m - r) for m, r in zip(mine, ref))
        out["repro_gate"].update({"ref_per_split_pp": ref, "mine_per_split_pp": mine, "max_abs_diff_pp": max_diff,
                                  "tol_pp": REPRO_TOL_PP, "pass": bool(max_diff <= REPRO_TOL_PP)})
        print(f"reproduction gate on splits {HELD}: L1b {ref} mine {[round(m, 6) for m in mine]} "
              f"max abs diff {max_diff:.6f} pp -> {'PASS' if out['repro_gate']['pass'] else 'FAIL'}", flush=True)
        if not out["repro_gate"]["pass"]: fail(a, out, "REPRODUCTION GATE FAILED")
    for tag, (k0, stratum, key) in CONFIGS.items():
        g_all = [float(gap(R, tag, key, s).mean()) for s in SPLITS]
        st = L1B.stats(g_all)
        out["configs"][tag] = {
            "config": f"k0 {k0}, stratum {stratum}, R = max(budget // 8, {FLOOR})", "budget_frac": key,
            "calls_per_split": [R[s]["budgets"][key] for s in SPLITS],
            "fixed_hit_pct_per_split": [float(L1B.rates(R, s, L1B.FIXED, key).mean()) for s in SPLITS],
            "config_hit_pct_per_split": [float(L1B.rates(R, s, tag, key).mean()) for s in SPLITS],
            "gap_pp_per_split": g_all, "all_splits": st,
            "held_splits_4_to_7": L1B.stats([g_all[SPLITS.index(s)] for s in HELD])}
        print(f"config {tag} {stratum} at {float(key):.0%} floor {FLOOR}: gap per split {[round(g, 2) for g in g_all]} "
              f"mean {st['mean_pp']:+.2f} sd {st['sd_pp']:.2f} above 0 {st['n_positive']} above +2.0 {st['n_above_pass_pp']} "
              f"calls {out['configs'][tag]['calls_per_split']}", flush=True)
    out["verdict"] = ("PASS" if out["configs"][A]["all_splits"]["mean_pp"] > L1B.PASS_PP else "KILL") if pool == "fresh" else "CONTEXT_ONLY"
    out["config_B_mean_above_pass"] = bool(out["configs"][B]["all_splits"]["mean_pp"] > L1B.PASS_PP)
    if pool == "fresh":
        L2 = json.load(open(L2B_RESULT))["configs"]
        out["floor10_reference"] = {"source": L2B_RESULT, "splits": "0 to 7", A: normalise(L2[A]), B: normalise(L2[B])}
    else:
        L1 = json.load(open(L1B_RESULT))["heldout_gap_pp"]
        ref10 = {"source": L1B_RESULT, "splits": "4 to 7"}
        ref10[A] = normalise(dict(L1["0.05"]["k02_Rdiv8_agreement_x_nd"], gap_pp_per_split=L1["0.05"]["k02_Rdiv8_agreement_x_nd"]["per_split_pp"]))
        ref10[B] = normalise(dict(L1["0.10"]["k050_Rdiv8_agreement_x_nd_x_trunc"], gap_pp_per_split=L1["0.10"]["k050_Rdiv8_agreement_x_nd_x_trunc"]["per_split_pp"]))
        out["floor10_reference"] = ref10
    with open(os.path.join(a.out, "replay.json"), "w") as fh: json.dump(out, fh, indent=1)
    with open(os.path.join(a.result_dir, "result.json"), "w") as fh: json.dump(out, fh, indent=1)
    write_table(a.result_dir, out)
    print(f"L3a pool {pool} verdict: {out['verdict']}", flush=True)
    print(f"wrote {os.path.join(a.out, 'replay.json')}, {os.path.join(a.result_dir, 'result.json')} and table.md", flush=True)


def write_table(out_dir, r):
    A_, B_, f = r["configs"][A], r["configs"][B], r["floor10_reference"]
    L = [f"# L3a: refit-floor {FLOOR} sensitivity, pool {r['pool']}", "",
         f"Verdict rule: config A at 5%, mean gap over split seeds 0 to 7 above +{L1B.PASS_PP:.1f} pp is PASS, else KILL. "
         f"Config B at 10% is a secondary read against the same bar. Verdict for this pool: **{r['verdict']}**.", "",
         f"Refit cadence R = max(budget // 8, {FLOOR}). Gap = config minus plain fixed, in pp, paired by tie-break seed "
         "(8 per split), averaged per split. sd is ddof 1. Compare like with like: the fresh pool compares splits 0 to 7 "
         "with floor 10 on splits 0 to 7; the L1B pool compares splits 4 to 7 with floor 10 on splits 4 to 7.", "",
         f"Spend gate: {r['spend_gate']['checks']} policy x budget x seed checks, failures {r['spend_gate']['failures']}."]
    rg = r["repro_gate"]
    if rg["applies"]:
        L.append(f"Reproduction gate (config A at 5% on splits 4 to 7 vs avrl/runs/L1b/result.json): "
                 f"{'PASS' if rg['pass'] else 'FAIL'}, max abs diff {rg['max_abs_diff_pp']:.6f} pp (tolerance {rg['tol_pp']} pp).")
    L += ["", f"## Floor {FLOOR} summary", "", "| config | splits | mean gap (pp) | sd | splits above 0 | splits above +2.0 |",
          "|---|---|---|---|---|---|"]
    for tag, c in ((A, A_), (B, B_)):
        for scope, label in (("all_splits", "0 to 7"), ("held_splits_4_to_7", "4 to 7")):
            s_ = c[scope]; n = len(s_["per_split_pp"])
            L.append(f"| {tag}: {c['config']} at {float(c['budget_frac']):.0%} | {label} | {s_['mean_pp']:+.2f} | {s_['sd_pp']:.2f} | "
                     f"{s_['n_positive']} of {n} | {s_['n_above_pass_pp']} of {n} |")
    L += ["", f"## Floor 10 reference ({f['splits']}, from {f['source']})", "", "| config | mean gap (pp) | sd | gap per split |", "|---|---|---|---|"]
    for tag, scale in ((A, "5%"), (B, "10%")):
        L.append(f"| {tag}: same rule at {scale}, floor 10 | {f[tag]['mean_pp']:+.2f} | {f[tag]['sd_pp']:.2f} | "
                 f"{' / '.join(f'{x:+.2f}' for x in f[tag]['gap_pp_per_split'])} |")
    L += ["", "## Gap per split (pp), floor 40", "", "| split | A 5% | B 10% | fixed 5% (%) | A 5% (%) | fixed 10% (%) | B 10% (%) |",
          "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(r["splits"]):
        L.append(f"| {s} | {A_['gap_pp_per_split'][i]:+.2f} | {B_['gap_pp_per_split'][i]:+.2f} | {A_['fixed_hit_pct_per_split'][i]:.2f} | "
                 f"{A_['config_hit_pct_per_split'][i]:.2f} | {B_['fixed_hit_pct_per_split'][i]:.2f} | {B_['config_hit_pct_per_split'][i]:.2f} |")
    L.append("")
    with open(os.path.join(out_dir, "table.md"), "w") as fh: fh.write("\n".join(L))


if __name__ == "__main__":
    main()
