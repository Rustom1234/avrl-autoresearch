"""L4b: refit-floor verdict for configs A and B on the third GSM8K pool. CPU only. No retuning, no argmax.

Copied from avrl/exp/l3a_floor_sensitivity.py. Changes:
  Policies: plain fixed threshold; config A at 5% (k0 2, stratum agreement x n_distinct); config B at 10% (k0 50,
  stratum agreement x n_distinct x any_truncated). Each runs at refit floor 40 (verdict) and floor 10 (reference). The
  floor 10 reference is computed here on the same pool; L3a read it from runs/L2b, which is a different pool.
  Verdict: per config, PASS if the mean gap over split seeds 0 to 7 at floor 40 is above +2.0 pp, else KILL. Each
  config is judged separately; there is no inconclusive band.
  Reproduction gate: runs first, on the L1B pool (runs/p0/feats.parquet). Config A at 5% and config B at 10%, floor 40,
  must match runs/L3a/l1b_pool/result.json per split and in mean to 0.01 pp. If it fails, no L4b verdict is written.
  Split seeds 0 to 7, 8 tie-break seeds, per-seed pairing and the spend gate (every policy takes int(f x n_sim) calls on
  every split) are as in L3a.

Usage: python -I l4b_replay.py --feats FEATS --out OUTDIR --result_dir RESULT_DIR [--repro_only]
Writes <out>/repro_l1b_pool.json, <out>/replay.json, <result_dir>/result.json and <result_dir>/table.md.
"""
import os, sys, json, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import l1b_tuned_threshold as L1B

ROOT = "/homes/rdubash/avrl_live/ROUTED"
L1B_POOL = os.path.join(ROOT, "avrl", "runs", "p0", "feats.parquet")
L3A_L1B_RESULT = os.path.join(ROOT, "avrl", "runs", "L3a", "l1b_pool", "result.json")
LABEL, GROUP = "proxy_error", "cell100"
SEEDS = list(range(8)); SPLITS = list(range(8)); FRACS = (0.05, 0.10)
A, B = "A", "B"
CONFIGS = {A: (2, "agreement_x_nd", "0.05"), B: (50, "agreement_x_nd_x_trunc", "0.10")}  # k0, stratum, scored budget
ARMS = {"A40": (A, 40), "A10": (A, 10), "B40": (B, 40), "B10": (B, 10)}  # arm: (config, refit floor)
POLICIES = [L1B.FIXED] + list(ARMS)
REPRO_TOL_PP = 0.01


def hits_for(S, pol, budget, seed):
    if pol == L1B.FIXED: return S.fixed(budget, seed)
    tag, floor = ARMS[pol]; k0, stratum, _ = CONFIGS[tag]
    return S.recal(budget, seed, k0, max(budget // 8, floor), stratum)


def key_of(arm): return CONFIGS[ARMS[arm][0]][2]


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


def run_pool(feats):
    with Pool(min(len(SPLITS), os.cpu_count())) as p:
        res = p.map(job, [(feats, s) for s in SPLITS])
    return {r["split"]: r for r in res}


def gap(R, arm, s):
    """Per-seed gap in pp (config minus plain fixed) on split s at the arm's budget. The split gap is its mean over the 8 tie-break seeds."""
    key = key_of(arm)
    return L1B.rates(R, s, arm, key) - L1B.rates(R, s, L1B.FIXED, key)


def spend_gate(R):
    bad = [dict(split=s, **b) for s in SPLITS for b in R[s]["spend_bad"]]
    checks = len(SPLITS) * len(POLICIES) * len(FRACS) * len(SEEDS)
    return {"checks": checks, "failures": len(bad), "examples": bad[:5], "pass": not bad}


def summarize(R, arm):
    tag, floor = ARMS[arm]; k0, stratum, _ = CONFIGS[tag]; key = key_of(arm)
    g_all = [float(gap(R, arm, s).mean()) for s in SPLITS]
    return {"config": f"k0 {k0}, stratum {stratum}, R = max(budget // 8, {floor})", "budget_frac": key, "refit_floor": floor,
            "calls_per_split": [R[s]["budgets"][key] for s in SPLITS],
            "fixed_hit_pct_per_split": [float(L1B.rates(R, s, L1B.FIXED, key).mean()) for s in SPLITS],
            "config_hit_pct_per_split": [float(L1B.rates(R, s, arm, key).mean()) for s in SPLITS],
            "gap_pp_per_split": g_all, "all_splits": L1B.stats(g_all)}


def repro_check(R):
    ref = json.load(open(L3A_L1B_RESULT))["configs"]; arms = {}
    for arm, tag in (("A40", A), ("B40", B)):
        mine = [float(gap(R, arm, s).mean()) for s in SPLITS]
        refv = [float(x) for x in ref[tag]["gap_pp_per_split"]]; ref_mean = float(ref[tag]["all_splits"]["mean_pp"])
        d = max(abs(m - r) for m, r in zip(mine, refv)); mean_mine = float(np.mean(mine))
        arms[arm] = {"ref_per_split_pp": refv, "mine_per_split_pp": mine, "ref_mean_pp": ref_mean, "mine_mean_pp": mean_mine,
                     "max_abs_diff_pp": d, "pass": bool(d <= REPRO_TOL_PP and abs(mean_mine - ref_mean) <= REPRO_TOL_PP)}
    return all(v["pass"] for v in arms.values()), arms


def verdict(c):
    return "PASS" if c["all_splits"]["mean_pp"] > L1B.PASS_PP else "KILL"


def fail(a, out, msg):
    out["verdict"] = "GATE_FAILED"; out["gate_message"] = msg
    with open(os.path.join(a.out, "replay.json"), "w") as fh: json.dump(out, fh, indent=1)
    with open(os.path.join(a.result_dir, "result.json"), "w") as fh: json.dump(out, fh, indent=1)
    print(msg, flush=True); sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True); ap.add_argument("--out", required=True); ap.add_argument("--result_dir", required=True)
    ap.add_argument("--repro_only", action="store_true", help="run the reproduction gate on the L1B pool and stop")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs(a.result_dir, exist_ok=True)

    # 1. Reproduction gate on the L1B pool. Must pass before any L4b number is read.
    R0 = run_pool(L1B_POOL)
    sg0 = spend_gate(R0); repro_ok, arms0 = repro_check(R0)
    rep = {"feats": L1B_POOL, "splits": SPLITS, "spend_gate": sg0, "repro_gate": {"pass": repro_ok, "tol_pp": REPRO_TOL_PP, "arms": arms0},
           "configs": {arm: summarize(R0, arm) for arm in ARMS}}
    with open(os.path.join(a.out, "repro_l1b_pool.json"), "w") as fh: json.dump(rep, fh, indent=1)
    for arm, v in arms0.items():
        print(f"reproduction {arm}: mine {[round(x, 4) for x in v['mine_per_split_pp']]} ref {[round(x, 4) for x in v['ref_per_split_pp']]} "
              f"mean {v['mine_mean_pp']:+.4f} vs ref {v['ref_mean_pp']:+.4f} max abs diff {v['max_abs_diff_pp']:.6f} pp", flush=True)
    print(f"L1B pool spend gate: {sg0['checks']} checks, failures {sg0['failures']}; reproduction gate: {'PASS' if repro_ok else 'FAIL'}", flush=True)
    if not sg0["pass"]: print("SPEND GATE FAILED on the L1B pool", flush=True); sys.exit(1)
    if not repro_ok: print("REPRODUCTION GATE FAILED: no L4b verdict written", flush=True); sys.exit(1)
    if a.repro_only: return

    # 2. L4b pool: verdict at floor 40, floor 10 as reference.
    R = run_pool(a.feats)
    sg = spend_gate(R)
    out = {"experiment": "L4b", "pool": "l4b_pool3", "feats": a.feats, "label": LABEL, "group": GROUP, "seeds": SEEDS,
           "splits": SPLITS, "budgets": list(FRACS), "refit_floors": {"verdict": 40, "reference": 10},
           "pass_threshold_pp": L1B.PASS_PP, "sd_ddof": 1, "spend_gate": sg,
           "repro_gate": {"pass": repro_ok, "file": os.path.join(a.out, "repro_l1b_pool.json")},
           "splits_info": {str(s): {k: R[s][k] for k in ("n_sim", "n_buckets", "n_groups", "hits_total", "budgets")} for s in SPLITS}}
    if not sg["pass"]: fail(a, out, "SPEND GATE FAILED")
    out["configs"] = {arm: summarize(R, arm) for arm in ARMS}
    out["verdicts"] = {"A_5pct_floor40": verdict(out["configs"]["A40"]), "B_10pct_floor40": verdict(out["configs"]["B40"])}
    for arm in ARMS:
        c = out["configs"][arm]; s_ = c["all_splits"]
        print(f"arm {arm} ({c['config']}, budget {c['budget_frac']}): gap per split {[round(g, 2) for g in c['gap_pp_per_split']]} "
              f"mean {s_['mean_pp']:+.2f} sd {s_['sd_pp']:.2f} above 0 {s_['n_positive']} above +2.0 {s_['n_above_pass_pp']} "
              f"calls {c['calls_per_split']}", flush=True)
    print(f"L4b verdicts at floor 40: A 5% {out['verdicts']['A_5pct_floor40']}, B 10% {out['verdicts']['B_10pct_floor40']}", flush=True)
    with open(os.path.join(a.out, "replay.json"), "w") as fh: json.dump(out, fh, indent=1)
    with open(os.path.join(a.result_dir, "result.json"), "w") as fh: json.dump(out, fh, indent=1)
    write_table(a.result_dir, out)
    print(f"wrote {os.path.join(a.out, 'replay.json')}, {os.path.join(a.result_dir, 'result.json')} and table.md", flush=True)


def write_table(out_dir, r):
    C = r["configs"]; V = r["verdicts"]
    L = ["# L4b: third GSM8K pool, configs A (5%) and B (10%)", "",
         f"Verdict rule: a config's mean gap over split seeds 0 to 7 at refit floor 40, paired against plain fixed, above +{L1B.PASS_PP:.1f} pp is PASS, else KILL. "
         "Each config is judged separately. Floor 10 is reference only.", "",
         f"Reproduction gate on the L1B pool (runs/L3a/l1b_pool): PASS. Spend gate: {r['spend_gate']['checks']} checks, failures {r['spend_gate']['failures']}.", "",
         "## Verdict at refit floor 40", "",
         "| config | budget | mean gap (pp) | sd (pp) | splits above 0 | splits above +2.0 | calls per split | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    for arm, label, vk in (("A40", "A: k0 2, agreement x n_distinct", "A_5pct_floor40"), ("B40", "B: k0 50, agreement x n_distinct x any_truncated", "B_10pct_floor40")):
        c = C[arm]; s_ = c["all_splits"]
        L.append(f"| {label} | {float(c['budget_frac']):.0%} | {s_['mean_pp']:+.2f} | {s_['sd_pp']:.2f} | {s_['n_positive']} of 8 | "
                 f"{s_['n_above_pass_pp']} of 8 | {min(c['calls_per_split'])} to {max(c['calls_per_split'])} | **{V[vk]}** |")
    L += ["", "## Floor 10 reference (same pool)", "", "| config | mean gap (pp) | sd (pp) | gap per split (pp) |", "|---|---|---|---|"]
    for arm, label in (("A10", "A at 5%, floor 10"), ("B10", "B at 10%, floor 10")):
        s_ = C[arm]["all_splits"]
        L.append(f"| {label} | {s_['mean_pp']:+.2f} | {s_['sd_pp']:.2f} | {' / '.join(f'{x:+.2f}' for x in s_['per_split_pp'])} |")
    L += ["", "## Gap per split (pp), refit floor 40", "", "| split | A 5% | B 10% | fixed 5% (%) | A 5% (%) | fixed 10% (%) | B 10% (%) |",
          "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(r["splits"]):
        L.append(f"| {s} | {C['A40']['gap_pp_per_split'][i]:+.2f} | {C['B40']['gap_pp_per_split'][i]:+.2f} | "
                 f"{C['A40']['fixed_hit_pct_per_split'][i]:.2f} | {C['A40']['config_hit_pct_per_split'][i]:.2f} | "
                 f"{C['B40']['fixed_hit_pct_per_split'][i]:.2f} | {C['B40']['config_hit_pct_per_split'][i]:.2f} |")
    L.append("")
    with open(os.path.join(out_dir, "table.md"), "w") as fh: fh.write("\n".join(L))


if __name__ == "__main__":
    main()
