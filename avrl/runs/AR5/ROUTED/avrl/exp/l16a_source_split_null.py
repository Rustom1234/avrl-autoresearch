"""L16a: source-split null for the registered test-row gain. CPU only (local shell), no GPU job, no ssh, no pip.

Question. The registered test rows (L4b pid <= 1318, 1,319 GSM8K test questions) give frozen minus plain fixed = +1.22 pp at a
10% budget (runs/L13a/L4b_pid_le1318), and the remaining L4b rows give +8.58 pp. Does the same comparison on random 1,319-prompt
cuts of the 3,000-prompt L2b pool (GSM8K train questions, disjoint from p0 and from the test questions) span +1.22 pp?

Protocol. Every piece is the L13a protocol, imported from avrl/exp/l13a_shrunk_refit.py (not edited):
  fit     20 frozen 200-label tables from runs/p0/feats.parquet: subset i = L13A.draw_subset (numpy RandomState(100 + i)),
          L13A.fit_table with k0 = 50, prior 1 - agreement.
  score   L13A.attach_score: each row gets its stratum's table posterior (the prior if the stratum is missing). No eval label is used.
  cuts    48 unweighted cuts c = 0 to 47 of 1,319 rows from runs/L2b/feats.parquet: RandomState(100 + c).choice(3000, 1319,
          replace=False), kept in ascending file order. The cuts overlap each other, so they are not independent draws.
  boot    200 bootstrap draws (RandomState(16).randint(0, 1319, size=(200, 1319))) of the registered test rows, each kept in
          ascending order with duplicates (ASSUMPTION, not in the spec: a row-level nonparametric bootstrap).
  scoring per frame and per fit subset: L1B.build_sim(scored frame, cell100, split) for split seeds 0 to 7, budget int(0.10 x n_sim),
          fixed = L13A.fixed_pick and frozen = L13A.frozen_pick over tie-break seeds 0 to 7. Gap per (subset, split) = mean over tie
          seeds of 100 (frozen hits - fixed hits) / budget. Subset value = mean over splits. Frame value (the metric) = mean over the
          20 subsets, in pp.
  Only the fixed and frozen arms are scored. The offset-ridge refit arms of L13a do not enter frozen minus fixed, so they are not run.

Gates (all before any number is reported):
  G1 reproduction: the registered test rows give frozen minus fixed equal to runs/L13a/L4b_pid_le1318 (+1.222287) and the full
     L2b pool gives runs/L13a/L2b (+4.723665), each within 1e-6 pp; the fixed-arm and frozen-arm mean rates match those files too.
  G2 spend: every fixed and frozen pick has int(0.10 x n_sim) distinct rows (zero failures).
  G3 overlap: zero eval questions occur in p0, for every frame.
  G4 cut frames hold 1,319 distinct pids.

Modes: gate; cuts A B (cut indices A to B-1); boot A B (draw indices A to B-1); combine (run after the replay, see L16a.sh).
Writes under --out (avrl/runs/L16a). Exit code 1 when a gate fails.
"""
import os, sys, json, argparse, glob
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import l1b_tuned_threshold as L1B
import l13a_shrunk_refit as L13A

RUNS = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
P0_FEATS = f"{RUNS}/p0/feats.parquet"
L2B_FEATS = f"{RUNS}/L2b/feats.parquet"
L4B_FEATS = f"{RUNS}/L4b/feats.parquet"
REF_TEST = f"{RUNS}/L13a/L4b_pid_le1318/result.json"
REF_L2B = f"{RUNS}/L13a/L2b/result.json"
LABEL, GROUP = "proxy_error", "cell100"
K0, BUDGET_FRAC = 50.0, 0.10
N_SUBSETS, SUBSET_SIZE = 20, 200
SPLITS, SEEDS = list(range(8)), list(range(8))
N_CUTS, CUT_SIZE, CUT_SEED0 = 48, 1319, 100
N_BOOT, BOOT_SEED = 200, 16
PID_TEST_MAX = 1318
TOL_PP = 1e-6
PROCS = 8


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_inputs():
    fit = pd.read_parquet(P0_FEATS)
    l2b = pd.read_parquet(L2B_FEATS)
    l4b = pd.read_parquet(L4B_FEATS)
    test = l4b[l4b["pid"].to_numpy() <= PID_TEST_MAX].reset_index(drop=True)
    return fit, l2b, test


def subset_tables(fit):
    """The 20 frozen 200-label tables, built exactly as the L13a subset loop builds them."""
    return [L13A.fit_table(fit.iloc[L13A.draw_subset(fit, i, SUBSET_SIZE)], K0) for i in range(N_SUBSETS)]


def split_job(args):
    scored, split = args
    sim, _ = L1B.build_sim(scored, GROUP, split)
    S = L1B.Sim(sim, LABEL)
    score = sim["frozen_score"].to_numpy(float)
    budget = int(BUDGET_FRAC * S.n)
    gaps, fixed_pct, frozen_pct, spend_bad = [], [], [], []
    for s in SEEDS:
        sel_f = L13A.fixed_pick(S, budget, s)
        sel_z = L13A.frozen_pick(S, score, budget, s)
        hf = int(S.y[sel_f].sum())
        assert hf == S.fixed(budget, s), "fixed_pick disagrees with L1B.Sim.fixed"
        hz = int(S.y[sel_z].sum())
        for name, sel in (("fixed", sel_f), ("frozen", sel_z)):
            d = int(len(np.unique(sel)))
            if len(sel) != budget or d != budget:
                spend_bad.append({"policy": name, "seed": int(s), "distinct": d, "expected": budget})
        f, z = 100.0 * hf / budget, 100.0 * hz / budget
        gaps.append(z - f); fixed_pct.append(f); frozen_pct.append(z)
    return {"split": split, "n_sim": int(S.n), "budget": budget, "gap_pp": float(np.mean(gaps)),
            "fixed_pct": float(np.mean(fixed_pct)), "frozen_pct": float(np.mean(frozen_pct)), "spend_bad": spend_bad}


def evaluate(frame, tables):
    """L13a frozen-minus-fixed summary of one eval frame, averaged over the fit subsets."""
    scored = [L13A.attach_score(frame, t) for t in tables]
    jobs = [(sc, s) for sc in scored for s in SPLITS]
    with Pool(min(PROCS, len(jobs))) as p:
        res = p.map(split_job, jobs)
    k = len(SPLITS)
    per = [res[i * k:(i + 1) * k] for i in range(len(tables))]
    for rows in per:
        assert [r["split"] for r in rows] == SPLITS
    n_sim = [per[0][j]["n_sim"] for j in range(k)]
    for rows in per:
        assert [r["n_sim"] for r in rows] == n_sim, "n_sim must depend on the split only"
    subset_gap = [float(np.mean([r["gap_pp"] for r in rows])) for rows in per]
    spend_bad = [dict(subset=i, **b) for i, rows in enumerate(per) for r in rows for b in r["spend_bad"]]
    return {
        "mean_pp": float(np.mean(subset_gap)),
        "subset_mean_pp": subset_gap,
        "split_mean_pp_over_subsets": [float(np.mean([rows[j]["gap_pp"] for rows in per])) for j in range(k)],
        "fixed_pct_mean": float(np.mean([r["fixed_pct"] for rows in per for r in rows])),
        "frozen_pct_mean": float(np.mean([r["frozen_pct"] for rows in per for r in rows])),
        "n_sim_per_split": n_sim,
        "budget_per_split": [int(BUDGET_FRAC * n) for n in n_sim],
        "spend_checks": 2 * len(SEEDS) * k * len(tables),
        "spend_failures": len(spend_bad), "spend_examples": spend_bad[:5],
        "rows": int(len(frame)), "distinct_pids": int(frame["pid"].nunique()),
    }


def gate_mode(out):
    fit, l2b, test = load_inputs()
    tables = subset_tables(fit)
    p0_q = set(fit["question"])
    ref_t = json.load(open(REF_TEST))["pools"]["L4b"]
    ref_l = json.load(open(REF_L2B))["pools"]["L2b"]
    t = evaluate(test, tables); t["overlap_with_p0"] = int(test["question"].isin(p0_q).sum())
    l = evaluate(l2b, tables); l["overlap_with_p0"] = int(l2b["question"].isin(p0_q).sum())

    def check(mine, ref):
        d = abs(mine - ref)
        return {"mine": mine, "ref": ref, "abs_diff_pp": d, "tol_pp": TOL_PP, "pass": bool(d <= TOL_PP)}

    checks = int(t["spend_checks"] + l["spend_checks"]); fails = int(t["spend_failures"] + l["spend_failures"])
    gates = {
        "test_rows_frozen_minus_fixed": check(t["mean_pp"], float(ref_t["frozen_vs_fixed_mean_pp"])),
        "test_rows_fixed_pct_mean": check(t["fixed_pct_mean"], float(ref_t["fixed_pct_mean"])),
        "test_rows_frozen_pct_mean": check(t["frozen_pct_mean"], float(ref_t["frozen_pct_mean"])),
        "l2b_frozen_minus_fixed": check(l["mean_pp"], float(ref_l["frozen_vs_fixed_mean_pp"])),
        "l2b_fixed_pct_mean": check(l["fixed_pct_mean"], float(ref_l["fixed_pct_mean"])),
        "test_rows_count_1319": {"rows": int(len(test)), "pass": bool(len(test) == CUT_SIZE)},
        "spend": {"checks": checks, "failures": fails, "pass": bool(fails == 0)},
        "overlap_with_p0_zero": {"test_rows": t["overlap_with_p0"], "l2b": l["overlap_with_p0"],
                                 "pass": bool(t["overlap_with_p0"] == 0 and l["overlap_with_p0"] == 0)},
    }
    ok = all(g["pass"] for g in gates.values())
    write_json(f"{out}/gate.json", {"experiment": "L16a", "pass": bool(ok), "tables": len(tables), "gates": gates,
                                    "test_rows": t, "l2b_full": l})
    for name, g in gates.items():
        print(f"{name}: pass {g['pass']} {({k: v for k, v in g.items() if k != 'pass'})}", flush=True)
    print(f"test rows frozen minus fixed {t['mean_pp']:+.6f} pp (ref {ref_t['frozen_vs_fixed_mean_pp']:+.6f}); "
          f"L2b full {l['mean_pp']:+.6f} pp (ref {ref_l['frozen_vs_fixed_mean_pp']:+.6f}); gate {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def cuts_mode(out, start, stop):
    fit, l2b, _ = load_inputs()
    tables = subset_tables(fit)
    p0_q = set(fit["question"])
    for c in range(start, stop):
        idx = np.sort(np.random.RandomState(CUT_SEED0 + c).choice(len(l2b), size=CUT_SIZE, replace=False))
        frame = l2b.iloc[idx].reset_index(drop=True)
        res = evaluate(frame, tables)
        res.update(unit=c, seed=CUT_SEED0 + c, positions=[int(x) for x in idx],
                   overlap_with_p0=int(frame["question"].isin(p0_q).sum()))
        write_json(f"{out}/cuts/cut_{c:02d}.json", res)
        print(f"cut {c:02d}: frozen minus fixed {res['mean_pp']:+.4f} pp, n_sim {min(res['n_sim_per_split'])} to "
              f"{max(res['n_sim_per_split'])}, spend failures {res['spend_failures']}, overlap {res['overlap_with_p0']}", flush=True)
    return 0


def boot_mode(out, start, stop):
    fit, _, test = load_inputs()
    tables = subset_tables(fit)
    p0_q = set(fit["question"])
    draws = np.random.RandomState(BOOT_SEED).randint(0, len(test), size=(N_BOOT, len(test)))
    for b in range(start, stop):
        idx = np.sort(draws[b])
        frame = test.iloc[idx].reset_index(drop=True)
        res = evaluate(frame, tables)
        res.update(unit=b, seed=BOOT_SEED, positions=[int(x) for x in idx],
                   overlap_with_p0=int(frame["question"].isin(p0_q).sum()))
        write_json(f"{out}/boot/boot_{b:03d}.json", res)
        print(f"boot {b:03d}: frozen minus fixed {res['mean_pp']:+.4f} pp, distinct pids {res['distinct_pids']}, "
              f"spend failures {res['spend_failures']}", flush=True)
    return 0


def combine_mode(out):
    done = os.path.join(out, "done.flag")
    if os.path.exists(done):
        os.remove(done)
    gate = json.load(open(f"{out}/gate.json"))
    if not gate["pass"]:
        write_json(f"{out}/result.json", {"experiment": "L16a", "verdict": "GATE_FAILED", "gates": gate["gates"]})
        print("GATE FAILED: no L16a verdict written", flush=True)
        return 1
    cuts = [json.load(open(p)) for p in sorted(glob.glob(f"{out}/cuts/cut_*.json"))]
    boots = [json.load(open(p)) for p in sorted(glob.glob(f"{out}/boot/boot_*.json"))]
    if len(cuts) != N_CUTS or len(boots) != N_BOOT:
        print(f"INCOMPLETE: {len(cuts)} of {N_CUTS} cuts, {len(boots)} of {N_BOOT} boot draws", flush=True)
        return 1
    frames = cuts + boots
    spend_fail = int(sum(f["spend_failures"] for f in frames))
    overlap = int(sum(f["overlap_with_p0"] for f in frames))
    cut_rows_ok = all(c["rows"] == CUT_SIZE and c["distinct_pids"] == CUT_SIZE for c in cuts)
    if spend_fail or overlap or not cut_rows_ok:
        write_json(f"{out}/result.json", {"experiment": "L16a", "verdict": "GATE_FAILED", "spend_failures": spend_fail,
                                          "overlap_with_p0": overlap, "cut_rows_ok": cut_rows_ok})
        print(f"GATE FAILED: spend failures {spend_fail}, overlap {overlap}, cut rows ok {cut_rows_ok}", flush=True)
        return 1

    cm = np.array([c["mean_pp"] for c in cuts], float)
    bm = np.array([b["mean_pp"] for b in boots], float)
    test_val = float(gate["gates"]["test_rows_frozen_minus_fixed"]["mine"])
    p5, p95 = (float(x) for x in np.percentile(cm, [5, 95]))
    if test_val < p5:
        verdict = "KILL"
    elif test_val > p95:
        verdict = "UNREGISTERED_ABOVE_P95"
    else:
        verdict = "PASS"
    rp = f"{out}/critic/replay.json"
    replay = json.load(open(rp)) if os.path.exists(rp) else {"pass": None, "note": "replay not run"}
    if replay.get("pass") is False:
        verdict = "REPLAY_MISMATCH"
    bp5, bp50, bp95 = (float(x) for x in np.percentile(bm, [5, 50, 95]))
    result = {
        "experiment": "L16a",
        "verdict": verdict,
        "metric": "frozen minus plain fixed hit rate at 10% budget, pp (mean over 20 fit subsets, 8 split seeds, 8 tie seeds)",
        "pass_rule": "+1.22 pp (registered test rows) lies between the 5th and 95th percentiles of the 48 train-cut means",
        "kill_rule": "test value below the 5th percentile of the 48 train-cut means",
        "unregistered_rule": "test value above the 95th percentile: reported as an unregistered outcome, not a pass or a kill",
        "registered_test_rows": {
            "rows": int(gate["test_rows"]["rows"]), "frozen_minus_fixed_pp": test_val,
            "ref_L13a_pp": float(json.load(open(REF_TEST))["pools"]["L4b"]["frozen_vs_fixed_mean_pp"]),
            "fixed_pct_mean": gate["test_rows"]["fixed_pct_mean"], "frozen_pct_mean": gate["test_rows"]["frozen_pct_mean"],
            "n_sim_per_split": gate["test_rows"]["n_sim_per_split"],
            "bootstrap": {"draws": len(boots), "seed": BOOT_SEED, "p5": bp5, "p50": bp50, "p95": bp95,
                          "mean": float(bm.mean()), "sd": float(bm.std(ddof=1)), "min": float(bm.min()), "max": float(bm.max()),
                          "values_pp": bm.tolist()}},
        "train_cuts": {
            "n": len(cuts), "rows_per_cut": CUT_SIZE, "p5": p5, "p95": p95, "min": float(cm.min()), "max": float(cm.max()),
            "median": float(np.median(cm)), "mean": float(cm.mean()), "sd": float(cm.std(ddof=1)),
            "n_above_test_value": int((cm > test_val).sum()), "values_pp": cm.tolist(),
            "fixed_pct_mean": float(np.mean([c["fixed_pct_mean"] for c in cuts])),
            "frozen_pct_mean": float(np.mean([c["frozen_pct_mean"] for c in cuts])),
            "n_sim_per_split": cuts[0]["n_sim_per_split"], "cut_seeds": [c["seed"] for c in cuts]},
        "l2b_full_pool_pp": float(gate["l2b_full"]["mean_pp"]),
        "gates": gate["gates"], "spend_failures": spend_fail, "overlap_with_p0": overlap,
        "replay": {k: replay.get(k) for k in ("pass", "max_abs_diff_pp", "note", "checked")},
        "caveats": ["the 48 cuts overlap each other (each pair shares about 580 rows on average), so the 5th and 95th "
                    "percentiles describe the spread of this design, not a calibrated interval",
                    "bootstrap draws keep duplicate rows (row-level nonparametric bootstrap, an assumption not in the spec)",
                    "only the fixed and frozen arms are scored; the L13a refit arms do not enter frozen minus fixed"],
    }
    write_json(f"{out}/result.json", result)
    open(done, "w").close()
    print(f"L16a verdict {verdict}: test rows {test_val:+.4f} pp; 48 train cuts p5 {p5:+.4f} median {np.median(cm):+.4f} "
          f"p95 {p95:+.4f} min {cm.min():+.4f} max {cm.max():+.4f}; bootstrap p5 {bp5:+.4f} p95 {bp95:+.4f}; "
          f"replay {replay.get('pass')}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["gate", "cuts", "boot", "combine"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "gate":
        sys.exit(gate_mode(a.out))
    if a.mode == "cuts":
        sys.exit(cuts_mode(a.out, a.start, a.stop))
    if a.mode == "boot":
        sys.exit(boot_mode(a.out, a.start, a.stop))
    sys.exit(combine_mode(a.out))


if __name__ == "__main__":
    main()
