"""L10a finalizer (CPU, shell). Registered definitions and bins: docstring of exp/l10a_boot_xfit_fixed.py (read it first). Reads the 24 XFIT=2 JSONs
<run_dir>/<pool>/<proxy>/xfit_f<f>.json (+ .log), the reproduction gate <run_dir>/gate_repro/gate.json, checks every gate, forms the registered statistic and the verdict, writes
<run_dir>/result.json and <run_dir>/table.txt. Exit 0 for PASS / KILL / INCONCLUSIVE (a result), exit 4 for FAILED (not a verdict).
Usage: l10a_finalize.py --run_dir <dir> [--n_boot 200] [--smoke]   (--smoke = synthetic null labels, reproduction gate not required, output labelled SMOKE, never a verdict)."""
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--run_dir", required=True); ap.add_argument("--n_boot", type=int, default=200); ap.add_argument("--smoke", action="store_true"); args = ap.parse_args()
R, NB = args.run_dir, args.n_boot
POOLS = {"L9c-weak-policy-tiebreak": "0.5B", "L8c-A-fresh-tiebreak": "1.5B"}; PROX = ("q3_4b", "rm"); FR = ("0.1", "0.15", "0.2", "0.25", "0.3", "0.4")
FROZEN = {"q3_4b": "agree_acc_sdiff", "rm": "agree_acc_soth"}; FIXED7 = ["agree_thr", "judge_thr", "blend50", "agree_acc", "blend50_acc", "agree_acc_soth", "agree_acc_sdiff"]
CELLS = [(pl, px) for pl in POOLS for px in PROX]; cname = lambda c: f"{POOLS[c[0]]} {c[1]}"
fails, J = [], {}

# ---------------- gates ----------------
for c in CELLS:
    for f in FR:
        fn = f"{R}/{c[0]}/{c[1]}/xfit_f{f}.json"; lg = fn[:-5] + ".log"; tag = f"{cname(c)} f{f}"
        if os.path.exists(lg):
            txt = open(lg).read()
            if "GATE FAILED" in txt or "Traceback" in txt: fails.append(f"{tag}: GATE FAILED or Traceback in {lg}")
        else: fails.append(f"{tag}: missing log {lg}")
        if not os.path.exists(fn): fails.append(f"{tag}: missing {fn}"); continue
        j = json.load(open(fn)); J[c, f] = j
        if j.get("mode") != "xfit2": fails.append(f"{tag}: mode {j.get('mode')}")
        if abs(j["budget_frac"] - float(f)) > 1e-12: fails.append(f"{tag}: budget_frac {j['budget_frac']}")
        if j["n_boot"] != NB or j["seeds_per_resample"] != 4 or j["n_pool"] != 2000: fails.append(f"{tag}: n_boot {j['n_boot']} seeds {j['seeds_per_resample']} n_pool {j['n_pool']}")
        if j["frozen"] != FROZEN[c[1]]: fails.append(f"{tag}: frozen {j['frozen']}")
        if j["fixed7"] != FIXED7: fails.append(f"{tag}: fixed7 {j['fixed7']}")
        if not all(j["sanity"].values()): fails.append(f"{tag}: sanity {j['sanity']}")
        for h, hv in enumerate(j["halves"]):
            fh, sel = hv["full_half"], hv["selection"]; mc = sel["mean_caught"]; ch = max(FIXED7, key=lambda p: mc[p])  # independent recomputation of CH, first listed wins ties
            if hv["held_out_parity"] != h or ch != sel["ch"]: fails.append(f"{tag} half {h}: CH {sel['ch']} stored, {ch} recomputed")
            if fh["budget"] != int(float(f) * 1000) or sel["budget"] != int(float(f) * 1000) or fh["n"] != 1000 or sel["n"] != 1000: fails.append(f"{tag} half {h}: budget/n {fh['budget']} {sel['budget']} {fh['n']} {sel['n']}")
            if fh["greedy_warm7_norefit"] != fh[j["frozen"]]: fails.append(f"{tag} half {h}: norefit != frozen on the full half")
            if len(hv["resamples"]) != NB: fails.append(f"{tag} half {h}: {len(hv['resamples'])} resamples")
            if any(r["greedy_warm7_norefit"] != r[j["frozen"]] for r in hv["resamples"][:20]): fails.append(f"{tag} half {h}: norefit != frozen in the first 20 resamples")
            if fh["in_sample_best"] != max(FIXED7, key=lambda p: np.mean(fh[p])): fails.append(f"{tag} half {h}: in_sample_best mismatch")
for h in (0, 1):  # the resample index stream must be identical across all 24 jobs (paired CI across cells and budgets)
    shas = {j["halves"][h]["idx_sha256"] for j in J.values()}
    if len(shas) != 1: fails.append(f"half {h}: {len(shas)} distinct resample index streams across jobs")
gate = None
if not args.smoke:
    gp = f"{R}/gate_repro/gate.json"
    if not os.path.exists(gp): fails.append("reproduction gate: gate.json missing")
    else:
        gate = json.load(open(gp))
        if not gate["pass"]: fails.append("reproduction gate: mismatch " + json.dumps({k: v for k, v in gate["cells"].items() if not v["ok"]}))
if fails:
    json.dump({"status": "FAILED", "smoke": args.smoke, "failures": fails}, open(f"{R}/result.json", "w"), indent=1); print("FAILED", *fails, sep="\n  "); sys.exit(4)

# ---------------- statistic ----------------
def arm(j, h, w):  # arm name per half: G learner, K cross-fitted best fixed (CH), A agree_acc, F frozen warm start, I in-sample best of seven on the held half, U uniform, O oracle
    hv = j["halves"][h]; return {"G": "greedy_blend7", "K": hv["selection"]["ch"], "A": "agree_acc", "F": j["frozen"], "I": hv["full_half"]["in_sample_best"], "U": "uniform", "O": "oracle"}[w]
def plug(j, w): return float(sum(np.mean(j["halves"][h]["full_half"][arm(j, h, w)]) for h in (0, 1)))  # sum over halves of the 8-seed full-half mean
def boot(j, w): return np.array([sum(np.mean(j["halves"][h]["resamples"][i][arm(j, h, w)]) for h in (0, 1)) for i in range(NB)])  # sum over halves of the 4-seed mean, per paired resample
pct = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]

def contrast(num, den):
    out = {"ratio": {}, "cell": {}}; cb, cp = {}, {}
    for c in CELLS:
        lp, lb = [], []; out["ratio"][cname(c)] = {}
        for f in FR:
            j = J[c, f]; pn, pd_ = plug(j, num), plug(j, den); bn, bd = boot(j, num), boot(j, den); assert pd_ > 0 and (bd > 0).all() and (bn > 0).all()
            lo, hi = pct(bn / bd); out["ratio"][cname(c)][f] = {"plug": pn / pd_, "lo": lo, "hi": hi, "num_plug": pn, "den_plug": pd_}; lp.append(np.log(pn / pd_)); lb.append(np.log(bn / bd))
        cp[c] = float(np.mean(lp)); cb[c] = np.mean(lb, axis=0); lo, hi = pct(cb[c])
        out["cell"][cname(c)] = {"plug": cp[c], "boot_mean": float(cb[c].mean()), "lo": lo, "hi": hi, "sd": float(cb[c].std(ddof=1))}
    pb = np.mean([cb[c] for c in CELLS], axis=0); pp = float(np.mean([cp[c] for c in CELLS])); lo, hi = pct(pb)
    out["pooled"] = {"plug": pp, "boot_mean": float(pb.mean()), "lo": lo, "hi": hi, "sd": float(pb.std(ddof=1)), "half_width": (hi - lo) / 2, "n_cells_plug_positive": int(sum(cp[c] > 0 for c in CELLS))}
    return out

S = {"G_over_K_crossfit": contrast("G", "K"), "G_over_agree_acc": contrast("G", "A"), "G_over_frozen": contrast("G", "F"), "G_over_insample_best": contrast("G", "I"), "K_over_agree_acc": contrast("K", "A"), "I_over_K": contrast("I", "K")}
P = S["G_over_K_crossfit"]["pooled"]; PASS = P["plug"] >= 0.02 and P["lo"] > 0 and P["n_cells_plug_positive"] >= 3; KILL = P["hi"] < 0.01
OUT = not (P["lo"] <= P["plug"] <= P["hi"]); DISC = (PASS and KILL) or OUT
if DISC: verdict = "INCONCLUSIVE, DISCORDANT"
elif PASS: verdict = "PASS"
elif KILL: verdict = "KILL"
elif P["half_width"] > 0.03: verdict = "INCONCLUSIVE, POWER-LIMITED"
else: verdict = "INCONCLUSIVE"
if args.smoke: verdict = "SMOKE (synthetic null labels), would read: " + verdict

# ---------------- context tables ----------------
ab = {"agree_thr": "ath", "judge_thr": "jth", "blend50": "b50", "agree_acc": "aa", "blend50_acc": "b50a", "agree_acc_soth": "soth", "agree_acc_sdiff": "sdif"}
chtab, curse, chcount = {}, {}, {p: 0 for p in FIXED7}
for c in CELLS:
    chtab[cname(c)] = {}; curse[cname(c)] = {}
    for f in FR:
        j = J[c, f]; chs = [arm(j, h, "K") for h in (0, 1)]; chtab[cname(c)][f] = {"half0": chs[0], "half1": chs[1], "in_sample_best_half0": arm(j, 0, "I"), "in_sample_best_half1": arm(j, 1, "I")}
        for p in chs: chcount[p] += 1
        gap = plug(j, "I") - plug(j, "K"); curse[cname(c)][f] = {"gap_caught": gap, "share_of_in_sample_best": gap / plug(j, "I")}
gaps = [v["gap_caught"] for t in curse.values() for v in t.values()]; shares = [v["share_of_in_sample_best"] for t in curse.values() for v in t.values()]
flag_cell = cname(("L8c-A-fresh-tiebreak", "q3_4b")); n_total = 4 * 6 * 2
res = {"status": "SMOKE" if args.smoke else "OK", "verdict": verdict, "bins": {"PASS": bool(PASS), "KILL": bool(KILL), "point_outside_ci": bool(OUT), "power_flag_halfwidth_gt_0.03": bool(P["half_width"] > 0.03)},
       "metric_pooled_C_G_over_CH": P, "pooled_ratio_equivalent_exp_C": {"plug": float(np.exp(P["plug"])), "lo": float(np.exp(P["lo"])), "hi": float(np.exp(P["hi"]))},
       "per_cell_C_G_over_CH": S["G_over_K_crossfit"]["cell"], "flag_L8c-A_1.5B_q3_4b_cell_C": S["G_over_K_crossfit"]["cell"][flag_cell],
       "contrasts": S, "ch_identity": chtab, "ch_counts_over_%d_cell_budget_half_choices" % n_total: chcount,
       "winners_curse_gap_in_sample_best_minus_CH": {"mean_gap_caught": float(np.mean(gaps)), "mean_share": float(np.mean(shares)), "per_cell_budget": curse},
       "gates": {"reproduction_gate": None if gate is None else gate["pass"], "n_jobs": len(J), "n_boot": NB, "seeds_per_resample": 4, "n_halves": 2, "all_norefit_and_ch_checks_passed": True, "resample_index_stream_identical_across_jobs": True},
       "sample_sizes": {"groups_per_pool": 2000, "groups_per_half": 1000, "cells": 4, "budgets": list(FR), "resamples": NB}}
json.dump(res, open(f"{R}/result.json", "w"), indent=1)

L = [f"L10a-xfit-fixed-comparator  verdict: {verdict}", f"gates: reproduction {'n/a (smoke)' if gate is None else gate['pass']}, 24 jobs, no GATE FAILED / Traceback, CH recomputed, norefit gates, identical resample index streams: all pass", "",
     f"Pooled C = mean over 4 cells and 6 budgets of log(greedy_blend7 / cross-fitted best fixed rule CH); 95% percentile CI over {NB} paired resamples",
     f"  plug-in {P['plug']:+.4f} (ratio {np.exp(P['plug']):.3f})  boot mean {P['boot_mean']:+.4f}  CI [{P['lo']:+.4f}, {P['hi']:+.4f}] (ratio [{np.exp(P['lo']):.3f}, {np.exp(P['hi']):.3f}])  SD {P['sd']:.4f}  half-width {P['half_width']:.4f}  cells with positive plug-in C: {P['n_cells_plug_positive']}/4",
     f"  PASS {PASS} (needs plug-in >= +0.02, CI lower > 0, >= 3 of 4 positive cells)   KILL {KILL} (needs CI upper < +0.01)   POWER-LIMITED flag (half-width > 0.03) {P['half_width'] > 0.03}", "",
     "Contrasts (pooled C plug-in [95% CI]; per-cell C plug-in):"]
for k, v in S.items():
    p_ = v["pooled"]; L.append(f"  {k:24s} {p_['plug']:+.4f} [{p_['lo']:+.4f}, {p_['hi']:+.4f}] SD {p_['sd']:.4f} pos {p_['n_cells_plug_positive']}/4 | " + "  ".join(f"{n} {d['plug']:+.3f}" for n, d in v["cell"].items()))
L += ["", "Per cell C (G / CH), plug-in [95% CI]:"] + [f"  {n:10s} {d['plug']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] SD {d['sd']:.4f}" for n, d in S["G_over_K_crossfit"]["cell"].items()]
L += ["", "Ratio G / CH per cell and budget, plug-in [95% percentile CI]:", "  cell        " + "".join(f"{'f=' + f:>26s}" for f in FR)]
for n, t in S["G_over_K_crossfit"]["ratio"].items(): L.append(f"  {n:10s}  " + "".join(f"{t[f]['plug']:>9.3f} [{t[f]['lo']:.3f},{t[f]['hi']:.3f}]" for f in FR))
L += ["", "Ratio G / agree_acc per cell and budget, plug-in:"] + [f"  {n:10s}  " + "".join(f"{t[f]['plug']:>26.3f}" for f in FR) for n, t in S["G_over_agree_acc"]["ratio"].items()]
L += ["", "Ratio G / in-sample best of seven (old convention) per cell and budget, plug-in:"] + [f"  {n:10s}  " + "".join(f"{t[f]['plug']:>26.3f}" for f in FR) for n, t in S["G_over_insample_best"]["ratio"].items()]
L += ["", "CH identity (half 0 / half 1; rule chosen on the OTHER half), and in-sample best on the held half in parentheses:"]
for n, t in chtab.items(): L.append(f"  {n:10s}  " + "  ".join(f"f{f}: {ab[t[f]['half0']]}/{ab[t[f]['half1']]} ({ab[t[f]['in_sample_best_half0']]}/{ab[t[f]['in_sample_best_half1']]})" for f in FR))
L += ["", f"CH counts over {n_total} (cell, budget, half) choices: " + ", ".join(f"{ab[p]} {chcount[p]}" for p in FIXED7), f"Winner's curse removed (in-sample best minus CH on the held halves, plug-in): mean {np.mean(gaps):.2f} caught errors, mean share {100 * np.mean(shares):.2f}% of the in-sample best",
      f"Flag (not a verdict condition), {flag_cell} cell C: {S['G_over_K_crossfit']['cell'][flag_cell]['plug']:+.4f} [{S['G_over_K_crossfit']['cell'][flag_cell]['lo']:+.4f}, {S['G_over_K_crossfit']['cell'][flag_cell]['hi']:+.4f}]"]
open(f"{R}/table.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
