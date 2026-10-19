"""L10a independent check (CPU). Written separately from exp/l10a_finalize.py and exp/l10a_boot_xfit_fixed.py (no import of either, different data layout: plain dict walks, one cell at a time).
(1) CH selection: for every (cell, budget, half) re-take the argmax of the other-half 8-seed mean caught over the seven rules in the listed order (first wins ties) from the saved selection.mean_caught and compare with the stored ch; count exact ties at the top. The selection half is never scored, so only the stored table is available for this step.
(2) Resample indices: regenerate default_rng(12345).integers(0, 1000, 1000) for each half from the raw feats_J.parquet and compare the number of judge errors in the resample with the stored n_err of all 200 resamples of every cell (this validates the same-index claim from the raw labels).
(3) One cell (L8c-A 1.5B Qwen3-4B, all six budgets) and the pooled statistic: ratio greedy_blend7 / CH per budget from the saved arrays, log-mean over budgets, plug-in and 200-resample percentile CI, then the pooled C over the four cells; compared with result.json to 1e-9.
(4) Context only: the plug-in's rank inside the bootstrap distribution, the basic CI (never used for the verdict), the CH-selection-free 'frozen' contrast.
Usage: l10a_critic_check.py <run_dir>; writes <run_dir>/_critic/check.txt"""
import json, math, os, sys
import numpy as np, pandas as pd

R = sys.argv[1]; A = "/homes/rdubash/avrl_live/FULL/avrl"; res = json.load(open(f"{R}/result.json")); out = []
def say(*a): s = " ".join(str(x) for x in a); out.append(s); print(s, flush=True)
POOLS = ("L9c-weak-policy-tiebreak", "L8c-A-fresh-tiebreak"); PROX = ("q3_4b", "rm"); FR = ("0.1", "0.15", "0.2", "0.25", "0.3", "0.4"); F7 = ["agree_thr", "judge_thr", "blend50", "agree_acc", "blend50_acc", "agree_acc_soth", "agree_acc_sdiff"]
ok = True
def load(pl, px, f): return json.load(open(f"{R}/{pl}/{px}/xfit_f{f}.json"))

# (1) CH selection from the stored table
bad = 0; nties = 0
for pl in POOLS:
    for px in PROX:
        for f in FR:
            for hv in load(pl, px, f)["halves"]:
                mc = [hv["selection"]["mean_caught"][r] for r in F7]; top = max(mc); first = F7[mc.index(top)]; bad += first != hv["selection"]["ch"]; nties += mc.count(top) > 1
say(f"(1) CH recomputed for 4 cells x 6 budgets x 2 halves = 48 choices, mismatches {bad}, choices with an exact tie at the top {nties}"); ok &= bad == 0

# (2) resample indices from raw labels
badn = 0; chk = 0
for pl in POOLS:
    for px in PROX:
        d = pd.read_parquet(f"{A}/runs/{pl}/pool/{px}/feats_J.parquet"); assert (d.pid.values == np.arange(len(d))).all() and len(d) == 2000
        for h in (0, 1):
            lab = d.judge_error.values[d.pid.values % 2 == h].astype(int); g = np.random.default_rng(12345); ne = [int(lab[g.integers(0, len(lab), len(lab))].sum()) for _ in range(200)]
            for f in FR:
                rows = load(pl, px, f)["halves"][h]["resamples"]; badn += sum(r["n_err"] != e for r, e in zip(rows, ne)); chk += len(rows)
say(f"(2) n_err of {chk} stored resamples vs regenerated default_rng(12345) resamples of the raw labels: mismatches {badn}"); ok &= badn == 0

# (3) one cell and the pooled statistic with separately written code
def cell_stats(pl, px):
    lp, lb = [], []
    for f in FR:
        j = load(pl, px, f); gp = kp = 0.0; gb = [0.0] * 200; kb = [0.0] * 200
        for h in (0, 1):
            hv = j["halves"][h]; ch = hv["selection"]["ch"]; gp += sum(hv["full_half"]["greedy_blend7"]) / 8; kp += sum(hv["full_half"][ch]) / 8
            for i, r in enumerate(hv["resamples"]): gb[i] += sum(r["greedy_blend7"]) / 4; kb[i] += sum(r[ch]) / 4
        lp.append(math.log(gp / kp)); lb.append([math.log(a / b) for a, b in zip(gb, kb)])
    return sum(lp) / 6, [sum(lb[k][i] for k in range(6)) / 6 for i in range(200)]
cells = {(pl, px): cell_stats(pl, px) for pl in POOLS for px in PROX}
pc = lambda x: tuple(float(np.percentile(x, q)) for q in (2.5, 97.5))
flag = cells[("L8c-A-fresh-tiebreak", "q3_4b")]; lo, hi = pc(flag[1]); jf = res["flag_L8c-A_1.5B_q3_4b_cell_C"]
say(f"(3) flag cell L8c-A 1.5B q3_4b: C plug-in {flag[0]:+.6f} CI [{lo:+.6f}, {hi:+.6f}] vs result.json {jf['plug']:+.6f} [{jf['lo']:+.6f}, {jf['hi']:+.6f}]"); ok &= max(abs(flag[0] - jf["plug"]), abs(lo - jf["lo"]), abs(hi - jf["hi"])) < 1e-9
pp = sum(v[0] for v in cells.values()) / 4; pb = [sum(v[1][i] for v in cells.values()) / 4 for i in range(200)]; plo, phi = pc(pb); P = res["metric_pooled_C_G_over_CH"]
say(f"    pooled C plug-in {pp:+.6f} CI [{plo:+.6f}, {phi:+.6f}] SD {np.std(pb, ddof=1):.6f} vs result.json {P['plug']:+.6f} [{P['lo']:+.6f}, {P['hi']:+.6f}] SD {P['sd']:.6f}"); ok &= max(abs(pp - P["plug"]), abs(plo - P["lo"]), abs(phi - P["hi"]), abs(np.std(pb, ddof=1) - P["sd"])) < 1e-9
say("    per-cell plug-in C:", {f"{k[0][:3]}/{k[1]}": round(v[0], 4) for k, v in cells.items()}, "positive cells:", sum(v[0] > 0 for v in cells.values()))

# (4) context
rank = float(np.mean(np.array(pb) < pp)); say(f"(4) plug-in {pp:+.4f} sits at the {100 * rank:.0f}th percentile of the bootstrap distribution (bootstrap mean {np.mean(pb):+.4f}, bias {np.mean(pb) - pp:+.4f}); basic CI (not the verdict) [{2 * pp - phi:+.4f}, {2 * pp - plo:+.4f}]")
bins = res["bins"]; say(f"    bins as stored: PASS {bins['PASS']}, KILL {bins['KILL']}, verdict {res['verdict']}; re-derived: PASS {pp >= 0.02 and plo > 0 and sum(v[0] > 0 for v in cells.values()) >= 3}, KILL {phi < 0.01}, half-width {(phi - plo) / 2:.4f}")
ok &= (pp >= 0.02 and plo > 0 and sum(v[0] > 0 for v in cells.values()) >= 3) == bins["PASS"] and (phi < 0.01) == bins["KILL"]
say("CRITIC CHECK", "PASS" if ok else "FAILED"); os.makedirs(f"{R}/_critic", exist_ok=True); open(f"{R}/_critic/check.txt", "w").write("\n".join(out) + "\n"); sys.exit(0 if ok else 1)
