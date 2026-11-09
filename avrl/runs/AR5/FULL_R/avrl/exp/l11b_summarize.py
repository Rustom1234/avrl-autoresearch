"""L11b summary. Reads runs/L11b/oracle_<tag>.json, r2.json, gate_auc.json, gate/gate_check.json and sim_<tag>_<cheap|orc|orc_ng>_<cell100|cell30>_s<0..3>.json (8 policy seeds each); writes result.json and summary.md.
Rules (spec): PASS = oracle delta AUC >= 0.03 with bootstrap CI lower end >= 0.015 and above the permutation null on at least one rung, AND OOF ridge R^2 of the 16 PCs for n_steps < 0.2, gate passing.
KILL = oracle delta AUC < 0.01 on both rungs, in the logistic and in the HGB check. Anything else = INCONCLUSIVE.
Two readings are reported. 'literal' uses the full oracle set (n_steps, n_numbers_q, log_abs_gold; HGB check with the same set). 'leakfree' drops log_abs_gold, because next to the rollout's own log_abs_ans it lets a model detect answer == gold,
i.e. read the label (reported: share of rows with equal magnitudes and the label rate there). The headline verdict is PASS only if the leak-free reading also passes, KILL if the literal reading kills (a leak can only inflate the delta, so a literal kill is also a leak-free kill)."""
import json, os, numpy as np
D = "runs/L11b"; TAGS = ("3B_q_corrupted", "7B_q_corrupted"); BUD = ("0.1", "0.25", "0.5"); LAYS = ("cell100", "cell30"); FS = ("cheap", "orc", "orc_ng")
J = lambda p: json.load(open(os.path.join(D, p)))
gate_auc, gate_sim, r2 = J("gate_auc.json"), J("gate/gate_check.json"), J("r2.json")
gate_ok = all(gate_auc.values()) and gate_sim["passed"]
rungs = {t: J(f"oracle_{t}.json") for t in TAGS}

def passes(d): return bool(d["mean"] >= 0.03 and d["ci_lo"] >= 0.015 and d["mean"] > d["null_max"])
r2_pc = r2["n_steps"]["ridge_pcs"]["mean"]
res = {"gate": {"cheap_auc_reproduced": gate_auc, "sim_regression": gate_sim, "passed": bool(gate_ok)}, "r2_n_steps": {k: v["mean"] for k, v in r2["n_steps"].items()},
       "r2_n_numbers_q": {k: v["mean"] for k, v in r2["n_numbers_q"].items()}, "r2_log_abs_gold": {k: v["mean"] for k, v in r2["log_abs_gold"].items()}, "rungs": {}}
for t, o in rungs.items():
    d = o["delta"]; slim = lambda k: {x: d[k][x] for x in ("mean", "ci_lo", "ci_hi") if x in d[k]} | ({"null_mean": d[k]["null_mean"], "null_sd": d[k]["null_sd"], "null_max": d[k]["null_max"]} if "null_mean" in d[k] else {})
    res["rungs"][t] = {"rows": o["rows"], "pids": o["pids"], "positives": o["positives"], "rate": o["rate"], "auc": {k: v["mean"] for k, v in o["auc"].items()},
                       "delta": {k: slim(k) for k in d}, "leak_diagnostic": o["leak_diagnostic"], "repro": o["repro"],
                       "pass_literal": passes(d["orc"]), "pass_leakfree": passes(d["orc_ng"]), "kill_literal": bool(d["orc"]["mean"] < 0.01 and d["hgb_orc"]["mean"] < 0.01),
                       "kill_leakfree": bool(d["orc_ng"]["mean"] < 0.01 and d["hgb_orc_ng"]["mean"] < 0.01),
                       "hgb_leakfree_meets_pass_numbers": bool(d["hgb_orc_ng"]["mean"] >= 0.03 and d["hgb_orc_ng"]["ci_lo"] >= 0.015 and d["hgb_orc_ng"]["mean"] > d["hgb_orc_ng"]["null_max"]),
                       "abs_auc_hgb_orc_ng_vs_logit_orc_ng": [o["auc"]["hgb_orc_ng"]["mean"], o["auc"]["orc_ng"]["mean"]]}
R = res["rungs"]; r2_ok = bool(r2_pc < 0.2)
pass_lit = gate_ok and r2_ok and any(R[t]["pass_literal"] for t in TAGS); pass_lf = gate_ok and r2_ok and any(R[t]["pass_leakfree"] for t in TAGS)
kill_lit = gate_ok and all(R[t]["kill_literal"] for t in TAGS); kill_lf = gate_ok and all(R[t]["kill_leakfree"] for t in TAGS)
res["notes"] = ["Pass/kill rule is applied to the logistic delta (primary metric); HGB is the nonlinear check. The full-oracle HGB delta (~+0.25) is label leakage: log_abs_gold next to log_abs_ans reveals answer == gold (label rate ~0 when the magnitudes match, 1.0 otherwise).",
    "hgb_leakfree_meets_pass_numbers marks rungs where the HGB delta without gold magnitude alone reaches the numeric pass thresholds; HGB deltas are taken against a weaker HGB baseline (abs_auc_hgb_orc_ng_vs_logit_orc_ng shows the logistic arm is still higher in absolute AUC).",
    "Secondary replay: the orc arm (with gold magnitude) is leak-inflated; read the orc_ng arm."]
res["pc_r2_n_steps"] = r2_pc; res["pc_r2_below_0.2"] = r2_ok
res["verdict_literal"] = "VOID" if not gate_ok else "PASS" if pass_lit else "KILL" if kill_lit else "INCONCLUSIVE"
res["verdict_leakfree"] = "VOID" if not gate_ok else "PASS" if pass_lf else "KILL" if kill_lf else "INCONCLUSIVE"
res["verdict"] = "VOID" if not gate_ok else "PASS" if (pass_lit and pass_lf) else "KILL" if kill_lit else "INCONCLUSIVE"

# secondary: allocation replay, pooled over 4 splits x 8 policy seeds
sim = {}
for t in TAGS:
    for lay in LAYS:
        for fs in FS:
            runs = [J(f"sim_{t}_{fs}_{lay}_s{sp}.json") for sp in range(4)]
            for b in BUD:
                row = {}
                for pol in ("uniform", "static", "greedy_online", "ts_online", "oracle"):
                    v = np.array([x for r in runs for x in r[f"budget_{b}"][pol]["per_seed"]], float); row[pol] = {"mean": float(v.mean()), "sd": float(v.std()), "n": int(len(v))}
                sim.setdefault(t, {}).setdefault(lay, {}).setdefault(b, {})[fs] = row
for t in TAGS:
    for lay in LAYS:
        for b in BUD:
            c = sim[t][lay][b]
            for fs in ("orc", "orc_ng"):
                for pol in ("greedy_online", "ts_online", "static"):
                    a, bb = c[fs][pol], c["cheap"][pol]; c[fs][pol]["ratio_vs_cheap"] = a["mean"] / bb["mean"]; c[fs][pol]["gap_z_vs_cheap"] = (a["mean"] - bb["mean"]) / float(np.hypot(a["sd"], bb["sd"]))
res["sim_pooled_4splits_x_8seeds"] = sim
json.dump(res, open(os.path.join(D, "result.json"), "w"), indent=1)

L = ["# L11b summary", "", f"Gate: cheap_r OOF AUC reproduced {gate_auc}; no-oracle replay vs L1a: {gate_sim['differences']} per-seed differences in {gate_sim['policy_budget_cells_compared']} policy x budget cells ({gate_sim['files_compared']} files). Passed: {gate_ok}.", "",
     f"Verdict (headline): **{res['verdict']}**; literal reading {res['verdict_literal']}; leak-free reading {res['verdict_leakfree']}. PC OOF R^2 for n_steps = {r2_pc:.4f} (< 0.2: {r2_ok}).", "",
     "## OOF AUC and delta over cheap_r (corrupted label, accept rule q; pid-bootstrap 95% CI, 200 resamples; permutation null: 20 donor-pid swaps, CV seed 0)", ""]
for t, o in R.items():
    a = o["auc"]; L += [f"### {t}: rows {o['rows']}, pids {o['pids']}, positives {o['positives']} ({o['rate']:.3f})", "",
        f"AUC: cheap_r {a['cheap']:.4f} | +orc {a['orc']:.4f} | +orc_ng {a['orc_ng']:.4f} | +PCs {a['pc']:.4f} | +PCs+orc {a['pc_orc']:.4f} | HGB cheap_r {a['hgb_cheap']:.4f} | HGB +orc {a['hgb_orc']:.4f} | HGB +orc_ng {a['hgb_orc_ng']:.4f}", "",
        "| contrast | delta AUC | 95% CI | null mean +- sd (max) |", "|---|---|---|---|"]
    for k, lab in (("orc", "cheap_r + 3 oracle"), ("orc_ng", "cheap_r + n_steps, n_numbers_q"), ("steps", "cheap_r + n_steps"), ("nnum", "cheap_r + n_numbers_q"), ("gold", "cheap_r + log_abs_gold"), ("pc", "cheap_r + 16 PCs"),
                   ("pc_orc_vs_pc", "PCs + 3 oracle over PCs"), ("hgb_orc", "HGB: + 3 oracle (leak-inflated)"), ("hgb_orc_ng", "HGB: + n_steps, n_numbers_q")):
        d = o["delta"][k]; nl = f"{d['null_mean']:+.4f} +- {d['null_sd']:.4f} ({d['null_max']:+.4f})" if "null_mean" in d else "-"
        L.append(f"| {lab} | {d['mean']:+.4f} | [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] | {nl} |")
    lk = o["leak_diagnostic"]; L += ["", f"Leak diagnostic: log_abs_ans == log_abs_gold in {lk['rows_log_abs_ans_equals_log_abs_gold']} rows ({lk['share']:.3f}); label rate {lk['label_rate_when_equal']} there vs {lk['label_rate_when_not_equal']:.3f} elsewhere.",
                                  f"Pass literal {o['pass_literal']}, pass leak-free {o['pass_leakfree']}, kill literal {o['kill_literal']}, kill leak-free {o['kill_leakfree']}.", ""]
L += ["## OOF R^2 (3000 prompts, pid folds, mean of 3 CV seeds)", "", "| target (sd) | ridge 16 PCs | ridge log_qlen | ridge PCs+qlen | HGB PCs | HGB PCs+qlen | TE cell30 | TE cell100 | TE cell300 |", "|---|---|---|---|---|---|---|---|---|"]
for tn in ("n_steps", "n_numbers_q", "log_abs_gold"):
    L.append(f"| {tn} ({r2['target_sd'][tn]:.2f}) | " + " | ".join(f"{r2[tn][k]['mean']:+.4f}" for k in ("ridge_pcs", "ridge_qlen", "ridge_pcs_qlen", "hgb_pcs", "hgb_pcs_qlen", "te_cell30", "te_cell100", "te_cell300")) + " |")
L += ["", "## Secondary (descriptive): errors caught, mean over 4 splits x 8 policy seeds; ratio = with oracle covariates / cheap_r alone, z = gap in pooled SD", ""]
for t in TAGS:
    for lay in LAYS:
        L += [f"### {t}, layout {lay}", "", "| budget | policy | uniform | cheap_r | +orc (ratio, z) | +orc_ng (ratio, z) |", "|---|---|---|---|---|---|"]
        for b in BUD:
            c = sim[t][lay][b]
            for pol in ("greedy_online", "ts_online"):
                L.append(f"| {float(b):.0%} | {pol} | {c['cheap']['uniform']['mean']:.1f} | {c['cheap'][pol]['mean']:.1f} | {c['orc'][pol]['mean']:.1f} ({c['orc'][pol]['ratio_vs_cheap']:.3f}, {c['orc'][pol]['gap_z_vs_cheap']:+.1f}) | {c['orc_ng'][pol]['mean']:.1f} ({c['orc_ng'][pol]['ratio_vs_cheap']:.3f}, {c['orc_ng'][pol]['gap_z_vs_cheap']:+.1f}) |")
        L.append("")
open(os.path.join(D, "summary.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L[:8])); print("verdict", res["verdict"], "| literal", res["verdict_literal"], "| leakfree", res["verdict_leakfree"], "| PC R2 n_steps", round(r2_pc, 4))
