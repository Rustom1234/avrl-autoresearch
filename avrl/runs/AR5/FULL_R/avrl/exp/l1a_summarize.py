"""L1a summary (CPU). Usage: l1a_summarize.py <runs/L1a>. Writes result.json and summary.md, prints the markdown.

Reads build.json, structure_<tag>.json, gate/gate_check.json and sim_<tag>_<cheap|pc>_<cell100|cell30|pid>_s<split>.json (8 policy seeds each).
Pooled ratio = mean over all (splits x 8 seeds) runs of arm A / same of arm B; pooled gap z = (mean A - mean B) / sqrt(sd_A^2 + sd_B^2) with the SD over all runs (so split spread is inside the SD);
per-split ratio and z use the 8 policy seeds of that split (np.std, ddof 0, the E04 convention). 'best label-free threshold' = per (split, budget) max over the mean of
thr_judge_low / thr_boundary / thr_tokens_long / thr_trunc_first; 'best recal_charged' = per (split, budget) max over recal_charged_100/200/400 (the same max-of-family convention, which if anything favours the fixed rule).

Pre-registered rules (L1a spec; primary = 3B, confirmation = 7B-AWQ, label = corrupted, accept rule q = matched accept rate, bucket = cell100):
PASS (gate passes and on at least one rung): (a) delta AUC +16 PCs >= 0.05 with bootstrap CI lower end >= 0.03, OR OOF cell-encoding delta AUC >= 0.02 with CI lower end >= 0.01 at cell30/100/300;
  AND (b) at 25% budget ts_online >= 1.05 x greedy_online (same features) with the gap above 2 pooled SD in at least 3 of 4 split seeds, AND ts_online >= 1.10 x the best label-free threshold in those seeds.
  (b) is evaluated for each feature set (cheap_r, cheap_r + PCs); passing on either counts (generous to PASS, declared before the data).
KILL (gate passes, on BOTH rungs): (a') delta AUC +16 PCs < 0.02 (point estimate) and its CI upper end < 0.03 AND OOF cell-encoding delta AUC < 0.01 at all three cell counts; AND (b') at 25% pooled over split seeds
  ts_online <= 1.03 x greedy_online on BOTH feature sets and greedy_online(cheap_r + PCs) <= 1.03 x greedy_online(cheap_r) (strict for KILL).
LITERAL BACKLOG RULE: kill if delta AUC +PCs < 0.01 and cell delta < 0.01 on both rungs and learned <= 1.05 x best label-free threshold at 25% (learned = the best of ts_online / greedy_online on either feature set, pooled);
  backlog pass: (delta AUC >= 0.05 or cell delta >= 0.02) and learned >= 1.10 x best label-free threshold with the pooled gap above 2 SD.
VOID if the regression gate (0 per-seed differences against runs/E04) or the build gate fails."""
import sys, os, json
import numpy as np

rd = sys.argv[1]
RUNG = {"3B": "3B_q", "7B": "7B_q"}; LABEL = "corrupted"
TAGS = ["3B_q_corrupted", "3B_q_judgewrong", "7B_q_corrupted", "7B_q_judgewrong", "7B_p05_corrupted", "7B_p05_judgewrong"]
FS = ["cheap", "pc"]; BUD = ["0.1", "0.25", "0.5"]; THR = ["thr_judge_low", "thr_boundary", "thr_tokens_long", "thr_trunc_first"]; REC = ["recal_charged_100", "recal_charged_200", "recal_charged_400"]
CELLS = ["cell30", "cell100", "cell300"]

def jl(p): return json.load(open(os.path.join(rd, p)))
def sims(tag, fs, lay, splits): return [jl(f"sim_{tag}_{fs}_{lay}_s{s}.json") for s in splits]
def A(ds, b, p): return np.array([d[f"budget_{b}"][p]["per_seed"] for d in ds], float)
def best(ds, b, names):
    out, ch = [], []
    for d in ds:
        row = d[f"budget_{b}"]; m = {p: np.mean(row[p]["per_seed"]) for p in names if p in row}; k = max(m, key=m.get); out.append(row[k]["per_seed"]); ch.append(k)
    return np.array(out, float), ch
def cmp(a, b):
    ma, mb = a.mean(), b.mean(); sa, sb = a.std(), b.std()
    return {"ratio": float(ma / mb), "z": float((ma - mb) / np.sqrt(sa ** 2 + sb ** 2)) if (sa ** 2 + sb ** 2) > 0 else None, "mean_a": float(ma), "mean_b": float(mb), "sd_a": float(sa), "sd_b": float(sb),
            "split_ratios": [float(x) for x in a.mean(1) / b.mean(1)],
            "split_z": [float((x.mean() - y.mean()) / np.sqrt(x.std() ** 2 + y.std() ** 2)) if (x.std() ** 2 + y.std() ** 2) > 0 else None for x, y in zip(a, b)],
            "split_ratio_sd": float((a.mean(1) / b.mean(1)).std(ddof=1)) if len(a) > 1 else None}

def replay(tag, lay, splits):
    """Per-feature-set absolute means and the ratio family at every budget."""
    R = {}
    D = {fs: sims(tag, fs, lay, splits) for fs in FS}
    R["n_sim"] = [d["n_sim"] for d in D["cheap"]]; R["n_fit"] = [d["n_fit"] for d in D["cheap"]]; R["hits_in_sim"] = [d["hits"] for d in D["cheap"]]; R["n_buckets"] = [d["n_buckets"] for d in D["cheap"]]
    for b in BUD:
        cell = {"abs": {}, "ratios": {}}
        for fs in FS:
            ds = D[fs]; pols = [p for p in ds[0][f"budget_{b}"]]
            cell["abs"][fs] = {p: {"mean": float(A(ds, b, p).mean()), "sd_runs": float(A(ds, b, p).std()), "split_means": [float(x) for x in A(ds, b, p).mean(1)], "split_sd_of_means": float(A(ds, b, p).mean(1).std(ddof=1)) if len(splits) > 1 else None} for p in pols}
            thr, thr_c = best(ds, b, THR); rec, rec_c = best(ds, b, REC) if any(p in ds[0][f"budget_{b}"] for p in REC) else (None, None)
            ts, gr, st, sm, un = A(ds, b, "ts_online"), A(ds, b, "greedy_online"), A(ds, b, "static"), A(ds, b, "static_matched"), A(ds, b, "uniform")
            r = {"ts_vs_greedy": cmp(ts, gr), "ts_vs_best_thr": cmp(ts, thr), "greedy_vs_best_thr": cmp(gr, thr), "ts_vs_static_free": cmp(ts, st), "greedy_vs_static_free": cmp(gr, st),
                 "ts_vs_static_matched": cmp(ts, sm), "greedy_vs_static_matched": cmp(gr, sm), "best_thr_vs_uniform": cmp(thr, un), "greedy_vs_uniform": cmp(gr, un)}
            if rec is not None: r["ts_vs_best_recal"] = cmp(ts, rec); r["greedy_vs_best_recal"] = cmp(gr, rec); r["best_recal_vs_best_thr"] = cmp(rec, thr)
            r["best_thr_choice"] = thr_c; r["best_recal_choice"] = rec_c
            cell["ratios"][fs] = r
        cell["ratios"]["greedy_pc_vs_greedy_cheap"] = cmp(A(D["pc"], b, "greedy_online"), A(D["cheap"], b, "greedy_online"))
        cell["ratios"]["ts_pc_vs_ts_cheap"] = cmp(A(D["pc"], b, "ts_online"), A(D["cheap"], b, "ts_online"))
        cell["ratios"]["static_pc_vs_static_cheap"] = cmp(A(D["pc"], b, "static"), A(D["cheap"], b, "static"))
        R[b] = cell
    return R

bj = jl("build.json"); st = {t: jl(f"structure_{t}.json") for t in TAGS}
gc = jl("gate/gate_check.json") if os.path.exists(os.path.join(rd, "gate", "gate_check.json")) else None
res = {"protocol": __doc__.split("\n")[1:], "build": bj, "regression_gate": gc, "structure": {}, "replay": {}}
for t in TAGS:
    s = st[t]; d = s["delta"]
    res["structure"][t] = {"rows": s["rows"], "pids": s["pids"], "positives": s["positives"], "rate": s["rate"], "judge_logit_only_cv_auc": s["auc"]["judge_only"]["mean"], "judge_logit_raw_auc": s["auc"]["judge_logit_raw"],
                           "cheap_r_auc": s["auc"]["cheap"]["mean"], "cheap_r_auc_seed_sd": float(np.std(s["auc"]["cheap"]["per_seed"])), "cheap_plus_agreement_auc_positive_control": s["auc"]["agreement"]["mean"],
                           "delta": {k: {kk: v[kk] for kk in ("mean", "ci_lo", "ci_hi", "boot_sd", "null_mean", "null_sd") if kk in v} for k, v in d.items()},
                           "fraction_of_agreement_gain_recovered": s["fraction_of_agreement_gain_recovered"], "agreement_gain": s["agreement_gain"]}
    for lay, spl in (("cell100", [0, 1, 2, 3]), ("cell30", [0]), ("pid", [0])):
        res["replay"].setdefault(t, {})[lay] = replay(t, lay, spl)

# ---------------- verdict
gate_ok = bool(bj["gate_passed"] and gc and gc["passed"])
rungs = {}
for rg, pre in RUNG.items():
    t = f"{pre}_{LABEL}"; S = res["structure"][t]; dl = S["delta"]; R = res["replay"][t]["cell100"]["0.25"]["ratios"]
    pc = dl["pc"]; cells = {c: dl[c] for c in CELLS}
    a_pass = (pc["mean"] >= 0.05 and pc["ci_lo"] >= 0.03) or any(v["mean"] >= 0.02 and v["ci_lo"] >= 0.01 for v in cells.values())
    a_kill = pc["mean"] < 0.02 and pc["ci_hi"] < 0.03 and all(v["mean"] < 0.01 for v in cells.values())
    b = {}
    for fs in FS:
        x = R[fs]["ts_vs_greedy"]; y = R[fs]["ts_vs_best_thr"]
        s1 = [i for i in range(4) if x["split_ratios"][i] >= 1.05 and x["split_z"][i] is not None and x["split_z"][i] > 2]
        b[fs] = {"splits_ts_ge_1.05_greedy_and_gt_2sd": s1, "n_splits": len(s1), "ts_vs_thr_in_those_splits": [y["split_ratios"][i] for i in s1], "pass": bool(len(s1) >= 3 and all(y["split_ratios"][i] >= 1.10 for i in s1)),
                 "pooled_ts_vs_greedy": x["ratio"], "pooled_kill_ok": bool(x["ratio"] <= 1.03)}
    b_pass = any(b[fs]["pass"] for fs in FS)
    gp = R["greedy_pc_vs_greedy_cheap"]; b_kill = all(b[fs]["pooled_kill_ok"] for fs in FS) and gp["ratio"] <= 1.03
    learned = {f"{arm}_{fs}": R[fs][f"{arm}_vs_best_thr"] for arm in ("ts", "greedy") for fs in FS}
    best_learned = max(learned, key=lambda k: learned[k]["ratio"])
    lit_kill_rung = pc["mean"] < 0.01 and all(v["mean"] < 0.01 for v in cells.values()) and all(v["ratio"] <= 1.05 for v in learned.values())
    lit_pass_rung = ((pc["mean"] >= 0.05) or any(v["mean"] >= 0.02 for v in cells.values())) and any(v["ratio"] >= 1.10 and v["z"] is not None and v["z"] > 2 for v in learned.values())
    rungs[rg] = {"a_pass": bool(a_pass), "b_pass": bool(b_pass), "a_kill": bool(a_kill), "b_kill": bool(b_kill), "b_detail": b, "pooled_greedy_pc_vs_greedy_cheap": gp["ratio"],
                 "delta_pc": pc["mean"], "delta_pc_ci": [pc["ci_lo"], pc["ci_hi"]], "delta_cells": {c: v["mean"] for c, v in cells.items()},
                 "learned_vs_best_label_free_thr_25": {k: v["ratio"] for k, v in learned.items()}, "best_learned_arm_vs_thr": [best_learned, learned[best_learned]["ratio"], learned[best_learned]["z"]],
                 "literal_backlog_kill_rung_condition": bool(lit_kill_rung), "literal_backlog_pass_rung_condition": bool(lit_pass_rung)}
passed = gate_ok and any(r["a_pass"] and r["b_pass"] for r in rungs.values()); killed = gate_ok and all(r["a_kill"] and r["b_kill"] for r in rungs.values())
res["verdict"] = {"gate_ok": gate_ok, "rungs": rungs, "PASS": bool(passed), "KILL": bool(killed and not passed), "outcome": "VOID" if not gate_ok else ("PASS" if passed else ("KILL" if killed else "INCONCLUSIVE")),
                  "literal_backlog_kill": bool(gate_ok and all(r["literal_backlog_kill_rung_condition"] for r in rungs.values())), "literal_backlog_pass": bool(gate_ok and any(r["literal_backlog_pass_rung_condition"] for r in rungs.values()))}
# ---------------- G03 point and claim-2 points (agreement-free regime)
g = {}
for rg, pre in RUNG.items():
    t = f"{pre}_{LABEL}"; R = res["replay"][t]["cell100"]["0.25"]["ratios"]; S = res["structure"][t]
    g[rg] = {"cheap_auc": S["cheap_r_auc"], "delta_auc_pc": S["delta"]["pc"]["mean"], "gain_greedy_cheap_over_best_thr": R["cheap"]["greedy_vs_best_thr"]["ratio"] - 1, "gain_ts_cheap_over_best_thr": R["cheap"]["ts_vs_best_thr"]["ratio"] - 1,
             "greedy_cheap_over_best_recal": R["cheap"]["greedy_vs_best_recal"]["ratio"], "greedy_cheap_over_static_free": R["cheap"]["greedy_vs_static_free"]["ratio"]}
res["g03_points"] = g
json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---------------- markdown
f3 = lambda x: "n/a" if x is None else f"{x:.3f}"; f4 = lambda x: "n/a" if x is None else f"{x:+.4f}"
L = []; P = L.append
P("## Gates"); P(f"- build gate: {bj['gate_passed']} (judge AUC {json.dumps({k: round(v['rollout_auc_judge_p_vs_wrong'], 4) for k, v in bj['gate'].items()})}); regression gate vs runs/E04 (group mode, split 0, cell100, cheap): {gc}")
P("\n## Structure (OOF AUC, 5 folds by pid, 3 CV seeds; delta vs cheap_r; [95% pid-bootstrap CI]; null = pid permutation mean +- SD, 20 perms)")
P("| table | rows | positives | rate | judge_logit only | cheap_r | +16 PCs | cell30 | cell100 | cell300 | HGB +PCs | cheap_r + agreement (control) | PC share of agreement gain |"); P("|" + "---|" * 13)
for t in TAGS:
    S = res["structure"][t]; d = S["delta"]
    cl = lambda k: f"{f4(d[k]['mean'])} [{f4(d[k]['ci_lo'])},{f4(d[k]['ci_hi'])}] null {f4(d[k]['null_mean'])}+-{d[k]['null_sd']:.4f}"
    P(f"| {t} | {S['rows']} | {S['positives']} | {S['rate']:.3f} | {S['judge_logit_only_cv_auc']:.4f} (raw {S['judge_logit_raw_auc']:.4f}) | {S['cheap_r_auc']:.4f} | {cl('pc')} | {cl('cell30')} | {cl('cell100')} | {cl('cell300')} | {f4(d['hgb_pc']['mean'])} [{f4(d['hgb_pc']['ci_lo'])},{f4(d['hgb_pc']['ci_hi'])}] | {S['cheap_plus_agreement_auc_positive_control']:.4f} (+{S['agreement_gain']:.4f}) | {f3(S['fraction_of_agreement_gain_recovered']['pc'])} |")
P("\nShrinkage-80 cell encodings (robustness): " + "; ".join(f"{t}: " + ", ".join(f"{c} {f4(res['structure'][t]['delta'][c + '_s80']['mean'])}" for c in CELLS) for t in TAGS))
for lay in ("cell100", "cell30", "pid"):
    for t in TAGS:
        RR = res["replay"][t][lay]
        P(f"\n## Replay {t} layout {lay} (n_sim {RR['n_sim']}, buckets {RR['n_buckets']}, errors in sim {RR['hits_in_sim']}); mean errors caught over {len(RR['n_sim'])} split(s) x 8 seeds")
        for b in BUD:
            c = RR[b]; pols = list(c["abs"]["cheap"].keys())
            P(f"\nbudget {float(b):.0%}"); P("| features | " + " | ".join(pols) + " |"); P("|" + "---|" * (len(pols) + 1))
            for fs in FS: P(f"| {fs} | " + " | ".join(f"{c['abs'][fs][p]['mean']:.1f}+-{c['abs'][fs][p]['sd_runs']:.1f}" if p in c['abs'][fs] else "-" for p in pols) + " |")
            for fs in FS:
                r = c["ratios"][fs]
                P(f"- {fs}: " + "; ".join(f"{k} {r[k]['ratio']:.3f} (z {f3(r[k]['z'])}, splits {[round(x, 3) for x in r[k]['split_ratios']]})" for k in ("ts_vs_greedy", "ts_vs_best_thr", "greedy_vs_best_thr", "ts_vs_best_recal", "greedy_vs_best_recal", "ts_vs_static_free", "greedy_vs_static_free") if k in r) + f"; best_thr {r['best_thr_choice']}; best_recal {r['best_recal_choice']}")
            P(f"- greedy(cheap+pc)/greedy(cheap) {c['ratios']['greedy_pc_vs_greedy_cheap']['ratio']:.3f} (z {f3(c['ratios']['greedy_pc_vs_greedy_cheap']['z'])}); ts(cheap+pc)/ts(cheap) {c['ratios']['ts_pc_vs_ts_cheap']['ratio']:.3f}")
P("\n## Verdict"); P(json.dumps(res["verdict"], indent=1)); P("\n## G03 points (25% budget, cell100, corrupted label)"); P(json.dumps(g, indent=1))
open(os.path.join(rd, "summary.md"), "w").write("\n".join(L)); print("\n".join(L))
