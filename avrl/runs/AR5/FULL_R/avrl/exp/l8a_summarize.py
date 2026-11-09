"""L8a summary (CPU), adapted from exp/l7a_summarize.py. Usage: l8a_summarize.py <runs/L8a> [<runs/L7a>]
Reads l8a_d<direction>_B<B>_o<order seed>_<mode>.json (direction 1 = 3B then 7B-AWQ, direction 2 = 7B-AWQ then 3B; per-step B in 62, 125; order seeds 0..15; 12 disjoint batches of 250; mode flip = relation flip f = -1:
judge_top and judge_mean z negated on every post-switch row, labels unchanged; mode gate = f = +1, the L7a drift run) and writes result.json and ledger_table.md. Ratios are ratios of means over order seeds (sums of hits),
CI = percentile bootstrap (10,000 resamples) over order seeds. "hits" are judge errors found (the label is judge_error).

Regression gate (VOID if it fails): the gate runs (f = +1) must reproduce runs/L7a drift per-batch hits exactly for every L7a arm and batch_base_errors, on all 16 order seeds, both directions, both B. Also void: a missing file,
a file whose header (budget, order seed, mode, flip factor) is wrong, any failed in-job self-check, a flip run whose batches 1..6 (any arm, base errors) differ from the gate run of the same cell and seed (nothing may differ before
the switch), or thr_agreement / restart_offset differing between flip and gate (they never see the flipped z columns; restart_offset refits inside each batch and thr_agreement uses agreement only).

Selection protocol (as the L6a/L7a runs): per cell the best FORGETTING arm (carry_plain_window_125/250/500, carry_plain_decay hl250, carry_plain_decay_hl100) and the best PERSISTENT arm (those five plus unbounded carry_plain and
carry_offset_lam1; never reset_at_switch, never the freeze arms) are each chosen by cumulative hits over batches 7..12 (mean over the selection seeds, ties to the first in list order) on order seeds 0..7 and reported on order
seeds 8..15 (VERDICT fold). Swapped fold (select on 8..15, report on 0..7) and pooled 16 (each seed reported with the arm chosen on the other half) are cross-checks.

Pre-registered verdict (L8a spec), per cell = direction x B (4 cells), verdict fold:
  HEADLINE      best forgetting arm / carry_plain over batches 7..12 (ratio of means, 95% percentile bootstrap CI over order seeds)
  VOID          if the regression gate / self-checks did not pass
  PASS          if best forgetting / carry_plain >= 1.03 with CI lower bound > 1 in at least 3 of 4 cells and, in those cells, best forgetting >= thr_agreement in batches 10..12
  KILL          if (i) carry_plain >= 0.99 x best forgetting over batches 7..12 in at least 3 of 4 cells, or (ii) best persistent < thr_agreement in both batches 7..9 and 10..12 in at least 3 of 4 cells
  INCONCLUSIVE  otherwise"""
import sys, os, json
import numpy as np

rd = sys.argv[1]; r7 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(rd)), "L7a")
DIRS = {"d1 3B->7B-AWQ": 1, "d2 7B-AWQ->3B": 2}; BUDS = (62, 125); NS, NB, NBOOT, SW = 16, 12, 10000, 6
MODES = {"flip": -1.0, "gate": 1.0}; PASS_R, KILL_R, RET_R = 1.03, 0.99, 1.03
PERSIST = ["carry_plain", "carry_plain_window_125", "carry_plain_window_250", "carry_plain_window_500", "carry_plain_decay", "carry_plain_decay_hl100", "carry_offset_lam1"]; FORGET = PERSIST[1:6]
ARMS = ["thr_agreement", "restart_offset"] + PERSIST + ["freeze_plain_k3", "freeze_plain_k6", "reset_at_switch"]
SHARED = [k for k in ARMS if k != "carry_plain_decay_hl100"]  # the arms runs/L7a drift holds
GRP = {"4-6 pre-switch": slice(3, 6), "7-9 post-switch": slice(6, 9), "10-12 recovery": slice(9, 12), "7-12": slice(6, 12)}
FA, FB = slice(0, 8), slice(8, 16)
rng = np.random.default_rng(0); IDX = {n: rng.integers(0, n, (NBOOT, n)) for n in (8, 16)}

def rat(num, den):
    """ratio of means of per-seed vectors with a percentile bootstrap CI over the order seeds (len 8 or 16)."""
    num, den = np.asarray(num, float), np.asarray(den, float); idx = IDX[len(num)]; rb = num[idx].mean(1) / den[idx].mean(1)
    return {"ratio": float(num.mean() / den.mean()), "ci_lo": float(np.percentile(rb, 2.5)), "ci_hi": float(np.percentile(rb, 97.5)), "n_order_seeds": int(len(num))}
inv = lambda r: {"ratio": 1 / r["ratio"], "ci_lo": 1 / r["ci_hi"], "ci_hi": 1 / r["ci_lo"], "n_order_seeds": r["n_order_seeds"]}

# ---------------- load + regression gate + self-checks
DATA, missing, mism, ncomp, nvals, chk_fail, pre_bad, meta_bad, inv_bad = {}, [], [], 0, 0, [], [], [], []
for dn, d in DIRS.items():
    for B in BUDS:
        cell = {m: {"A": {}, "base": [], "info": {}} for m in MODES}
        for m, fac in MODES.items():
            for o in range(NS):
                f = os.path.join(rd, f"l8a_d{d}_B{B}_o{o}_{m}.json")
                if not os.path.exists(f): missing.append(f); continue
                x = json.load(open(f))
                if not (x["budget"] == B and x["order_seed"] == o and x["n_batches"] == NB and x["m"] == 250 and x["mode"] == "drift" and x["flip_factor"] == fac and set(x["arms"]) == set(ARMS)): meta_bad.append(f); continue
                for k, v in x["arms"].items(): cell[m]["A"].setdefault(k, []).append(v)
                cell[m]["base"].append(x["batch_base_errors"]); cell[m]["info"][o] = x["arm_info"]
                bad = [k for k, v in x["checks"].items() if not v]
                if bad: chk_fail.append({"cell": f"{dn} B={B}", "mode": m, "order_seed": o, "failed_selfchecks": bad})
                if m == "gate":
                    g = os.path.join(r7, f"l7a_d{d}_B{B}_o{o}_drift.json")
                    if not os.path.exists(g): mism.append({"cell": f"{dn} B={B}", "order_seed": o, "arm": "L7a file missing", "file": g}); continue
                    ref = json.load(open(g))
                    if set(ref["arms"]) != set(SHARED): mism.append({"cell": f"{dn} B={B}", "order_seed": o, "arm": "L7a arm set differs", "l7a_arms": sorted(ref["arms"])}); continue
                    pairs = [(k, x["arms"][k], ref["arms"][k]) for k in SHARED] + [("batch_base_errors", x["batch_base_errors"], ref["batch_base_errors"])]
                    for k, a_, r_ in pairs:
                        ncomp += 1; nvals += len(r_); diffs = [i for i in range(len(r_)) if a_[i] != r_[i]] if len(a_) == len(r_) else list(range(max(len(a_), len(r_))))
                        if diffs: mism.append({"cell": f"{dn} B={B}", "order_seed": o, "arm": k, "batches_0idx": diffs, "l8a_gate": a_, "l7a": r_})
        if all(len(cell[m]["base"]) == NS for m in MODES):
            Af, Ag = cell["flip"]["A"], cell["gate"]["A"]
            for k in ARMS:  # nothing may differ before the switch: flip vs gate, batches 1..6, every arm, every seed
                bad = [o for o in range(NS) if Af[k][o][:SW] != Ag[k][o][:SW]]
                if bad: pre_bad.append({"cell": f"{dn} B={B}", "arm": k, "order_seeds": bad})
            bad = [o for o in range(NS) if cell["flip"]["base"][o] != cell["gate"]["base"][o]]
            if bad: pre_bad.append({"cell": f"{dn} B={B}", "arm": "batch_base_errors", "order_seeds": bad})
            for k in ("thr_agreement", "restart_offset"):  # never see the flipped columns: identical over all 12 batches
                bad = [o for o in range(NS) if Af[k][o] != Ag[k][o]]
                if bad: inv_bad.append({"cell": f"{dn} B={B}", "arm": k, "order_seeds": bad})
        DATA[(dn, B)] = {m: {"A": {k: np.array(v, float) for k, v in cell[m]["A"].items()}, "base": np.array(cell[m]["base"], float), "info": cell[m]["info"]} for m in MODES}
nd = sum(len(m.get("batches_0idx", [1])) for m in mism); n_expected = 2 * 2 * NS * (len(SHARED) + 1)  # 2 directions x 2 B x 16 seeds x (11 arms + batch_base_errors)
reset_flip_diff = sorted({(cn, o) for (dn, B), C in DATA.items() if len(C["flip"]["A"].get("reset_at_switch", [])) == NS == len(C["gate"]["A"].get("reset_at_switch", [])) for cn in [f"{dn} B={B}"]
                          for o in range(NS) if not np.array_equal(C["flip"]["A"]["reset_at_switch"][o], C["gate"]["A"]["reset_at_switch"][o])})
gate = {"passed": bool(not missing and not mism and not chk_fail and not pre_bad and not meta_bad and not inv_bad and ncomp == n_expected), "vector_comparisons": ncomp, "vector_comparisons_expected": n_expected,
        "batch_values_compared": nvals, "batch_value_differences": nd, "per_batch_differences_flip_f_plus1_vs_L7a_drift": nd, "arms_compared": SHARED + ["batch_base_errors"], "order_seeds": "0..15", "missing_files": missing[:20], "n_missing_files": len(missing),
        "mismatches_first20": mism[:20], "bad_headers_first20": meta_bad[:20], "in_job_selfcheck_failures": chk_fail[:20], "n_in_job_selfcheck_failures": len(chk_fail), "pre_switch_flip_vs_gate_differences": pre_bad[:20],
        "n_pre_switch_differences": len(pre_bad), "flip_invariant_arm_differences_thr_restart": inv_bad[:20], "n_flip_invariant_arm_differences": len(inv_bad),
        "reset_at_switch_flip_vs_gate_seeds_that_differ (informational, not a void condition)": [list(t) for t in reset_flip_diff]}
if not gate["passed"]:
    json.dump({"verdict": {"outcome": "VOID", "regression_gate_passed": False}, "regression_gate_vs_L7a": gate}, open(os.path.join(rd, "result.json"), "w"), indent=1)
    print("VOID: regression gate / self-checks did not pass", {k: v for k, v in gate.items() if k not in ("mismatches_first20", "in_job_selfcheck_failures", "pre_switch_flip_vs_gate_differences", "missing_files", "bad_headers_first20", "flip_invariant_arm_differences_thr_restart")})
    print(json.dumps(mism[:3])); print(chk_fail[:3]); print(pre_bad[:3]); print(meta_bad[:3]); print(inv_bad[:3]); sys.exit(3)

# ---------------- per-cell analysis
def ret(A, k):
    """per-batch ratio to thr_agreement (ratio of means over the 16 order seeds) for batches 7..12; (first post-switch batch with ratio >= 1.03, first from which it stays >= 1.03); 1 = batch 7, None = never."""
    r = [float(A[k][:, b].sum() / A["thr_agreement"][:, b].sum()) for b in range(SW, NB)]
    return r, next((i + 1 for i, x in enumerate(r) if x >= RET_R), None), next((i + 1 for i in range(len(r)) if all(x >= RET_R for x in r[i:])), None)

res = {"protocol": __doc__.split("\n")[1:], "cells": {}}
for (dn, B), C in DATA.items():
    A, G = C["flip"]["A"], C["gate"]["A"]
    assert all(a in A and a in G for a in ARMS) and all(v.shape == (NS, NB) for v in A.values()) and all(v.shape == (NS, NB) for v in G.values())
    S = lambda arr, k, sl, g=slice(SW, NB): arr[k][sl, g].sum(1)  # per-seed hits over batches g for the seeds in sl
    out = {"direction": dn, "budget": B, "n_order_seeds": NS, "n_batches": NB, "mean_base_errors_per_batch_pre_switch_table_A": float(C["flip"]["base"][:, :SW].mean()), "mean_base_errors_per_batch_post_switch_table_B": float(C["flip"]["base"][:, SW:].mean())}
    out["per_batch_mean_hits_flip"] = {k: [float(x) for x in A[k].mean(0)] for k in ARMS}; out["per_batch_mean_hits_gate_f_plus1"] = {k: [float(x) for x in G[k].mean(0)] for k in ARMS}
    out["per_batch_ratio_vs_thr_flip"] = {k: [float(A[k][:, b].sum() / A["thr_agreement"][:, b].sum()) for b in range(NB)] for k in ARMS}
    out["per_batch_ratio_vs_thr_gate_f_plus1"] = {k: [float(G[k][:, b].sum() / G["thr_agreement"][:, b].sum()) for b in range(NB)] for k in ARMS}
    tab = {}
    for k in ARMS:
        t = {"groups": {g: {"mean_hits_per_batch": float(A[k][:, sl].mean()), "over_thr": rat(A[k][:, sl].sum(1), A["thr_agreement"][:, sl].sum(1))} for g, sl in GRP.items()}}
        _, t["first_post_switch_batch_ge_1.03"], t["first_post_switch_batch_stays_ge_1.03"] = ret(A, k); _, t["gate_first_post_switch_batch_ge_1.03"], t["gate_first_post_switch_batch_stays_ge_1.03"] = ret(G, k)
        t["groups_gate_f_plus1_over_thr"] = {g: float(G[k][:, sl].sum() / G["thr_agreement"][:, sl].sum()) for g, sl in GRP.items()}
        t["flip_over_gate_thr_normalised"] = {g: float((A[k][:, sl].sum() / A["thr_agreement"][:, sl].sum()) / (G[k][:, sl].sum() / G["thr_agreement"][:, sl].sum())) for g, sl in GRP.items()}
        if k in PERSIST + ["freeze_plain_k3", "reset_at_switch"]: t["batches_7_12_over_freeze_plain_k6_16seeds"] = rat(S(A, k, slice(None)), S(A, "freeze_plain_k6", slice(None)))
        if k in FORGET: t["batches_7_12_over_carry_plain_16seeds"] = rat(S(A, k, slice(None)), S(A, "carry_plain", slice(None))); t["batches_10_12_over_carry_plain_16seeds"] = rat(S(A, k, slice(None), slice(9, 12)), S(A, "carry_plain", slice(None), slice(9, 12)))
        tab[k] = t
    out["table"] = tab
    out["fit_sets"] = {k: {"n_fit_mean": float(np.mean([C["flip"]["info"][o][k]["n_fit"] for o in range(NS)])), "fit_err_rate_mean": float(np.mean([C["flip"]["info"][o][k]["fit_err_rate"] for o in range(NS)])),
                           "n_fallback_thr_of_16": int(sum(C["flip"]["info"][o][k]["fallback_thr"] for o in range(NS)))} for k in ("freeze_plain_k3", "freeze_plain_k6")}
    allp = {g: {k: float(A[k][:, GRP[g]].sum() / A["thr_agreement"][:, GRP[g]].sum()) for k in PERSIST} for g in ("7-9 post-switch", "10-12 recovery")}
    out["every_persistent_arm_below_thr_all16"] = {g: {"all_below_thr": bool(max(v.values()) < 1.0), "max_ratio_to_thr": max(v.values()), "arm_with_max": max(v, key=v.get)} for g, v in allp.items()}
    selP = lambda sl: max(PERSIST, key=lambda k: S(A, k, sl).mean()); selF = lambda sl: max(FORGET, key=lambda k: S(A, k, sl).mean())  # ties go to the first arm in list order
    pA, pB, fA, fB = selP(FA), selP(FB), selF(FA), selF(FB)
    out["selection"] = {"best_persistent_on_seeds0_7": pA, "best_persistent_on_seeds8_15": pB, "best_forgetting_on_seeds0_7": fA, "best_forgetting_on_seeds8_15": fB,
                        "mean_hits_batches_7_12_by_persistent_arm_seeds0_7": {k: float(S(A, k, FA).mean()) for k in PERSIST}, "mean_hits_batches_7_12_by_persistent_arm_seeds8_15": {k: float(S(A, k, FB).mean()) for k in PERSIST},
                        "mean_hits_batches_7_12_thr_agreement_seeds0_7": float(S(A, "thr_agreement", FA).mean()), "mean_hits_batches_7_12_thr_agreement_seeds8_15": float(S(A, "thr_agreement", FB).mean()),
                        "best_forgetting_arm_by_post_switch_hits_all16": max(FORGET, key=lambda k: S(A, k, slice(None)).mean())}
    def fold_stats(P_of, F_of):  # P_of, F_of: {order seed: persistent / forgetting arm reported for that seed}
        sd = sorted(P_of); assert sd == sorted(F_of)
        pick = lambda arr, of, sl: np.array([arr[of[s]][s, sl].sum() for s in sd], float); fz = lambda arr, k, sl: np.array([arr[k][s, sl].sum() for s in sd], float)  # pick: reported arm per seed, fz: a fixed arm, hits over batches sl
        lbl = lambda of: of[sd[0]] if len(set(of.values())) == 1 else f"{of[0]} (seeds 0-7), {of[8]} (seeds 8-15)"
        g712, g79, g1012 = GRP["7-12"], GRP["7-9 post-switch"], GRP["10-12 recovery"]
        r_fc = rat(pick(A, F_of, g712), fz(A, "carry_plain", g712)); r_fc1012 = rat(pick(A, F_of, g1012), fz(A, "carry_plain", g1012)); r_ft1012 = rat(pick(A, F_of, g1012), fz(A, "thr_agreement", g1012)); r_ft79 = rat(pick(A, F_of, g79), fz(A, "thr_agreement", g79))
        r_pt79 = rat(pick(A, P_of, g79), fz(A, "thr_agreement", g79)); r_pt1012 = rat(pick(A, P_of, g1012), fz(A, "thr_agreement", g1012)); r_pf = rat(pick(A, P_of, g712), fz(A, "freeze_plain_k6", g712))
        r_gfc = rat(pick(G, F_of, g712), fz(G, "carry_plain", g712))  # same arms on the f = +1 run (the L7a mild judge swap): the dose reference
        return {"best_forgetting_arm": lbl(F_of), "best_persistent_arm": lbl(P_of), "best_forgetting_over_carry_plain_7_12": r_fc, "best_forgetting_over_carry_plain_10_12": r_fc1012, "carry_plain_over_best_forgetting_7_12": inv(r_fc),
                "best_forgetting_over_thr_7_9": r_ft79, "best_forgetting_over_thr_10_12": r_ft1012, "carry_plain_over_thr_7_9": rat(fz(A, "carry_plain", g79), fz(A, "thr_agreement", g79)), "carry_plain_over_thr_10_12": rat(fz(A, "carry_plain", g1012), fz(A, "thr_agreement", g1012)),
                "best_persistent_over_thr_7_9": r_pt79, "best_persistent_over_thr_10_12": r_pt1012, "best_persistent_over_thr_7_12": rat(pick(A, P_of, g712), fz(A, "thr_agreement", g712)), "best_persistent_over_freeze_plain_k6_7_12": r_pf,
                "reset_at_switch_over_best_persistent_UPPER_BOUND": rat(fz(A, "reset_at_switch", g712), pick(A, P_of, g712)), "reset_at_switch_over_best_forgetting_UPPER_BOUND": rat(fz(A, "reset_at_switch", g712), pick(A, F_of, g712)),
                "reset_at_switch_over_carry_plain_UPPER_BOUND": rat(fz(A, "reset_at_switch", g712), fz(A, "carry_plain", g712)), "freeze_plain_k6_over_thr_7_12": rat(fz(A, "freeze_plain_k6", g712), fz(A, "thr_agreement", g712)),
                "gate_f_plus1_same_forgetting_arms_over_carry_plain_7_12": r_gfc,
                "pass_cell": bool(r_fc["ratio"] >= PASS_R and r_fc["ci_lo"] > 1.0 and r_ft1012["ratio"] >= 1.0), "kill_i_cell_carry_plain_ge_0.99_best_forgetting": bool(1 / r_fc["ratio"] >= KILL_R),
                "kill_ii_cell_best_persistent_below_thr_in_7_9_and_10_12": bool(r_pt79["ratio"] < 1.0 and r_pt1012["ratio"] < 1.0)}
    folds = {"A": ({s: pA for s in range(8, 16)}, {s: fA for s in range(8, 16)}, "select on order seeds 0-7, report on 8-15 (VERDICT)"), "B": ({s: pB for s in range(8)}, {s: fB for s in range(8)}, "select on 8-15, report on 0-7"),
             "pooled16": ({**{s: pB for s in range(8)}, **{s: pA for s in range(8, 16)}}, {**{s: fB for s in range(8)}, **{s: fA for s in range(8, 16)}}, "each seed reported with the arm chosen on the other half")}
    out["verdict_folds"] = {fn: {"desc": d_, **fold_stats(po, fo)} for fn, (po, fo, d_) in folds.items()}; res["cells"][f"{dn} B={B}"] = out

# ---------------- verdict
def cell_eval(fold):
    g = lambda c, k: c["verdict_folds"][fold][k]
    rows = {cn: {"best_forgetting_arm": g(c, "best_forgetting_arm"), "best_persistent_arm": g(c, "best_persistent_arm"), "best_forgetting_over_carry_plain": g(c, "best_forgetting_over_carry_plain_7_12")["ratio"], "ci_lo": g(c, "best_forgetting_over_carry_plain_7_12")["ci_lo"],
                "best_forgetting_over_thr_10_12": g(c, "best_forgetting_over_thr_10_12")["ratio"], "carry_plain_over_best_forgetting": g(c, "carry_plain_over_best_forgetting_7_12")["ratio"], "best_persistent_over_thr_7_9": g(c, "best_persistent_over_thr_7_9")["ratio"],
                "best_persistent_over_thr_10_12": g(c, "best_persistent_over_thr_10_12")["ratio"], "carry_plain_over_thr_7_9": g(c, "carry_plain_over_thr_7_9")["ratio"], "carry_plain_over_thr_10_12": g(c, "carry_plain_over_thr_10_12")["ratio"],
                "pass_cell": g(c, "pass_cell"), "kill_i_cell": g(c, "kill_i_cell_carry_plain_ge_0.99_best_forgetting"), "kill_ii_cell": g(c, "kill_ii_cell_best_persistent_below_thr_in_7_9_and_10_12")} for cn, c in res["cells"].items()}
    nP, nKi, nKii = (sum(r[k] for r in rows.values()) for k in ("pass_cell", "kill_i_cell", "kill_ii_cell"))
    return {"cells": rows, "n_pass_cells_of_4": int(nP), "n_kill_i_cells_of_4": int(nKi), "n_kill_ii_cells_of_4": int(nKii), "PASS": bool(nP >= 3), "KILL_i": bool(nKi >= 3), "KILL_ii": bool(nKii >= 3), "KILL": bool(nKi >= 3 or nKii >= 3)}
vA, vB, vP = cell_eval("A"), cell_eval("B"), cell_eval("pooled16")
outcome = "PASS" if vA["PASS"] else "KILL" if vA["KILL"] else "INCONCLUSIVE"
cp_below = sum(r["carry_plain_over_thr_7_9"] < 1.0 for r in vA["cells"].values())
follow = {"PASS": "a forgetting horizon is required under a relation change: the best window or half-life becomes the Phase 1 spec and E13 gets its harsh-drift number; L8c (drift detector) is live as the alternative",
          "KILL": ("; ".join(x for x, c in (("kill (i): persistence needs no forgetting horizon even under a relation flip", vA["KILL_i"]), ("kill (ii): the best persistent arm stays below thr_agreement, persistence is unsafe under a relation change, Phase 1 needs a detector (L8c) or a thr_agreement fallback and claim 2's drift-robustness reading is withdrawn", vA["KILL_ii"])) if c)),
          "INCONCLUSIVE": "add an f = 0 (judge features ablated) dose row"}[outcome]
res["verdict"] = {"outcome": outcome, "regression_gate_passed": True, "verdict_fold": "best forgetting arm and best persistent arm chosen per cell on order seeds 0-7 by batches 7-12 hits, reported on order seeds 8-15 (8 order seeds)", "foldA": vA, "crosscheck_swapped_fold": vB, "crosscheck_pooled16": vP,
                  "kill_conditions_met": {"i_carry_plain_ge_0.99_best_forgetting_in_3_of_4": vA["KILL_i"], "ii_best_persistent_below_thr_in_7_9_and_10_12_in_3_of_4": vA["KILL_ii"]},
                  "cells_of_4_where_carry_plain_below_thr_in_7_9_verdict_fold": int(cp_below), "followup_by_outcome": follow + ("; L8c (drift detector) is live because unbounded carry_plain falls below thr_agreement in batches 7-9 in >= 1 cell" if outcome != "PASS" and cp_below >= 1 else "")}
res["regression_gate_vs_L7a"] = {k: v for k, v in gate.items() if k not in ("mismatches_first20",)}
res["headline_fold_A"] = {cn: {"best_forgetting_arm": r["best_forgetting_arm"], "best_forgetting_over_carry_plain_7_12": r["best_forgetting_over_carry_plain"], "ci_lo": r["ci_lo"], "best_forgetting_over_thr_10_12": r["best_forgetting_over_thr_10_12"],
                                "carry_plain_over_best_forgetting": r["carry_plain_over_best_forgetting"], "best_persistent_arm": r["best_persistent_arm"], "best_persistent_over_thr_7_9": r["best_persistent_over_thr_7_9"], "best_persistent_over_thr_10_12": r["best_persistent_over_thr_10_12"]} for cn, r in vA["cells"].items()}
res["notes"] = ["The flip multiplies the per-batch z of judge_top and judge_mean by -1 on every row of batches 7-12 (labels unchanged); labels carried across the switch keep the z they had when collected. f = +1 is the L7a drift run (gate). The other seven features are untouched, so the flip reverses the relation of two features, not of the whole proxy.",
                "restart_offset and thr_agreement never see the flipped columns and are identical in flip and gate runs; reset_at_switch learns only from post-switch rows whose flip is a consistent sign change (its flip and gate hits are expected, not guaranteed, identical: see reset_at_switch_flip_vs_gate_seeds_that_differ).",
                "Direction confound: 7B-AWQ judge labels are cleaner than 3B labels (judge_error rate 0.438 vs 0.473), so direction 1 moves to an easier table and direction 2 to a noisier one; the two directions are reported separately and never pooled into one claim.",
                "Selection departs from the 8-seed spec exactly as the L6a/L7a runs did: best arm chosen on order seeds 0-7, reported on 8-15, swapped fold and pooled 16 as cross-checks. The 16 order seeds permute the same 3,000 prompts, so order-seed CIs are optimistic (order and tie-break variation only, not prompt-sampling variation).",
                "reset_at_switch knows the switch and is an upper bound, never an arm in the verdict. freeze_plain_k3 / k6 are frozen before the switch (k <= 6) and never enter the persistent selection.",
                "The flip is a synthetic, maximal single-step reversal of two features of one population and one task (GSM8K judge_error labels from E04) at a known batch; offline replay; an artificial worst case for the two flipped features, not a measured judge change."]
json.dump(res, open(os.path.join(rd, "result.json"), "w"), indent=1)

# ---------------- ledger markdown
L = []; P = L.append
f3 = lambda r: f"{r['ratio']:.3f} ({r['ci_lo']:.3f} to {r['ci_hi']:.3f})"; fb = lambda x: "never" if x is None else str(x)
P(f"# L8a ledger table (CPU replay, regression gate vs runs/L7a: {gate['batch_value_differences']} per-batch differences in {gate['batch_values_compared']} values ({gate['vector_comparisons']} vectors), in-job self-check failures {gate['n_in_job_selfcheck_failures']}, pre-switch flip vs gate differences {gate['n_pre_switch_differences']})")
P(f"\nVerdict: **{outcome}** (verdict fold: best forgetting / best persistent arm chosen on order seeds 0-7 by batches 7-12, reported on 8-15; PASS cells {vA['n_pass_cells_of_4']} of 4, kill (i) cells {vA['n_kill_i_cells_of_4']} of 4, kill (ii) cells {vA['n_kill_ii_cells_of_4']} of 4). "
  f"Swapped fold: PASS {vB['n_pass_cells_of_4']}/4, kill (i) {vB['n_kill_i_cells_of_4']}/4, kill (ii) {vB['n_kill_ii_cells_of_4']}/4. Pooled 16: PASS {vP['n_pass_cells_of_4']}/4, kill (i) {vP['n_kill_i_cells_of_4']}/4, kill (ii) {vP['n_kill_ii_cells_of_4']}/4. Follow-up: {res['verdict']['followup_by_outcome']}.")
P("\nCaveats: synthetic single-step sign flip of two features (judge_top, judge_mean) at a known batch; the 16 order seeds permute the same 3,000 prompts, so CIs are optimistic; direction confound (7B-AWQ labels are cleaner); one population, offline replay on weak-judge labels.")
P("\n## Headline (verdict fold A, 8 order seeds 8-15)\n"); P("| cell | best forgetting (chosen on 0-7) | best forgetting / carry_plain, batches 7-12 (95% CI) | carry_plain / best forgetting | best forgetting / thr 10-12 | carry_plain / thr 7-9 | carry_plain / thr 10-12 | best persistent | best persistent / thr 7-9 | best persistent / thr 10-12 | best persistent / freeze_k6 (95% CI) | same forgetting arms / carry_plain at f=+1 | reset_at_switch / best persistent (upper bound) | pass | kill (i) | kill (ii) |"); P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for cn, c in res["cells"].items():
    f = c["verdict_folds"]["A"]; P(f"| {cn} | {f['best_forgetting_arm']} | {f3(f['best_forgetting_over_carry_plain_7_12'])} | {f['carry_plain_over_best_forgetting_7_12']['ratio']:.3f} | {f['best_forgetting_over_thr_10_12']['ratio']:.3f} | {f['carry_plain_over_thr_7_9']['ratio']:.3f} | {f['carry_plain_over_thr_10_12']['ratio']:.3f} | {f['best_persistent_arm']} | {f['best_persistent_over_thr_7_9']['ratio']:.3f} | {f['best_persistent_over_thr_10_12']['ratio']:.3f} | {f3(f['best_persistent_over_freeze_plain_k6_7_12'])} | {f3(f['gate_f_plus1_same_forgetting_arms_over_carry_plain_7_12'])} | {f3(f['reset_at_switch_over_best_persistent_UPPER_BOUND'])} | {f['pass_cell']} | {f['kill_i_cell_carry_plain_ge_0.99_best_forgetting']} | {f['kill_ii_cell_best_persistent_below_thr_in_7_9_and_10_12']} |")
for cn, c in res["cells"].items():
    tb = c["table"]; P(f"\n## {cn} (per-step budget {c['budget']} of 250; mean base errors per batch: table A {c['mean_base_errors_per_batch_pre_switch_table_A']:.1f}, table B {c['mean_base_errors_per_batch_post_switch_table_B']:.1f}; 16 order seeds)")
    P("\nFlip run (f = -1), all 16 order seeds. Hits per batch and ratio to thr_agreement in the batch group (CI over 16 order seeds); ratio to thr at f = +1 (the L7a judge swap) for 7-9 / 10-12; first post-switch batch with per-batch ratio >= 1.03 (1 = batch 7) / first from which it stays >= 1.03 (flip | f=+1); batches 7-12 / carry_plain (forgetting arms) and / freeze_plain_k6.\n")
    P("| arm | hits/batch 4-6 | 7-9 | 10-12 | / thr 4-6 | / thr 7-9 | / thr 10-12 | f=+1 / thr 7-9 | f=+1 / thr 10-12 | first >= 1.03 / stays (flip) | (f=+1) | 7-12 / carry_plain | 7-12 / freeze_k6 |"); P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ARMS:
        t = tb[k]; g = t["groups"]; fz = t.get("batches_7_12_over_freeze_plain_k6_16seeds"); fc = t.get("batches_7_12_over_carry_plain_16seeds"); gg = t["groups_gate_f_plus1_over_thr"]
        P(f"| {k} | " + " | ".join(f"{g[x]['mean_hits_per_batch']:.1f}" for x in list(GRP)[:3]) + " | " + " | ".join(f"{g[x]['over_thr']['ratio']:.3f}" for x in list(GRP)[:3]) + f" | {gg['7-9 post-switch']:.3f} | {gg['10-12 recovery']:.3f} | {fb(t['first_post_switch_batch_ge_1.03'])} / {fb(t['first_post_switch_batch_stays_ge_1.03'])} | {fb(t['gate_first_post_switch_batch_ge_1.03'])} / {fb(t['gate_first_post_switch_batch_stays_ge_1.03'])} | " + ("n/a" if fc is None else f3(fc)) + " | " + ("n/a" if fz is None else f3(fz)) + " |")
    s = c["selection"]; ep = c["every_persistent_arm_below_thr_all16"]
    P(f"\nSelection (mean hits over batches 7-12): best persistent on seeds 0-7 = {s['best_persistent_on_seeds0_7']}, on 8-15 = {s['best_persistent_on_seeds8_15']}; best forgetting on 0-7 = {s['best_forgetting_on_seeds0_7']}, on 8-15 = {s['best_forgetting_on_seeds8_15']}; best forgetting on all 16 = {s['best_forgetting_arm_by_post_switch_hits_all16']}.")
    P(f"Every persistent arm below thr_agreement (all 16 seeds): batches 7-9 {ep['7-9 post-switch']['all_below_thr']} (max ratio {ep['7-9 post-switch']['max_ratio_to_thr']:.3f}, {ep['7-9 post-switch']['arm_with_max']}); batches 10-12 {ep['10-12 recovery']['all_below_thr']} (max ratio {ep['10-12 recovery']['max_ratio_to_thr']:.3f}, {ep['10-12 recovery']['arm_with_max']}).")
    P("\nFrozen fits (flip run, mean over 16 order seeds): " + "; ".join(f"{k}: n_fit {v['n_fit_mean']:.0f}, label error rate {v['fit_err_rate_mean']:.2f}" + (f", {v['n_fallback_thr_of_16']} one-class fallbacks" if v["n_fallback_thr_of_16"] else "") for k, v in c["fit_sets"].items()))
    for nm, key in (("flip run (f = -1)", "per_batch_ratio_vs_thr_flip"), ("gate run (f = +1, the L7a judge swap)", "per_batch_ratio_vs_thr_gate_f_plus1")):
        P(f"\nArm / thr_agreement by batch index, {nm} (ratio of means over 16 order seeds; the switch is between batch 6 and 7):\n"); P("| arm | " + " | ".join(str(i + 1) for i in range(NB)) + " |"); P("|---|" + "---|" * NB)
        for k in ARMS[1:]: P(f"| {k} | " + " | ".join(f"{x:.3f}" for x in c[key][k]) + " |")
    P("\nVerdict folds (batches 7-12 unless stated; CI over 8 order seeds, 16 for pooled):\n"); P("| fold | best forgetting | best forgetting / carry_plain (95% CI) | 10-12 | carry_plain / best forgetting | best forgetting / thr 10-12 | best persistent | best persistent / thr 7-9 | / thr 10-12 | / freeze_k6 | reset / best persistent (upper bound) |"); P("|---|---|---|---|---|---|---|---|---|---|---|")
    for fn, f in c["verdict_folds"].items():
        P(f"| {fn} | {f['best_forgetting_arm']} | {f3(f['best_forgetting_over_carry_plain_7_12'])} | {f['best_forgetting_over_carry_plain_10_12']['ratio']:.3f} | {f['carry_plain_over_best_forgetting_7_12']['ratio']:.3f} | {f['best_forgetting_over_thr_10_12']['ratio']:.3f} | {f['best_persistent_arm']} | {f['best_persistent_over_thr_7_9']['ratio']:.3f} | {f['best_persistent_over_thr_10_12']['ratio']:.3f} | {f['best_persistent_over_freeze_plain_k6_7_12']['ratio']:.3f} | {f['reset_at_switch_over_best_persistent_UPPER_BOUND']['ratio']:.3f} |")
open(os.path.join(rd, "ledger_table.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
