"""L3a finalize: assemble <run_dir>/result.json from gates.json + result_stage1.json and apply the pre-registered verdict bins.
Stage 2 (sim_warm_boot_hs.py) is only defined when stage 1 clears; this script refuses to label a run whose stage 1 cleared but stage 2 is missing."""
import os, sys, json
Q = sys.argv[1]; g = json.load(open(os.path.join(Q, "gates.json"))); r = json.load(open(os.path.join(Q, "result_stage1.json"))); s1 = r["stage1"]; c = r["confident"]["sets"]; o = r["overall"]["sets"]
assert g["all_pass"] and r["reproduction_gate"]["pass"], "a gate failed"
assert not s1["clears"], "stage 1 cleared: stage 2 must be run before a verdict"
label = "KILL" if s1["kill_candidate"] else "INCONCLUSIVE_POWER_LIMITED"
reason = (f"primary delta AUC {s1['primary_delta']:+.4f} (fold SD {s1['primary_delta_fold_sd']:.4f}), bootstrap 95% CI [{s1['primary_boot_ci95'][0]:+.4f}, {s1['primary_boot_ci95'][1]:+.4f}]: "
          f"below the 0.05 pass bar (CI lower end not above 0), but " + ("all three kill conditions hold" if label == "KILL" else
          f"kill not met (delta >= 0.02: {not s1['kill_conditions']['primary_delta_lt_0.02']}, CI upper end {s1['primary_boot_ci95'][1]:+.4f} >= 0.05: {not s1['kill_conditions']['boot_ci_upper_lt_0.05']}); stage 2 not run by design"))
res = {"id": "L3a-hidden-state-probe", "verdict": {"label": label, "reason": reason}, "gates": {"hidden_state_extraction": g, "e21_reproduction": r["reproduction_gate"], "sanity_hsA20_alone_overall_auc": r["sanity"]},
       "confident_subset": {"n": r["confident"]["n"], "n_err": r["confident"]["n_err"], "auc_agreement_only": r["confident"]["auc_agreement_only"], "sets": c},
       "overall_subset": {"n": r["overall"]["n"], "n_err": r["overall"]["n_err"], "auc_agreement_only": r["overall"]["auc_agreement_only"], "sets": o},
       "primary": {"set": s1["primary"], "auc_cheap": c["cheap"]["auc"], "auc_cheap_hsA20": c[s1["primary"]]["auc"], "delta": s1["primary_delta"], "delta_fold_sd": s1["primary_delta_fold_sd"],
                   "boot_ci95": s1["primary_boot_ci95"], "boot_sd": s1["primary_boot_sd"], "control_cheap_emb_delta": c["cheap_emb"]["delta"], "control_cheap_emb_boot_ci95": c["cheap_emb"]["boot_ci95"]},
       "stage1": s1, "free_label_ceiling": r["free_label_ceiling"], "stage2": {"run": False, "reason": "stage 1 primary delta < 0.05 (gate: delta >= 0.05 with CI lower end > 0)"}}
json.dump(res, open(os.path.join(Q, "result.json"), "w"), indent=1); print(label, "|", reason)
