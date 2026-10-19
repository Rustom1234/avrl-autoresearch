"""E10 assembly (CPU only): merge the GPU rollout summary, the structure readout (e10_analyze.py) and the E03 replay into
<run_dir>/result.json, and apply the pre-registered E10 rules mechanically. Re-running it is safe: the new keys are overwritten.

Rules (pre-registered in avrl/RESEARCH_LOG.md, E10 pre-registration):
  gate:   confident-subset (agreement >= 0.75) proxy error rate >= 0.03.
  claim2: gate fail -> no claim-2 verdict (logged as GATE_FAIL). Else KILL if point D(ts_online) at 25% <= 0,
          PASS if D > 0 and the 95% group-bootstrap CI excludes 0, INCONCLUSIVE otherwise.
  claim3: KILL if AUC gain (cheap+emb minus cheap, proxy_error) < 0.05 AND cell100 neighbor ratio < 1.5 (no gate condition,
          as written in the kill rule). PASS if gate passes AND gain >= 0.05 AND ratio >= 1.5. INCONCLUSIVE otherwise.
"""
import json, os, sys

RUN = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else "/homes/rdubash/avrl_live/CASCADE/avrl/runs/E10"
S = json.load(open(os.path.join(RUN, "summary.json")))        # p0_rollouts.py summary
ST = json.load(open(os.path.join(RUN, "result.json")))        # e10_analyze.py structure readout (p0 format)
RP = json.load(open(os.path.join(RUN, "replay", "result.json")))  # e03_threshold_bootstrap.py

GATE_MIN = 0.03
conf_err = ST["confident_subset"]["proxy_error_rate"]
gate_pass = conf_err >= GATE_MIN

p25, b25 = RP["D_point"]["0.25"], RP["D_bootstrap"]["ts_online"]["ci95"]
D25 = p25["ts_online"]
if not gate_pass:
    c2 = "GATE_FAIL (no claim-2 verdict)"
elif D25 <= 0:
    c2 = "KILL"
elif b25[0] > 0:
    c2 = "PASS"
else:
    c2 = "INCONCLUSIVE"

gain = ST["auc_cheap_emb"] - ST["auc_cheap"]
ratio = ST["neighbor_ratio"]["cell100"]
if gain < 0.05 and ratio < 1.5:
    c3 = "KILL"
elif gate_pass and gain >= 0.05 and ratio >= 1.5:
    c3 = "PASS"
else:
    c3 = "INCONCLUSIVE" + (" (gate fail blocks pass)" if not gate_pass and gain >= 0.05 and ratio >= 1.5 else "")

adm = {arm: RP["point"]["0.25"][arm]["admitted"] for arm in ("uniform", "threshold", "ts_online", "greedy_online")}
out = dict(ST)  # keep the structure readout keys at top level
out.update({
    "exp": "E10",
    "model": "OLMo-2-0425-1B-Instruct (substitute for Llama-3.2-3B, not on disk)",
    "rollouts": {k: S[k] for k in ("n_prompts", "G", "rollout_accuracy", "majority_accuracy", "group_proxy_error_rate",
                                   "rollout_proxy_error_rate", "frac_unanimous", "proxy_error_when_unanimous", "proxy_error_when_split", "wall_s")},
    "gate": {"confident_subset_proxy_error_rate": conf_err, "n_confident": ST["confident_subset"]["n"], "min": GATE_MIN, "pass": gate_pass},
    "claim2": {"verdict": c2, "metric": "admitted at 25% (1 - caught/total proxy errors), 8-seed mean",
               "admitted_25": adm, "pool": RP["pool"],
               "D_ts_online_25": D25, "D_ts_online_25_ci95": b25,
               "D_ts_online_10": RP["D_point"]["0.1"]["ts_online"], "D_ts_online_50": RP["D_point"]["0.5"]["ts_online"],
               "D_greedy_online_25": p25["greedy_online"], "D_greedy_online_25_ci95": RP["D_bootstrap"]["greedy_online"]["ci95"]},
    "claim3": {"verdict": c3, "auc_cheap": ST["auc_cheap"], "auc_cheap_emb": ST["auc_cheap_emb"], "auc_gain": gain,
               "neighbor_ratio_cell100": ratio, "neighbor_ratio_cell30": ST["neighbor_ratio"]["cell30"],
               "confident_subset_auc_cheap": ST["confident_subset"]["auc_cheap"], "confident_subset_auc_cheap_emb": ST["confident_subset"]["auc_cheap_emb"]},
    "replay_config": RP["config"],
    "reproduction_vs_E01": RP.get("reproduction_vs_E01"),
})
json.dump(out, open(os.path.join(RUN, "result.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("gate", "claim2", "claim3")}, indent=1))
