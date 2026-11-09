"""E06 assembly (CPU only): merge the GPU rollout summary (summary.json), the structure readout (e06_analyze.py, result.json),
the E03-style replay (replay/result.json, e03_threshold_bootstrap.py) and the gate readout (e06_gate.py, gate.json) into
<run_dir>/result.json, and apply the pre-registered E06 rules mechanically (avrl/RESEARCH_LOG.md, "E06 pre-registration").
Re-running it is safe: the new keys are overwritten. Copy of avrl/exp/e10_assemble.py with the gate, truncation and claim rules
changed as the E06 entry states.

Rules:
  gate (cutoff 0.75): truncated share > 0.10 -> TRUNCATION_FLAGGED (no gate verdict); else rate < 0.03 -> KILL;
      else rate >= 0.03 and k >= 100 -> OPEN; else rate >= 0.03 and k < 100 -> INCONCLUSIVE (power-limited, descriptive).
  claim2: only when the gate is OPEN. KILL if point D(ts_online) at 25% <= 0; PASS if D > 0 and the 95% bootstrap CI excludes 0;
      INCONCLUSIVE otherwise. Any other gate state: NO_VERDICT.
  claim3: not gated. KILL if AUC gain (cheap+emb minus cheap) < 0.05 AND cell100 neighbor ratio < 1.5. PASS if gain >= 0.05
      AND ratio >= 1.5. INCONCLUSIVE otherwise. The truncation flag is recorded next to it.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e06_gate import gate

RUN = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else "/homes/rdubash/avrl_live/CASCADE/avrl/runs/E06"
S = json.load(open(os.path.join(RUN, "summary.json")))               # p0_rollouts.py summary
ST = json.load(open(os.path.join(RUN, "result.json")))               # e06_analyze.py structure readout (p0 format)
RP = json.load(open(os.path.join(RUN, "replay", "result.json")))     # e03_threshold_bootstrap.py
GT = gate(RUN)                                                        # e06_gate.py: confident-subset counts, truncation share
json.dump(GT, open(os.path.join(RUN, "gate.json"), "w"), indent=1)

c75, c625, trunc = GT["cut_0.75"], GT["cut_0.625"], GT["truncated_share"]
rate, k, n = c75["rate"], c75["k"], c75["n"]
if trunc > 0.10:
    gate_state = "TRUNCATION_FLAGGED (no gate verdict)"
elif rate < 0.03:
    gate_state = "KILL"
elif rate >= 0.03 and k >= 100:
    gate_state = "OPEN"
else:
    gate_state = "INCONCLUSIVE (power-limited, descriptive only)"

p25 = RP["D_point"]["0.25"]; D25 = p25["ts_online"]; b25 = RP["D_bootstrap"]["ts_online"]["ci95"]
if gate_state != "OPEN":
    c2 = "NO_VERDICT (gate " + gate_state + ")"
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
elif gain >= 0.05 and ratio >= 1.5:
    c3 = "PASS"
else:
    c3 = "INCONCLUSIVE"

adm = {arm: RP["point"]["0.25"][arm]["admitted"] for arm in ("uniform", "threshold", "ts_online", "greedy_online")}
out = dict(ST)  # keep the structure readout keys at top level
out.update({
    "exp": "E06",
    "model": "Qwen2.5-0.5B-Instruct (active-audit models; same family as the E01 policy Qwen2.5-1.5B-Instruct)",
    "rollouts": {k_: S[k_] for k_ in ("n_prompts", "G", "rollout_accuracy", "majority_accuracy", "group_proxy_error_rate",
                                      "rollout_proxy_error_rate", "frac_unanimous", "proxy_error_when_unanimous", "proxy_error_when_split", "wall_s")},
    "gate": {"cut_0.75": c75, "cut_0.625": c625, "truncated_share": trunc, "truncated_rollouts": GT["truncated_rollouts"],
             "state": gate_state, "rule": "OPEN: rate>=0.03 and k>=100; KILL: rate<0.03; INCONCLUSIVE: rate>=0.03 and k<100; truncation share>0.10 flags, no gate verdict"},
    "claim2": {"verdict": c2, "metric": "admitted at 25% (1 - caught/total proxy errors), 8-seed mean",
               "admitted_25": adm, "pool": RP["pool"],
               "D_ts_online_25": D25, "D_ts_online_25_ci95": b25,
               "D_ts_online_10": RP["D_point"]["0.1"]["ts_online"], "D_ts_online_50": RP["D_point"]["0.5"]["ts_online"],
               "D_greedy_online_25": p25["greedy_online"], "D_greedy_online_25_ci95": RP["D_bootstrap"]["greedy_online"]["ci95"]},
    "claim3": {"verdict": c3, "auc_cheap": ST["auc_cheap"], "auc_cheap_emb": ST["auc_cheap_emb"], "auc_gain": gain,
               "neighbor_ratio_cell100": ratio, "neighbor_ratio_cell30": ST["neighbor_ratio"]["cell30"],
               "truncation_flag": trunc > 0.10, "truncated_share": trunc,
               "confident_subset_auc_cheap": ST["confident_subset"]["auc_cheap"], "confident_subset_auc_cheap_emb": ST["confident_subset"]["auc_cheap_emb"]},
    "replay_config": RP["config"],
    "reproduction_vs_E01": RP.get("reproduction_vs_E01"),
})
json.dump(out, open(os.path.join(RUN, "result.json"), "w"), indent=1)
print(json.dumps({"gate": out["gate"]["state"], "gate_cut_0.75": c75, "claim2": c2, "claim3": c3, "truncated_share": trunc}, indent=1))
