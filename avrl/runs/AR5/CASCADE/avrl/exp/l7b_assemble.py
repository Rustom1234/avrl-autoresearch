"""L7b assembly (CPU only): merges the GPU summary, the structure metrics, the split-group pick accuracy and the claim-2 replay
into avrl/runs/L7b/result.json with the pre-registered verdicts (RESEARCH_LOG.md, L7b pre-registration). Copy of e04_assemble.py
with paths repointed and the capacity readout added. The post-hoc block is descriptive only and does not enter any verdict."""
import json
import pandas as pd
from sklearn.metrics import roc_auc_score

RUN = "/homes/rdubash/avrl_live/CASCADE/avrl/runs/L7b"
CAP_PASS, CAP_KILL = 0.397, 0.347  # pre-registered: split-group pick accuracy at least 39.7% passes, at most 34.7% kills
gpu = json.load(open(f"{RUN}/judge_gpu_summary.json"))
st = json.load(open(f"{RUN}/structure.json"))
rp = json.load(open(f"{RUN}/replay/result.json"))
cap = json.load(open(f"{RUN}/pick_accuracy.json"))
S = pd.read_parquet(f"{RUN}/judge_scores.parquet")
F = pd.read_parquet(f"{RUN}/feats_judge.parquet")
R = pd.read_parquet("/homes/rdubash/avrl_live/CASCADE/avrl/runs/p0/rollouts.parquet")
X = S.merge(R[["pid", "k", "correct"]], on=["pid", "k"], validate="one_to_one")

posthoc = {
    "note": "post-hoc, descriptive only; not a pre-registered metric",
    "rollout_auc_p_yes_vs_correct": float(roc_auc_score(X.correct, X.p_yes)),
    "mean_p_yes_correct_rollouts": float(X.p_yes[X.correct == 1].mean()),
    "mean_p_yes_wrong_rollouts": float(X.p_yes[X.correct == 0].mean()),
    "judge_pick_accuracy_all_groups": 1.0 - st["judge_error"]["rate_all"],
    "random_rollout_accuracy": float(F.pass_rate.mean()),
    "majority_accuracy": float(F.majority_correct.mean()),
    "judge_pick_is_empty_answer_frac": float((F.judge_pick == "").mean()),
}

s_v = st["structure_verdict"]
D25 = rp["D_point"]["0.25"]["ts_online"]
ci25 = rp["D_bootstrap"]["ts_online"]["ci95"]
c_v = "PASS" if (D25 > 0 and ci25[0] > 0) else "KILL"
acc = cap["pick_accuracy_split"]
k_v = "PASS" if acc >= CAP_PASS else ("KILL" if acc <= CAP_KILL else "INCONCLUSIVE")
if s_v.startswith("FAILED"):
    joint = "FAILED_PROXY (no verdict on proxy type)"
elif "KILL" in (s_v, c_v, k_v):
    joint = "KILL"
elif s_v == "PASS" and c_v == "PASS" and k_v == "PASS":
    joint = "GO"
else:
    joint = "INCONCLUSIVE"

pool_total = rp["pool"]["total_errors"]
budget25 = rp["pool"]["budget_docs"]["0.25"]
oracle_admitted = 1 - min(budget25, pool_total) / pool_total

result = {
    "exp": "L7b",
    "verdict": joint,
    "structure_verdict": s_v,
    "claim2_verdict": c_v,
    "capacity_verdict": k_v,
    "verdict_rule": "KILL if structure KILL or claim-2 KILL or capacity KILL; GO if all three PASS; INCONCLUSIVE otherwise",
    "gates_read_before_verdict": st["gates"],
    "capacity_readout_split_groups": {
        "pick_accuracy_split": acc, "pick_accuracy_split_se_binomial": cap["pick_accuracy_split_se_binomial"],
        "n_split_groups": cap["n_split_groups"], "uniform_over_distinct_split": cap["uniform_over_distinct_split"],
        "paired_diff_pick_minus_uniform_split": cap["paired_diff_pick_minus_uniform_split"], "paired_diff_se_split": cap["paired_diff_se_split"],
        "pass_at_least": CAP_PASS, "kill_at_most": CAP_KILL,
    },
    "claim2_verdict_source": "replay copy of E03 rule with LABEL=judge_error (the replay's own json says exp E03)",
    "judge": {"model": gpu["judge"], "topk": gpu["topk"], "enable_thinking": gpu["enable_thinking"],
              "chat_template_fallback": gpu["chat_template_fallback"], "n_rows": gpu["n_rows"], "n_groups": gpu["n_groups"],
              "p_yes_mean": gpu["p_yes_mean"], "p_yes_sd": gpu["p_yes_sd"], "p_yes_quantiles": gpu["p_yes_quantiles"],
              "n_fallback_yes": gpu["n_fallback_yes"], "n_fallback_no": gpu["n_fallback_no"], "n_both_fallback": gpu["n_both_fallback"]},
    "judge_error": st["judge_error"],
    "proxy_error_reference": st["proxy_error_reference"],
    "sizes": st["sizes"],
    "auc_judge_error": st["auc_judge_error"],
    "neighbor_ratio_judge_error_cell100": st["neighbor_ratio_judge_error_cell100"],
    "overlap_proxy_vs_judge": st["overlap"],
    "claim2_readout_25pct": {
        "budget_docs": budget25, "sim_groups": rp["pool"]["n_sim"], "total_judge_errors_in_pool": pool_total,
        "seeds": rp["config"]["seeds"], "boot_resamples": rp["config"]["boot_resamples"], "boot_seed": rp["config"]["boot_seed"],
        "admitted_threshold": rp["point"]["0.25"]["threshold"]["admitted"], "admitted_ts_online": rp["point"]["0.25"]["ts_online"]["admitted"],
        "admitted_greedy_online": rp["point"]["0.25"]["greedy_online"]["admitted"], "admitted_uniform": rp["point"]["0.25"]["uniform"]["admitted"],
        "admitted_oracle_ceiling": oracle_admitted,
        "max_possible_D_ts_online_vs_threshold": rp["point"]["0.25"]["threshold"]["admitted"] - oracle_admitted,
        "D_ts_online_point": D25, "D_ts_online_ci95": ci25,
        "D_greedy_online_point": rp["D_point"]["0.25"]["greedy_online"], "D_greedy_online_ci95": rp["D_bootstrap"]["greedy_online"]["ci95"],
        "descriptive_D_ts_online_by_budget": {b: rp["D_point"][b]["ts_online"] for b in rp["D_point"]},
        "replay_reproduction_check": rp["reproduction_vs_E01"],
        "replay_reproduction_note": "expected mismatch: the check compares against the E01 proxy-label json, not the judge label",
    },
    "posthoc_descriptive": posthoc,
    "files": {"judge_scores": f"{RUN}/judge_scores.parquet", "feats_judge": f"{RUN}/feats_judge.parquet", "structure": f"{RUN}/structure.json",
              "replay": f"{RUN}/replay/result.json", "pick_accuracy": f"{RUN}/pick_accuracy.json", "verify": f"{RUN}/verify/verify.json",
              "gpu_summary": f"{RUN}/judge_gpu_summary.json"},
}
json.dump(result, open(f"{RUN}/result.json", "w"), indent=1)
print(json.dumps({k: result[k] for k in ("verdict", "structure_verdict", "claim2_verdict", "capacity_verdict")}, indent=1))
print(json.dumps(result["capacity_readout_split_groups"], indent=1))
print(json.dumps(result["claim2_readout_25pct"]["admitted_oracle_ceiling"]), json.dumps(result["claim2_readout_25pct"]["max_possible_D_ts_online_vs_threshold"]))
