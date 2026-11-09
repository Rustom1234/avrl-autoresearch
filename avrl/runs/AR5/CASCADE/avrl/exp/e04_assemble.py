"""E04 assembly (CPU only): merges the GPU summary, the structure metrics and the claim-2 replay into avrl/runs/E04/result.json,
with the pre-registered verdicts. The post-hoc block is descriptive only, added after the structure readout was read; it does not enter any verdict."""
import json
import pandas as pd
from sklearn.metrics import roc_auc_score

RUN = "/homes/rdubash/avrl_live/CASCADE/avrl/runs/E04"
gpu = json.load(open(f"{RUN}/judge_gpu_summary.json"))
st = json.load(open(f"{RUN}/structure.json"))
rp = json.load(open(f"{RUN}/replay/result.json"))
S = pd.read_parquet(f"{RUN}/judge_scores.parquet")
F = pd.read_parquet(f"{RUN}/feats_judge.parquet")
R = pd.read_parquet("/homes/rdubash/avrl_live/CASCADE/avrl/runs/p0/rollouts.parquet")
X = S.merge(R[["pid", "k", "correct"]], on=["pid", "k"], validate="one_to_one")

posthoc = {
    "note": "post-hoc, descriptive only; not a pre-registered metric",
    "rollout_auc_p_yes_vs_correct": float(roc_auc_score(X.correct, X.p_yes)),
    "mean_p_yes_correct_rollouts": float(X.p_yes[X.correct == 1].mean()),
    "mean_p_yes_wrong_rollouts": float(X.p_yes[X.correct == 0].mean()),
    "judge_pick_accuracy": 1.0 - st["judge_error"]["rate_all"],
    "random_pick_accuracy_expected": float(F.pass_rate.mean()),
    "majority_accuracy": float(F.majority_correct.mean()),
    "judge_pick_is_empty_answer_frac": float((F.judge_pick == "").mean()),
}

s_v = st["structure_verdict"]
D25 = rp["D_point"]["0.25"]["ts_online"]
ci25 = rp["D_bootstrap"]["ts_online"]["ci95"]
c_v = "PASS" if (D25 > 0 and ci25[0] > 0) else "KILL"
if s_v.startswith("FAILED"):
    joint = "FAILED_PROXY (no verdict on proxy type)"
elif s_v == "PASS" and c_v == "PASS":
    joint = "GO"
elif s_v != "PASS" and c_v != "PASS" and "KILL" in (s_v, c_v):
    joint = "KILL"
else:
    joint = "INCONCLUSIVE"

result = {
    "exp": "E04",
    "verdict": joint,
    "structure_verdict": s_v,
    "claim2_verdict": c_v,
    "claim2_verdict_source": "replay copy of E03 rule with LABEL=judge_error (the replay's own json says exp E03)",
    "gates_read_before_verdict": st["gates"],
    "judge": {"model": gpu["judge"], "topk": gpu["topk"], "n_rows": gpu["n_rows"], "n_groups": gpu["n_groups"],
              "p_yes_mean": gpu["p_yes_mean"], "p_yes_sd": gpu["p_yes_sd"], "p_yes_quantiles": gpu["p_yes_quantiles"],
              "n_fallback_yes": gpu["n_fallback_yes"], "n_fallback_no": gpu["n_fallback_no"], "n_both_fallback": gpu["n_both_fallback"]},
    "judge_error": st["judge_error"],
    "proxy_error_reference": st["proxy_error_reference"],
    "sizes": st["sizes"],
    "auc_judge_error": st["auc_judge_error"],
    "neighbor_ratio_judge_error_cell100": st["neighbor_ratio_judge_error_cell100"],
    "overlap_proxy_vs_judge": st["overlap"],
    "claim2_readout_25pct": {
        "budget_docs": rp["pool"]["budget_docs"]["0.25"], "sim_groups": rp["pool"]["n_sim"], "total_judge_errors_in_pool": rp["pool"]["total_errors"],
        "seeds": rp["config"]["seeds"] if "seeds" in rp["config"] else 8, "boot_resamples": rp["config"]["boot_resamples"], "boot_seed": rp["config"]["boot_seed"],
        "admitted_threshold": rp["point"]["0.25"]["threshold"]["admitted"], "admitted_ts_online": rp["point"]["0.25"]["ts_online"]["admitted"],
        "admitted_greedy_online": rp["point"]["0.25"]["greedy_online"]["admitted"], "admitted_uniform": rp["point"]["0.25"]["uniform"]["admitted"],
        "D_ts_online_point": D25, "D_ts_online_ci95": ci25, "D_greedy_online_point": rp["D_point"]["0.25"]["greedy_online"],
        "D_greedy_online_ci95": rp["D_bootstrap"]["greedy_online"]["ci95"],
        "descriptive_D_ts_online_by_budget": {b: rp["D_point"][b]["ts_online"] for b in rp["D_point"]},
        "replay_reproduction_check": rp["reproduction_vs_E01"],
        "replay_reproduction_note": "expected mismatch: the check compares against the E01 proxy-label json, not the judge label",
    },
    "posthoc_descriptive": posthoc,
    "files": {"judge_scores": f"{RUN}/judge_scores.parquet", "feats_judge": f"{RUN}/feats_judge.parquet", "structure": f"{RUN}/structure.json",
              "replay": f"{RUN}/replay/result.json", "gpu_summary": f"{RUN}/judge_gpu_summary.json"},
}
json.dump(result, open(f"{RUN}/result.json", "w"), indent=1)
print(json.dumps({k: result[k] for k in ("verdict", "structure_verdict", "claim2_verdict")}, indent=1))
print(json.dumps(posthoc, indent=1))
