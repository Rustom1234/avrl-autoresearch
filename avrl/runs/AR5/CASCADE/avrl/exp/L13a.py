"""L13a readout (CPU, read-only on the run dir except result.json). Pre-registered rule (avrl/RESEARCH_LOG.md, L13a pre-registration):
at cutoff 0.625 on the confident subset (agreement >= 0.625), PASS (gate opens) if rate >= 0.03 AND k >= 100,
KILL if rate < 0.03, INCONCLUSIVE if rate >= 0.03 and k < 100. Truncated share above 0.10 is flagged, not a verdict change.
Recomputes n and k from groups.parquet, checks gate.json, exit_code, run.log and the rollout count, and reports overlap with E06 prompts.
Usage: L13a.py [run_dir]   (default avrl/runs/L13a). Writes <run_dir>/result.json."""
import json, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from e06_gate import wilson95

RUN = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "avrl", "runs", "L13a")
N_PROMPTS, G_SAMPLES, CUT, RATE_FLOOR, K_FLOOR, TRUNC_FLAG = 7473, 8, 0.625, 0.03, 100, 0.10

G = pd.read_parquet(os.path.join(RUN, "groups.parquet"))
R = pd.read_parquet(os.path.join(RUN, "rollouts.parquet"))
gate = json.load(open(os.path.join(RUN, "gate.json")))
summary = json.load(open(os.path.join(RUN, "summary.json")))
exit_code = open(os.path.join(RUN, "exit_code")).read().strip()
tracebacks = open(os.path.join(RUN, "run.log"), errors="replace").read().count("Traceback")


def cell(df, cut):
    c = df[df.agreement >= cut]
    n, k = int(len(c)), int(c.proxy_error.sum())
    lo, hi = wilson95(k, n)
    return {"n": n, "k": k, "rate": k / n if n else float("nan"), "wilson95": [lo, hi]}


primary = cell(G, CUT)
descriptive = cell(G, 0.75)
recomputed_matches = (primary["n"], primary["k"]) == (gate["cut_0.625"]["n"], gate["cut_0.625"]["k"]) and \
    (descriptive["n"], descriptive["k"]) == (gate["cut_0.75"]["n"], gate["cut_0.75"]["k"])

rate, k = primary["rate"], primary["k"]
if rate < RATE_FLOOR:
    verdict = "KILL"
elif k >= K_FLOOR:
    verdict = "PASS"
else:
    verdict = "INCONCLUSIVE"

# overlap with the E06 prompt set (expected to be the first 3,000 prompts of the same seed-0 permutation)
E06 = pd.read_parquet(os.path.join(ROOT, "avrl", "runs", "E06", "groups.parquet"))
in_e06 = G.question.isin(set(E06.question))
prefix_match = bool(len(E06) <= len(G) and (G.question.iloc[:len(E06)].values == E06.question.values).all())
overlap = {"prefix_of_pid_order_matches_E06": prefix_match, "groups_shared_with_E06": int(in_e06.sum()),
           "at_0.625_shared": cell(G[in_e06], CUT), "at_0.625_not_in_E06": cell(G[~in_e06], CUT)}

out = {
    "experiment": "L13a", "policy": "Qwen2.5-0.5B-Instruct", "n_prompts": N_PROMPTS, "G": G_SAMPLES, "max_tokens": 512, "seed": 0,
    "verdict_at_0.625": verdict,
    "primary_cutoff_0.625": primary,
    "descriptive_cutoff_0.75": descriptive,
    "truncated_rollouts": gate["truncated_rollouts"], "truncated_share": gate["truncated_share"],
    "truncated_flag_above_0.10": bool(gate["truncated_share"] > TRUNC_FLAG),
    "rollout_accuracy": summary["rollout_accuracy"], "majority_accuracy": summary["majority_accuracy"],
    "group_proxy_error_rate": summary["group_proxy_error_rate"],
    "checks": {"exit_code": exit_code, "tracebacks_in_run_log": tracebacks, "groups": int(len(G)),
               "groups_expected": N_PROMPTS, "rollouts": int(len(R)), "rollouts_expected": N_PROMPTS * G_SAMPLES,
               "recomputed_matches_gate_json": recomputed_matches},
    "overlap_with_E06": overlap,
    "claim2_readout": "not run; separate pre-registered item, runs only if verdict is PASS",
}
json.dump(out, open(os.path.join(RUN, "result.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
