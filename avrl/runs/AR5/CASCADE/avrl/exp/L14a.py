"""L14a readout (CPU, read-only on avrl/runs/L13a and avrl/runs/L14a, writes only avrl/runs/L14a/result.json).
Pre-registered rule (avrl/RESEARCH_LOG.md, L14a pre-registration). Pool = the L13a train groups with pid >= 3000 (4,473 prompts
E06 did not use) plus all 1,319 GSM8K test groups, tagged split train or test and keyed by (split, pid).
At cutoff 0.625 on the confident subset (agreement >= 0.625): PASS if pooled rate >= 0.03 AND pooled k >= 100,
KILL if pooled rate < 0.03, INCONCLUSIVE if rate >= 0.03 and k < 100. A non-zero exit code or a traceback is FAILED.
Truncated share above 0.10 is flagged, not a verdict change. The 0.75 row is descriptive. The claim 2 readout (L14b) is not run here.
Integrity checks (counts, prompt overlap, pre-registered L13a pool counts) are reported; any failed check gives FAILED, not a verdict.
Usage: /homes/rdubash/active-audit/.venv/bin/python avrl/exp/L14a.py"""
import json, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from e06_gate import wilson95

L13A = os.path.join(ROOT, "avrl", "runs", "L13a")
RUN = os.path.join(ROOT, "avrl", "runs", "L14a")
CUT, RATE_FLOOR, K_FLOOR, TRUNC_FLAG = 0.625, 0.03, 100, 0.10
N_TRAIN_POOL, N_TEST, G_SAMPLES, MAX_TOK = 4473, 1319, 8, 512
PREREG_TRAIN_POOL_CELL = (1541, 89)  # n, k at 0.625, recomputed from L13a groups before the L14a run (RESEARCH_LOG)


def cell(df, cut):
    c = df[df.agreement >= cut]
    n, k = int(len(c)), int(c.proxy_error.sum())
    lo, hi = wilson95(k, n)
    return {"n": n, "k": k, "rate": k / n if n else float("nan"), "wilson95": [lo, hi]}


g13 = pd.read_parquet(os.path.join(L13A, "groups.parquet"))
r13 = pd.read_parquet(os.path.join(L13A, "rollouts.parquet"))
g13 = g13[g13.pid >= 3000].assign(split="train")
r13 = r13[r13.pid >= 3000].assign(split="train")
g14 = pd.read_parquet(os.path.join(RUN, "groups.parquet"))
r14 = pd.read_parquet(os.path.join(RUN, "rollouts.parquet"))
test_src = pd.read_parquet(os.path.join(ROOT, "avrl", "data", "gsm8k_test.parquet"))
E06 = pd.read_parquet(os.path.join(ROOT, "avrl", "runs", "E06", "groups.parquet"))
exit_code = open(os.path.join(RUN, "exit_code")).read().strip()
tracebacks = open(os.path.join(RUN, "run.log"), errors="replace").read().count("Traceback")

G = pd.concat([g13, g14], ignore_index=True)
R = pd.concat([r13, r14], ignore_index=True)
primary = cell(G, CUT)
descriptive = cell(G, 0.75)
rate, k = primary["rate"], primary["k"]

conf_keys = G[G.agreement >= CUT][["split", "pid"]]
trunc_conf = float(R.merge(conf_keys, on=["split", "pid"]).truncated.mean())
trunc_all = float(R.truncated.mean())

checks = {
    "groups_train_pool": int(len(g13)) == N_TRAIN_POOL,
    "groups_test": int(len(g14)) == N_TEST,
    "rollouts_train_pool": int(len(r13)) == N_TRAIN_POOL * G_SAMPLES,
    "rollouts_test": int(len(r14)) == N_TEST * G_SAMPLES,
    "split_pid_unique": not bool(G.duplicated(["split", "pid"]).any()),
    "test_pids_complete": sorted(g14.pid.tolist()) == list(range(N_TEST)),
    "test_questions_match_parquet": set(g14.question) == set(test_src.question) and g14.question.nunique() == N_TEST,
    "no_question_overlap_train_test": len(set(g13.question) & set(g14.question)) == 0,
    "no_question_overlap_pool_E06": len(set(g13.question) & set(E06.question)) == 0,
    "train_part_matches_prereg_cell": (cell(g13, CUT)["n"], cell(g13, CUT)["k"]) == PREREG_TRAIN_POOL_CELL,
    "exit_code_zero": exit_code == "0",
    "no_traceback_in_run_log": tracebacks == 0,
}
structural_ok = all(checks.values())

if not structural_ok:
    verdict = "FAILED"
elif rate < RATE_FLOOR:
    verdict = "KILL"
elif k >= K_FLOOR:
    verdict = "PASS"
else:
    verdict = "INCONCLUSIVE"

out = {
    "experiment": "L14a", "policy": "Qwen2.5-0.5B-Instruct", "G": G_SAMPLES, "max_tokens": MAX_TOK, "seed": 0,
    "pool": {"train_part": "avrl/runs/L13a groups and rollouts with pid >= 3000 (split train)",
             "test_part": "avrl/data/gsm8k_test.parquet, all 1,319 prompts (split test)", "keyed_by": ["split", "pid"]},
    "verdict_at_0.625": verdict,
    "primary_cutoff_0.625": primary,
    "descriptive_cutoff_0.75": descriptive,
    "parts_at_0.625": {"train_part_L13a_pid_ge_3000": cell(g13, CUT), "test_part": cell(g14, CUT)},
    "truncated_rollouts": int(R.truncated.sum()), "truncated_share": trunc_all,
    "truncated_flag_above_0.10": bool(trunc_all > TRUNC_FLAG),
    "truncated_share_confident_subset": trunc_conf,
    "rollout_accuracy_pooled": float(R.correct.mean()),
    "majority_accuracy_pooled": float(G.majority_correct.mean()),
    "checks": {**{name: bool(ok) for name, ok in checks.items()}, "exit_code": exit_code, "tracebacks_in_run_log": tracebacks},
    "claim2_readout": "not run; pre-registered as L14b, runs only if verdict is PASS",
}
json.dump(out, open(os.path.join(RUN, "result.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
