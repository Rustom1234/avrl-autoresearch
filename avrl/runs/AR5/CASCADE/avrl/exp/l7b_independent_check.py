"""L7b independent check (CPU only): recomputes the headline L7b numbers directly from the raw parquets,
without using any L7b pipeline module. Reads avrl/runs/p0/rollouts.parquet, avrl/runs/p0/groups.parquet and
avrl/runs/L7b/judge_scores.parquet. Writes avrl/runs/L7b/independent_check.json.

Pick rule (pre-registered): per group, the distinct answer with the highest mean p_yes, scored against rollouts.correct.
Split groups are the groups with more than one distinct answer. Judge error is a group where the pick is not correct.
Proxy error is a group where the majority answer is not correct (groups.proxy_error).
"""
import json, os
import pandas as pd

ROOT = "/homes/rdubash/avrl_live/CASCADE"
os.chdir(ROOT)
OUT = "avrl/runs/L7b/independent_check.json"

R = pd.read_parquet("avrl/runs/p0/rollouts.parquet")
G = pd.read_parquet("avrl/runs/p0/groups.parquet")
S = pd.read_parquet("avrl/runs/L7b/judge_scores.parquet")

# join judge scores onto rollouts by (pid, k), and check the alignment is one-to-one
M = R.merge(S[["pid", "k", "answer", "p_yes"]], on=["pid", "k"], how="outer", suffixes=("", "_s"), indicator=True)
assert (M["_merge"] == "both").all(), "rollout and judge rows do not align one-to-one"
assert (M["answer"] == M["answer_s"]).all(), "answer mismatch between rollouts and judge scores"

rows = []
for pid, g in M.groupby("pid", sort=True):
    per_ans = g.groupby("answer").agg(mean_p=("p_yes", "mean"), correct=("correct", "max"), correct_min=("correct", "min"))
    assert (per_ans["correct"] == per_ans["correct_min"]).all(), f"correctness not constant within answer, pid {pid}"
    n_distinct = len(per_ans)
    pick_ans = per_ans["mean_p"].idxmax()
    rows.append(dict(pid=int(pid), n_distinct=n_distinct,
                     pick_correct=int(per_ans.loc[pick_ans, "correct"]),
                     uniform_correct=float(per_ans["correct"].mean()),
                     judge_error=int(per_ans.loc[pick_ans, "correct"] == 0)))
P = pd.DataFrame(rows)
P = P.merge(G[["pid", "proxy_error", "majority_correct"]], on="pid", how="left")

split = P[P.n_distinct > 1]
unan = P[P.n_distinct == 1]
out = {
    "n_groups": int(len(P)),
    "n_split_groups": int(len(split)),
    "n_unanimous_groups": int(len(unan)),
    "judge_error_groups_all": int(P.judge_error.sum()),
    "judge_error_rate_all": float(P.judge_error.mean()),
    "proxy_error_groups_all": int(P.proxy_error.sum()),
    "pick_accuracy_split": float(split.pick_correct.mean()),
    "uniform_over_distinct_split": float(split.uniform_correct.mean()),
    "paired_diff_pick_minus_uniform_split": float((split.pick_correct - split.uniform_correct).mean()),
    "pick_accuracy_all_groups": float(P.pick_correct.mean()),
    "judge_error_unanimous_groups": int(unan.judge_error.sum()),
}
# judge error and proxy error overlap (Jaccard over group sets)
J = P[["pid", "judge_error", "proxy_error"]]
both = int(((J.judge_error == 1) & (J.proxy_error == 1)).sum())
jud_only = int(((J.judge_error == 1) & (J.proxy_error == 0)).sum())
prox_only = int(((J.judge_error == 0) & (J.proxy_error == 1)).sum())
out["overlap"] = dict(both=both, proxy_only=prox_only, judge_only=jud_only,
                      jaccard=both / (both + jud_only + prox_only) if (both + jud_only + prox_only) else None)
# pick equals majority: the judge pick is correct exactly when the majority answer is correct on the same group
# (only used as a descriptive cross-check; it is not a verdict input)
out["n_split_pick_correct_and_majority_correct"] = int(((split.pick_correct == 1) & (split.majority_correct == 1)).sum())

with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
