"""L7b CPU step: split-group pick accuracy of the judge (metric a of the L7b pre-registration). CPU only.

Reads avrl/runs/L7b/judge_scores.parquet, avrl/runs/p0/rollouts.parquet (answer, correct) and avrl/runs/p0/groups.parquet (agreement).
Pick per group = distinct answer with the highest mean p_yes over its rollouts; exact ties go to the lower answer string.
The pick is correct iff the picked answer is correct (rollouts.correct, checked constant within each answer).
Split groups = agreement < 1 (non-unanimous). Writes avrl/runs/L7b/pick_accuracy.json.
"""
import json, os
import numpy as np, pandas as pd

ROOT = "/homes/rdubash/avrl_live/CASCADE"
os.chdir(ROOT)
RUN = "avrl/runs/L7b"

S = pd.read_parquet(f"{RUN}/judge_scores.parquet")
R = pd.read_parquet("avrl/runs/p0/rollouts.parquet")
G = pd.read_parquet("avrl/runs/p0/groups.parquet")[["pid", "agreement"]]

T = S.merge(R[["pid", "k", "answer", "correct"]], on=["pid", "k"], suffixes=("_s", "_r"), validate="one_to_one")
assert len(T) == len(S) == len(R) and (T.answer_s.astype(str) == T.answer_r.astype(str)).all(), "judge rows do not align with rollouts"
A = (T.assign(answer=T.answer_s.astype(str)).groupby(["pid", "answer"])
       .agg(mp=("p_yes", "mean"), c_max=("correct", "max"), c_min=("correct", "min")).reset_index())
assert (A.c_max == A.c_min).all(), "correctness not constant within answer"

top = A.sort_values(["pid", "mp", "answer"], ascending=[True, False, True]).groupby("pid", as_index=False).head(1)
uni = A.groupby("pid").c_max.mean().rename("uniform")
P = (top.set_index("pid")[["c_max"]].rename(columns={"c_max": "pick_correct"})
        .join(uni).join(G.set_index("pid")))
assert len(P) == G.pid.nunique() == 3000, "group count differs from p0"

split = P[P.agreement < 1]
n = len(split)
p = float(split.pick_correct.mean())
d = split.pick_correct - split.uniform
res = {
    "n_groups": int(len(P)), "n_split_groups": int(n), "n_unanimous_groups": int((P.agreement == 1).sum()),
    "pick_accuracy_split": p,
    "pick_accuracy_split_se_binomial": float(np.sqrt(p * (1 - p) / n)),
    "uniform_over_distinct_split": float(split.uniform.mean()),
    "paired_diff_pick_minus_uniform_split": float(d.mean()),
    "paired_diff_se_split": float(d.std(ddof=1) / np.sqrt(n)),
    "pick_accuracy_all_groups": float(P.pick_correct.mean()),
    "thresholds": {"pass_at_least": 0.397, "kill_at_most": 0.347, "uniform_reference": 0.297},
}
json.dump(res, open(os.path.join(RUN, "pick_accuracy.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
