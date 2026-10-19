"""E04 verification: independent recomputation of the headline numbers in avrl/runs/E04 (CPU only).

Reads avrl/runs/E04/judge_scores.parquet, avrl/runs/p0/rollouts.parquet, avrl/runs/E04/feats_judge.parquet.
Writes avrl/runs/E04/verify/verify.json and prints it.

Differences from e04_judge_structure.py, on purpose: the judge pick is a plain loop over groups (not a pandas
sort), and judge correctness is read from rollouts.correct (not from a gold string comparison). Both must agree
with the run's own judge_error column.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = "/homes/rdubash/avrl_live/CASCADE"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from aa import data as DATA  # noqa: E402

OUT = os.path.join(ROOT, "avrl/runs/E04/verify")
os.makedirs(OUT, exist_ok=True)

S = pd.read_parquet("avrl/runs/E04/judge_scores.parquet")
R = pd.read_parquet("avrl/runs/p0/rollouts.parquet")
F = pd.read_parquet("avrl/runs/E04/feats_judge.parquet")

res = {"n_scores": int(len(S)), "n_groups": int(F.pid.nunique())}
res["fallbacks"] = {"fb_yes": int(S.fb_yes.sum()), "fb_no": int(S.fb_no.sum())}

# 1. every scored (pid, k) is a rollout with the same answer string
M = S.merge(R[["pid", "k", "answer", "correct"]], on=["pid", "k"], how="outer", suffixes=("_s", "_r"), indicator=True)
res["align"] = {"both": int((M._merge == "both").sum()),
                "left_only": int((M._merge == "left_only").sum()),
                "right_only": int((M._merge == "right_only").sum()),
                "answer_mismatch": int((M.answer_s.astype(str) != M.answer_r.astype(str)).sum())}

# 2. gate: spread of p_yes over all rows
py = S.p_yes.values.astype(float)
res["p_yes"] = {"sd_ddof1": float(py.std(ddof=1)), "mean": float(py.mean()), "nan": int(np.isnan(py).sum())}

# 3. judge pick by plain loop: mean p_yes per distinct answer, max wins, exact ties go to the lowest answer string
T = S.assign(answer=S.answer.astype(str)).merge(R[["pid", "k", "correct"]], on=["pid", "k"])
A = (T.groupby(["pid", "answer"])
     .agg(mp=("p_yes", "mean"), c_max=("correct", "max"), c_min=("correct", "min"))
     .reset_index())
res["correct_constant_within_answer"] = bool((A.c_max == A.c_min).all())
pick, corr_of, gap = {}, {}, {}
for pid, grp in A.groupby("pid"):
    mx = grp.mp.max()
    pick[pid] = sorted(grp.answer[grp.mp == mx])[0]
    srt = np.sort(grp.mp.values)[::-1]
    gap[pid] = float(srt[0] - srt[1]) if len(srt) > 1 else float("nan")
    for ans, c in zip(grp.answer, grp.c_max):
        corr_of[(pid, ans)] = int(bool(c))

pk = F.pid.map(pick)
res["pick_mismatch_vs_feats_judge_pick"] = int((pk.astype(str).values != F.judge_pick.astype(str).values).sum())
je = 1 - np.array([corr_of[(p, a)] for p, a in zip(F.pid, pk)], dtype=int)
res["judge_error_mismatch_vs_feats_judge_error"] = int((je != F.judge_error.values.astype(int)).sum())
gold_err = (F.judge_pick.astype(str).values != F.gold.astype(str).values).astype(int)
res["judge_error_gold_string_vs_rollout_correct_mismatch"] = int((gold_err != je).sum())

# 4. error rates
pe = F.proxy_error.values.astype(int)
un = F.agreement.values == 1
conf = F.agreement.values >= 0.75
res["n_unanimous"] = int(un.sum())
res["n_confident"] = int(conf.sum())
res["unanimous_equals_n_distinct_1"] = bool(((F.n_distinct.values == 1) == un).all())
res["rates"] = {
    "judge_all": float(je.mean()), "judge_n_errors": int(je.sum()),
    "judge_confident": float(je[conf].mean()),
    "judge_unanimous": float(je[un].mean()), "judge_split": float(je[~un].mean()),
    "proxy_all": float(pe.mean()), "proxy_n_errors": int(pe.sum()),
    "proxy_unanimous": float(pe[un].mean()), "proxy_split": float(pe[~un].mean()),
}
res["majority_correct_consistent_with_proxy_error"] = bool((F.majority_correct.astype(int).values == 1 - pe).all())
res["frac_pick_equals_majority"] = float((F.judge_pick.astype(str).values == F.majority.astype(str).values).mean())
res["accuracy"] = {"judge_pick": float(1 - je.mean()),
                   "random_rollout": float(R.correct.astype(float).mean()),
                   "majority": float(1 - pe.mean())}

# 5. post-hoc: judge wrong rate within split groups by top-two gap (descriptive only)
gp = F.pid.map(gap).values.astype(float)
sp = ~un
res["split_judge_wrong_by_top2_gap"] = {
    "gap_lt_0.01": float(je[sp & (gp < 0.01)].mean()), "n_lt": int((sp & (gp < 0.01)).sum()),
    "gap_ge_0.05": float(je[sp & (gp >= 0.05)].mean()), "n_ge": int((sp & (gp >= 0.05)).sum()),
}

# 5b. post-hoc chance reference inside split groups: judge pick vs a uniform pick over distinct answers,
#     and vs a uniform pick over rollouts (descriptive only)
agree = dict(zip(F.pid, F.agreement.values))
uni = [float(grp.c_max.astype(float).mean()) for pid, grp in A.groupby("pid") if agree[pid] < 1]
Rs = R[R.pid.map(agree) < 1]
res["split_correct_rates"] = {"judge_pick": float(1 - je[sp].mean()),
                              "uniform_over_distinct_answers": float(np.mean(uni)),
                              "uniform_over_rollouts": float(Rs.correct.astype(float).mean()),
                              "majority": float(1 - pe[sp].mean())}

# 6. rollout-level AUC of p_yes against rollout correctness (post-hoc)
Q = S.merge(R[["pid", "k", "correct"]], on=["pid", "k"])
res["rollout_auc_p_yes"] = float(roc_auc_score(Q.correct.astype(int), Q.p_yes.astype(float)))

# 7. structure: 5-fold balanced logistic AUC, same configuration as p0_analyze.py / e04_judge_structure.py
cheap = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
embc = [f"pc{i}" for i in range(16)]


def LR():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))


def auc(cols, y):
    return float(cross_val_score(LR(), F[cols].values, y, cv=5, scoring="roc_auc").mean())


res["auc_judge_error"] = {"cheap": auc(cheap, je), "cheap_emb": auc(cheap + embc, je)}
res["auc_judge_error"]["gain"] = res["auc_judge_error"]["cheap_emb"] - res["auc_judge_error"]["cheap"]
res["auc_proxy_error"] = {"cheap": auc(cheap, pe), "cheap_emb": auc(cheap + embc, pe)}

# 8. neighbor ratio at cell100 (proxy check reproduces E01's cell100 value of 1.03 if the helper is unchanged)
res["neighbor_ratio_cell100"] = {
    "judge_error": float(DATA.neighbor_hit_ratio(je, F.cell100.values)["ratio"]),
    "proxy_error_E01_check": float(DATA.neighbor_hit_ratio(pe, F.cell100.values)["ratio"]),
}

# 9. overlap of the two error sets
Ap, Bj = set(F.pid[pe == 1]), set(F.pid[je == 1])
res["overlap"] = {"both": len(Ap & Bj), "proxy_only": len(Ap - Bj), "judge_only": len(Bj - Ap),
                  "jaccard": len(Ap & Bj) / len(Ap | Bj)}

with open(os.path.join(OUT, "verify.json"), "w") as f:
    json.dump(res, f, indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
