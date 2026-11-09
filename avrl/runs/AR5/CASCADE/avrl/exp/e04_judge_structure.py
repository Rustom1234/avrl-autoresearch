"""E04 CPU step: structure of the judge-error label on the p0 GSM8K rollouts (CPU only).

Inputs: avrl/runs/E04/judge_scores.parquet (GPU step) and avrl/runs/p0/feats.parquet (written by p0_analyze.py; it holds the
6 cheap columns, the 16 MiniLM PCA columns and cell30/100/300, so they are read here, not recomputed).
Judge pick per group = the distinct answer with the highest mean p_yes over its rollouts (ties go to the lower answer string).
judge_error = judge pick != gold. proxy_error stays the E01 majority label.

Outputs in --out: feats_judge.parquet (feats plus judge_pick and judge_error, the input of the claim-2 replay) and
structure.json (gates, judge error rates, AUCs, neighbor ratio, Jaccard, structure verdict).
The AUC code is p0_analyze.py's: 5-fold cross_val_score, balanced logistic regression on standardized columns.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = "/homes/rdubash/avrl_live/CASCADE"
sys.path.insert(0, ROOT)
from aa import data as DATA

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
ap.add_argument("--feats", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

S = pd.read_parquet(a.scores)
F = pd.read_parquet(a.feats)
assert not S.duplicated(["pid", "k"]).any(), "duplicate (pid, k) in judge scores"
assert set(S.pid) == set(F.pid) and len(F) == S.pid.nunique(), "pid sets differ between scores and feats"

# Judge pick: sort by p_yes descending, then answer ascending, keep the first row per group.
M = S.groupby(["pid", "answer"], as_index=False).p_yes.mean()
top = M.sort_values(["pid", "p_yes", "answer"], ascending=[True, False, True]).groupby("pid", as_index=False).head(1)
F = F.copy()
F["judge_pick"] = F.pid.map(top.set_index("pid").answer)
F["judge_error"] = (F.judge_pick != F.gold).astype(int)
F.to_parquet(os.path.join(a.out, "feats_judge.parquet"), index=False)

cheap = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
embc = [f"pc{i}" for i in range(16)]
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))


def auc(df, cols, y):
    return float(cross_val_score(LR(), df[cols].values, y, cv=5, scoring="roc_auc").mean())


def conc_nr(y, g):
    return float(DATA.neighbor_hit_ratio(y, g)["ratio"])


py = S.p_yes.values
sd = float(py.std(ddof=1))
pick_eq_major = float((F.judge_pick == F.majority).mean())
constant = sd < 0.02

yj, yp = F.judge_error.values, F.proxy_error.values
un = (F.agreement == 1).values
conf = F[F.agreement >= 0.75]
A = set(F.pid[F.proxy_error == 1]); B = set(F.pid[F.judge_error == 1])
jac = len(A & B) / len(A | B)
auc_c, auc_ce = auc(F, cheap, yj), auc(F, cheap + embc, yj)
gain = auc_ce - auc_c
nr100 = conc_nr(yj, F.cell100.values)

if constant:
    verdict = "FAILED_PROXY (constant judge, p_yes sd < 0.02; no verdict on proxy type)"
elif gain >= 0.05 and jac < 0.5:
    verdict = "PASS"
elif jac > 0.8 or (gain < 0.05 and nr100 < 1.5):
    verdict = "KILL"
else:
    verdict = "INCONCLUSIVE"

res = {
    "gates": {"p_yes_sd_all_rows": sd, "constant_judge": bool(constant), "n_rows": int(len(S)),
              "frac_groups_judge_pick_equals_majority": pick_eq_major},
    "sizes": {"n_groups": int(len(F)), "n_confident": int(len(conf)), "n_unanimous": int(un.sum()), "n_split": int((~un).sum())},
    "judge_error": {"rate_all": float(yj.mean()), "rate_confident_agreement_ge_0.75": float(conf.judge_error.mean()),
                    "rate_unanimous": float(yj[un].mean()), "rate_split": float(yj[~un].mean()), "n_errors": int(yj.sum())},
    "proxy_error_reference": {"rate_all": float(yp.mean()), "rate_unanimous": float(yp[un].mean()), "rate_split": float(yp[~un].mean()),
                              "n_errors": int(yp.sum()), "auc_cheap_recomputed": auc(F, cheap, yp), "auc_cheap_emb_recomputed": auc(F, cheap + embc, yp)},
    "auc_judge_error": {"cheap": auc_c, "cheap_emb": auc_ce, "gain_cheap_emb_minus_cheap": gain},
    "neighbor_ratio_judge_error_cell100": nr100,
    "overlap": {"proxy_error_groups": len(A), "judge_error_groups": len(B), "both": len(A & B),
                "proxy_only": len(A - B), "judge_only": len(B - A), "jaccard": jac},
    "structure_verdict": verdict,
    "rules": {"pass": "gain >= 0.05 AND Jaccard < 0.5", "kill": "Jaccard > 0.8 OR (gain < 0.05 AND cell100 neighbor ratio < 1.5)",
              "constant_judge": "p_yes sd < 0.02 is a failed proxy"},
}
json.dump(res, open(os.path.join(a.out, "structure.json"), "w"), indent=1)
print(json.dumps(res, indent=1), flush=True)
