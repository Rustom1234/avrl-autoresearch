"""L2a critic, part 4 (CPU): symmetric stress test. Adds learned candidates with smaller first-fit size n0 and refit period R than the L2a grid corner (n0 = R = 25).
Reproduces greedy_n25_R25 first (must match the saved per-seed hits), then runs (n0, R) in the list for one (label set, split) at budgets 25% and 50%.
Usage: l2a_critic4.py <runs/L2a> <runs/E04> <3B|7B> <split> ; writes runs/L2a/critic_learned_<lab>_<split>.json"""
import sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
rd, ed, lab, split = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
nm = {"3B": "Qwen2.5-3B-Instruct", "7B": "Qwen2.5-7B-Instruct-AWQ"}[lab]
cheap = "agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std".split(","); G = "cell100"; LABEL = "judge_error"; MIN = 3
df = pd.read_parquet(f"{ed}/feats_{nm}.parquet"); cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
r0 = np.random.default_rng(split); groups = df[G].unique(); r0.shuffle(groups); fitg = set(groups[: len(groups) // 4]); sim = df[~df[G].isin(fitg)].reset_index(drop=True)
Xs = sim[cheap].fillna(0).values; y = sim[LABEL].values.astype(int); N = len(y)
saved = json.load(open(f"{rd}/sim_{nm}_split{split}.json"))
def greedy(budget, seed, n0, R):
    rng = np.random.default_rng(seed); score = rng.random(N) * 1e-3; seen = np.zeros(N, bool); hits = spent = 0; nxt = n0
    while spent < budget:
        mu = np.where(~seen)[0]; c = mu[np.argsort(-score[mu])][0]; seen[c] = True; hits += int(y[c]); spent += 1
        if spent >= nxt and 0 < y[seen].sum() < seen.sum():
            score = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")).fit(Xs[seen], y[seen]).predict_proba(Xs)[:, 1]; nxt += R
    return hits
out = {}
for bf in (0.25, 0.5):
    budget = int(bf * N); key = str(bf)
    if bf == 0.25:  # reproduction check (argmax with first-index ties equals argsort(-score)[0]; scores are continuous so ties do not occur)
        mine = [greedy(budget, s, 25, 25) for s in range(8)]; sv = saved["budget_0.25"]["greedy_n25_R25"]["per_seed"]; print(lab, split, "repro greedy_n25_R25:", "MATCH" if mine == sv else f"DIFF {mine} vs {sv}", flush=True)
    out[key] = {f"greedy_n{n0}_R{R}": [greedy(budget, s, n0, R) for s in range(8)] for n0, R in ((10, 10), (15, 15), (20, 20), (25, 10), (25, 5), (10, 5))}
json.dump(out, open(f"{rd}/critic_learned_{lab}_{split}.json", "w"))
print(lab, split, "done", flush=True)
