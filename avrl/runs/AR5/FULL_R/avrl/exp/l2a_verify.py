"""L2a audit (CPU). Independent of l2a_summarize.py and l2a_sim_alloc.py policy code.
(1) Recompute the four headline ratios from the raw sim_*.json per_seed hits (selection on seeds 0-3, eval on 4-7).
(2) Re-implement greedy_n25_R25, greedy_seeded_n25_R25, seeded_charged_100, split_charged_100, recal_charged_100 label by label and compare with saved per_seed hits (3B split 0, 25% budget).
Usage: l2a_verify.py <runs/L2a>   (run with active-audit/.venv/bin/python, OMP/OPENBLAS threads 1)"""
import sys, json, os, re
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rd = sys.argv[1]
NAMES = {"3B": "Qwen2.5-3B-Instruct", "7B": "Qwen2.5-7B-Instruct-AWQ"}
# ---------- (1) headline ratios from raw json
rng = np.random.default_rng(123)
for lab, nm in NAMES.items():
    D = [json.load(open(f"{rd}/sim_{nm}_split{s}.json")) for s in range(4)]
    for b in ("0.25", "0.5"):
        pols = set(D[0][f"budget_{b}"])
        H = {p: np.array([d[f"budget_{b}"][p]["per_seed"] for d in D], float) for p in pols}  # (4 splits, 8 seeds)
        fixed = [p for p in pols if p == "thr_agreement" or re.match(r"(recal|seeded|split)_charged_\d+$", p)]
        learn = [p for p in pols if p.startswith("greedy")]
        bf = max(fixed, key=lambda p: H[p][:, :4].mean()); bl = max(learn, key=lambda p: H[p][:, :4].mean())
        L, F = H[bl][:, 4:], H[bf][:, 4:]
        sr = L.mean(1) / F.mean(1); cb = []
        for _ in range(10000):
            ix = rng.integers(0, 4, 4); cb.append(L[ix].mean() / F[ix].mean())
        # budget sanity: no policy can exceed budget hits; oracle <= budget
        print(f"{lab} {b}: fixed {bf} learned {bl} ratio {L.mean()/F.mean():.3f} cluster CI [{np.percentile(cb,2.5):.3f},{np.percentile(cb,97.5):.3f}] split ratios {np.round(sr,3).tolist()} "
              f"| n fixed cands {len(fixed)} learned cands {len(learn)} | learned best at grid corner: {bl}")
        # leave-one-split-out: does the ratio stay above 1.03 when any one split is dropped?
        lo = [L[[i for i in range(4) if i != j]].mean() / F[[i for i in range(4) if i != j]].mean() for j in range(4)]
        print(f"    leave-one-split-out ratios {np.round(lo,3).tolist()}")

# ---------- (2) independent re-implementation, 3B split 0, 25% budget
MIN = 3; SPLIT = 0
cheap = "agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std".split(",")
df = pd.read_parquet("runs/E04/feats_Qwen2.5-3B-Instruct.parquet"); G = "cell100"; LABEL = "judge_error"
cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
r0 = np.random.default_rng(SPLIT); groups = df[G].unique(); r0.shuffle(groups)
fitg = set(groups[: len(groups) // 4]); sim = df[~df[G].isin(fitg)].reset_index(drop=True)
Xs = sim[cheap].fillna(0).values; y = sim[LABEL].values.astype(int); agree = sim["agreement"].values; nd = sim["n_distinct"].values
N = len(y); budget = int(0.25 * N)
print(f"\nsim N {N}, budget {budget}, errors {y.sum()}")
def fitscore(idx, it): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=it, class_weight="balanced")).fit(Xs[idx], y[idx]).predict_proba(Xs)[:, 1]

def greedy(seed, n0, R, seeded):
    r = np.random.default_rng(seed)
    if seeded:
        ag = np.lexsort((r.random(N), agree)); sc = np.empty(N); sc[ag] = -np.arange(N, dtype=float)
    else: sc = r.random(N) * 1e-3
    seenmask = np.zeros(N, bool); hits = 0; nxt = n0
    for t in range(1, budget + 1):               # one label at a time (K = 1)
        cand = np.where(~seenmask)[0]; i = cand[np.argsort(-sc[cand])][0]
        seenmask[i] = True; hits += y[i]
        if t >= nxt and 0 < y[seenmask].sum() < seenmask.sum():   # first fit after n0 labels, then every R labels; retry each step while only one class has been seen
            sc = fitscore(np.where(seenmask)[0], 500); nxt += R
    return int(hits)

def charged(cal, rest_order):
    cal = np.asarray(cal)
    if y[cal].sum() == 0 or y[cal].sum() == len(cal): order = rest_order
    else:
        sc = fitscore(cal, 1000); sc[cal] = -1; order = np.concatenate([cal, np.argsort(-sc)[: N - len(cal)]])
    return int(y[order[:budget]].sum())

def recal(seed, n):
    r = np.random.default_rng(seed); perm = r.permutation(N); return charged(perm[:n], perm)
def seeded(seed, n):
    r = np.random.default_rng(seed); ag = np.lexsort((r.random(N), agree)); return charged(ag[:n], ag)
def split(seed, n):
    r = np.random.default_rng(seed); pool = np.where(nd > 1)[0]; cal = r.choice(pool, min(n, len(pool)), replace=False)
    ag = np.lexsort((r.random(N), agree)); return charged(cal, np.concatenate([cal, ag[~np.isin(ag, cal)]]))

d = json.load(open(f"{rd}/sim_Qwen2.5-3B-Instruct_split0.json"))["budget_0.25"]
bad = 0
for name, fn in (("greedy_n25_R25", lambda s: greedy(s, 25, 25, False)), ("greedy_seeded_n25_R25", lambda s: greedy(s, 25, 25, True)),
                 ("greedy_n100_R400", lambda s: greedy(s, 100, 400, False)),
                 ("seeded_charged_100", lambda s: seeded(s, 100)), ("split_charged_100", lambda s: split(s, 100)), ("recal_charged_100", lambda s: recal(s, 100))):
    mine = [fn(s) for s in range(8)]; saved = d[name]["per_seed"]; ok = mine == saved; bad += (not ok)
    print(f"{name}: independent {mine} saved {saved} {'MATCH' if ok else 'DIFF'}")
print("independent re-implementation:", "ALL MATCH" if bad == 0 else f"{bad} DIFFS")
