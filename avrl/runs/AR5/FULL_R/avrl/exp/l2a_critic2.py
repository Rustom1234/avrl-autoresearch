"""L2a critic, part 2 (CPU): stress test of the fixed-family grid. The L2a fixed family stops at n = 400 (n <= budget // 2) and uses n in 25, 50, 100, 200, 400.
This script adds n in 75, 150, 300, 350, 550, 700 (n < budget) for recal / seeded / split charged on every (label set, split, budget in 25%, 50%) and writes
runs/L2a/critic_extra_fixed.json with per_seed hits for seeds 0..7. It first checks that its re-implementation reproduces saved hits for existing n.
Usage: l2a_critic2.py <runs/L2a> <runs/E04> [out json name] [comma list of n]"""
import sys, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
rd, ed = sys.argv[1], sys.argv[2]
OUTF = sys.argv[3] if len(sys.argv) > 3 else "critic_extra_fixed.json"
NAMES = {"3B": "Qwen2.5-3B-Instruct", "7B": "Qwen2.5-7B-Instruct-AWQ"}
cheap = "agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std".split(","); G = "cell100"; LABEL = "judge_error"; MIN = 3
NEW = tuple(int(v) for v in sys.argv[4].split(",")) if len(sys.argv) > 4 else (75, 150, 300, 350, 550, 700); out = {}; bad = 0; chk = 0
for lab, nm in NAMES.items():
    df0 = pd.read_parquet(f"{ed}/feats_{nm}.parquet")
    for split in range(4):
        df = df0.copy(); cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
        r0 = np.random.default_rng(split); groups = df[G].unique(); r0.shuffle(groups); fitg = set(groups[: len(groups) // 4]); sim = df[~df[G].isin(fitg)].reset_index(drop=True)
        Xs = sim[cheap].fillna(0).values; y = sim[LABEL].values.astype(int); agree = sim["agreement"].values; nd = sim["n_distinct"].values; N = len(y)
        saved = json.load(open(f"{rd}/sim_{nm}_split{split}.json")); assert saved["n_sim"] == N and saved["hits"] == int(y.sum())
        def charged(cal, fallback, budget):
            cal = np.asarray(cal)
            if y[cal].sum() == 0 or y[cal].sum() == len(cal): order = fallback
            else:
                sc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xs[cal], y[cal]).predict_proba(Xs)[:, 1]; sc[cal] = -1
                order = np.concatenate([cal, np.argsort(-sc)[: N - len(cal)]])
            return int(y[order[:budget]].sum())
        def pol(fam, n, seed, budget):
            r = np.random.default_rng(seed)
            if fam == "recal": perm = r.permutation(N); return charged(perm[:n], perm, budget)
            if fam == "seeded": ag = np.lexsort((r.random(N), agree)); return charged(ag[:n], ag, budget)
            pool = np.where(nd > 1)[0]; cal = r.choice(pool, min(n, len(pool)), replace=False); ag = np.lexsort((r.random(N), agree))
            return charged(cal, np.concatenate([cal, ag[~np.isin(ag, cal)]]), budget)
        for bf in (0.25, 0.5):
            budget = int(bf * N); key = f"{lab}|{split}|{bf}"; out[key] = {}
            for fam in ("recal", "seeded", "split"):
                for n in NEW:
                    if n < budget: out[key][f"{fam}_charged_{n}"] = [pol(fam, n, s, budget) for s in range(8)]
            # reproduction check on existing n (seeds 0..7)
            for fam, n in (("seeded", 400 if bf == 0.5 else 200), ("split", 200), ("recal", 100)):
                mine = [pol(fam, n, s, budget) for s in range(8)]; sv = saved[f"budget_{bf}"][f"{fam}_charged_{n}"]["per_seed"]; chk += 1; bad += (mine != sv)
        print(lab, split, "done", flush=True)
print(f"reproduction of saved per-seed hits for existing n: {chk} policy-cells checked, {bad} differ")
json.dump(out, open(f"{rd}/{OUTF}", "w"))
