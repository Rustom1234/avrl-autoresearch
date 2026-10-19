"""L4b critic: is the weak test-only result a size effect or a population effect? CPU only. Independent Pool code (l4b_critic_indep.py).
(a) Random halves of the full L4b pool (a random 1,319 prompts, same size as the test-only rows), cell100 buckets from the joint
    clustering, canonical splits 0 to 7, floor 40, B at 10% and A at 5%, 6 random halves.
(b) The same random-half control on the L2b pool (3,000 prompts) and the p0 (L1B) pool, 6 random halves of 1,319 prompts.
(c) Test-only rows re-bucketed by TF-IDF KMeans (100 clusters), a stand-in for re-running l2b_feats on test rows alone
    (the MiniLM model is not in a permitted directory), canonical splits 0 to 7, floor 40.
Usage: python -I l4b_critic_sizecontrol.py
"""
import sys
sys.path.insert(0, "/homes/rdubash/avrl_live/ROUTED/avrl/exp")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import l4b_critic_indep as C


def eight(sub, G="cell100", floor=40):
    ga, gb = [], []
    for s in range(8):
        d = C.build(sub, s, G=G)
        P = C.Pool(d); b5, b10 = int(0.05 * P.n), int(0.10 * P.n); a, b = [], []
        for sd in range(8):
            a.append(100 * (P.y[P.recal_set(b5, sd, 2, max(b5 // 8, floor), "x")[0]].sum() - P.y[P.fixed_set(b5, sd)].sum()) / b5)
            b.append(100 * (P.y[P.recal_set(b10, sd, 50, max(b10 // 8, floor), "xt")[0]].sum() - P.y[P.fixed_set(b10, sd)].sum()) / b10)
        ga.append(np.mean(a)); gb.append(np.mean(b))
    return float(np.mean(ga)), float(np.mean(gb))


F = pd.read_parquet(f"{C.RUN}/L4b/feats.parquet")
for name, path in (("L4b full", "L4b"), ("L2b", "L2b"), ("p0 (L1B)", "p0")):
    D = pd.read_parquet(f"{C.RUN}/{path}/feats.parquet"); res = []
    for h in range(6):
        idx = np.random.default_rng(900 + h).permutation(len(D))[:1319]
        res.append(eight(D.iloc[idx].reset_index(drop=True)))
    ra = [r[0] for r in res]; rb = [r[1] for r in res]
    print(f"[random 1,319-prompt halves of {name}] floor 40: A 5% means {[round(x, 2) for x in ra]} avg {np.mean(ra):+.2f}; B 10% means {[round(x, 2) for x in rb]} avg {np.mean(rb):+.2f}, above +2.0: {sum(x > 2 for x in rb)} of 6", flush=True)
T = F[F.pid < 1319].reset_index(drop=True)
X = TfidfVectorizer(min_df=2, stop_words="english").fit_transform(T.question)
T["tfidf100"] = KMeans(n_clusters=100, n_init=3, random_state=0).fit_predict(X)
print("test-only tfidf100 buckets with >= 3 docs", int((T.tfidf100.value_counts() >= 3).sum()), flush=True)
a, b = eight(T, G="tfidf100"); print(f"[test only, TF-IDF KMeans 100 buckets] floor 40: A 5% mean {a:+.2f}, B 10% mean {b:+.2f}", flush=True)
a, b = eight(T, G="tfidf100", floor=10); print(f"[test only, TF-IDF KMeans 100 buckets] floor 10: A 5% mean {a:+.2f}, B 10% mean {b:+.2f}", flush=True)
