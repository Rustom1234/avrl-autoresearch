"""L6a independent audit (CPU). Usage: l6a_verify.py <runs/L6a>. (1) re-derives the verdict-fold headline per cell from the raw per-seed JSONs with its own bootstrap seed (424242) and compares to result.json;
(2) re-implements amort_seed / amort_rand / amort_split / freeze_plain from the feature tables in a different code structure for a few (cell, order seed) pairs and compares all 12 per-batch hits;
(3) checks amort_rand_n batch-1 hits == uniform batch-1 hits (same rng stream); (4) post hoc, extrapolative: steady-state ratio to carry_plain after calibration and a break-even horizon (stationary replay only)."""
import sys, os, json, math
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
rd = sys.argv[1]; R = json.load(open(os.path.join(rd, "result.json")))
J = {"3B": "Qwen2.5-3B-Instruct", "7B-AWQ": "Qwen2.5-7B-Instruct-AWQ"}; AM = [f"amort_{f}_{n}" for f in ("seed", "rand", "split") for n in (125, 250, 500)]
CJ = "agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std".split(",")
rng = np.random.default_rng(424242); IDX = rng.integers(0, 8, (10000, 8)); ok = True
def ld(j, B, o): return json.load(open(os.path.join(rd, f"amort_{j}_B{B}_o{o}.json")))
print("== (1) headline from raw JSONs")
for lab, j in J.items():
    for B in (62, 125):
        D = [ld(j, B, o)["arms"] for o in range(16)]; T = lambda k: np.array([sum(d[k]) for d in D], float)
        best = max(AM, key=lambda k: T(k)[:8].mean()); num, den = T("carry_plain")[8:], T(best)[8:]; rb = num[IDX].mean(1) / den[IDX].mean(1)
        c = R["cells"][f"{lab} B={B}"]["verdict_folds"]["A"]; d_ = abs(num.mean() / den.mean() - c["carry_plain_over_best"]["ratio"]); dl = abs(np.percentile(rb, 2.5) - c["carry_plain_over_best"]["ci_lo"])
        good = best == c["best_arm"] and d_ < 1e-9 and dl < 0.01; ok &= good
        print(f"{lab} B={B}: best {best} (summarizer {c['best_arm']}), carry_plain/best {num.mean()/den.mean():.4f} CI lo {np.percentile(rb,2.5):.4f} (summarizer {c['carry_plain_over_best']['ci_lo']:.4f}) {'OK' if good else 'MISMATCH'}")
print("== (2) independent re-implementation of the new arms")
fit = lambda X, y, it=1000: make_pipeline(StandardScaler(), LogisticRegression(max_iter=it, class_weight="balanced")).fit(X, y) if 0 < y.sum() < len(y) else None
def indep(kind, n, B, seed, X, y, ag, nd, perm):
    k = math.ceil(n / B); r = np.random.default_rng(seed); labs, hits, clf = [], [], None
    for bi in range(12):
        b = perm[bi * 250:(bi + 1) * 250]
        if bi >= k or (bi == k - 1 and kind == "seed" and False):
            pass
        if bi < k - 1 or (bi == k - 1 and kind == "seed"):
            if kind == "seed": pk = np.lexsort((np.random.default_rng(seed).random(250), ag[b]))[:B]
            elif kind == "rand": pk = r.permutation(250)[:B]
            else: pk = r.choice(np.flatnonzero(nd[b] > 1), B, replace=False)
            hits.append(int(y[b[pk]].sum())); labs += list(b[pk])
            if bi == k - 1: clf = fit(X[labs], y[labs])
        elif bi == k - 1:
            m = n - (k - 1) * B; pk = r.permutation(250)[:m] if kind == "rand" else r.choice(np.flatnonzero(nd[b] > 1), m, replace=False); labs += list(b[pk]); clf = fit(X[labs], y[labs])
            if clf is None:
                o = np.lexsort((np.random.default_rng(seed).random(250), ag[b])); rest = [i for i in o if i not in set(pk)]
            else:
                s = clf.predict_proba(X[b])[:, 1]; s[pk] = -1; rest = list(np.argsort(-s))
            hits.append(int(y[b[list(pk) + rest[:B - m]]].sum()))
        else:
            if clf is None: o = np.lexsort((np.random.default_rng(seed).random(250), ag[b])); hits.append(int(y[b[o[:B]]].sum()))
            else: hits.append(int(y[b[np.argsort(-clf.predict_proba(X[b])[:, 1])[:B]]].sum()))
    return hits
for lab, B, o in (("3B", 125, 5), ("3B", 62, 3), ("7B-AWQ", 62, 11), ("7B-AWQ", 125, 14)):
    df = pd.read_parquet(f"/homes/rdubash/avrl_live/FULL_R/avrl/runs/E04/feats_{J[lab]}.parquet"); X = df[CJ].fillna(0).values; y = df["judge_error"].values.astype(int); ag = df["agreement"].values; nd = df["n_distinct"].values
    perm = np.random.default_rng(10000 + o).permutation(len(df)); d = ld(J[lab], B, o)
    for kind in ("seed", "rand", "split"):
        for n in (125, 250, 500):
            h = indep(kind, n, B, o, X, y, ag, nd, perm); g = d["arms"][f"amort_{kind}_{n}"]; ok &= (h == g)
            if h != g: print("MISMATCH", lab, B, o, kind, n, h, g)
    # freeze_plain re-derivation: replay carry_plain's labels from scratch (independent loop), freeze at k*B labels
    L, sc_hits = [], []
    for bi in range(12):
        b = perm[bi * 250:(bi + 1) * 250]; r = np.random.default_rng(o); score = r.random(250) * 1e-3; seen = np.zeros(250, bool); hh = 0; fitted = -1
        for _ in range(B):
            if len(L) != fitted and len(L) >= 3 and 0 < y[L].sum() < len(L): score = fit(X[L], y[L], 500).predict_proba(X[b])[:, 1]; fitted = len(L)
            c = np.flatnonzero(~seen)[np.argmax(score[~seen])]; seen[c] = True; hh += y[b[c]]; L.append(int(b[c]))
        sc_hits.append(int(hh))
    ok &= (sc_hits == d["arms"]["carry_plain"])
    for n in (125, 250, 500):
        k = math.ceil(n / B); clf = fit(X[L[:k * B]], y[L[:k * B]], 500); h = sc_hits[:k] + [int(y[perm[bi * 250:(bi + 1) * 250]][np.argsort(-clf.predict_proba(X[perm[bi * 250:(bi + 1) * 250]])[:, 1])[:B]].sum()) for bi in range(k, 12)]
        ok &= (h == d["arms"][f"freeze_plain_{n}"])
        if h != d["arms"][f"freeze_plain_{n}"]: print("MISMATCH freeze", lab, B, o, n)
    print(f"{lab} B={B} order {o}: carry_plain + 9 amort + 3 freeze re-derived, equal = {ok}")
print("== (3) amort_rand_n batch 1 == uniform batch 1")
n_bad = 0
for lab, j in J.items():
    for B in (62, 125):
        for o in range(16):
            a = ld(j, B, o)["arms"]; n_bad += sum(a[f"amort_rand_{n}"][0] != a["uniform"][0] for n in (125, 250, 500))
print("violations:", n_bad); ok &= n_bad == 0
print("== (4) post hoc, extrapolative (stationary replay): after-calibration ratio to carry_plain, and break-even horizon")
print("cell | arm | steady (batches max(k+1,7)..12) arm/carry_plain | deficit vs carry_plain at batch 12 (errors, mean per seed) | per-batch gain in steady state | batches to break even")
for lab, j in J.items():
    for B in (62, 125):
        D = [ld(j, B, o)["arms"] for o in range(16)]; M = lambda k: np.array([d[k] for d in D], float)
        for kind in ("rand", "split"):
            for n in (125, 250, 500):
                k = math.ceil(n / B); s0 = max(k, 6); a = M(f"amort_{kind}_{n}"); c = M("carry_plain"); deficit = (c - a).sum(1).mean(); g = (a[:, s0:] - c[:, s0:]).sum(1).mean() / (12 - s0)
                print(f"{lab} B={B} | amort_{kind}_{n} | {a[:, s0:].sum() / c[:, s0:].sum():.3f} | {deficit:.1f} | {g:+.2f} | " + (f"{deficit / g:.0f}" if g > 0 else "never (carry_plain still ahead)"))
print("== (5) decomposition of the 12-batch deficit vs carry_plain for the verdict-fold best arm (chosen on seeds 0-7), reported on seeds 8-15, mean errors per seed")
print("cell | arm | k | deficit total | during calibration batches 1..k | after calibration batches k+1..12 | carry_plain advantage over thr in batches 1..k (errors)")
for lab, j in J.items():
    for B in (62, 125):
        D = [ld(j, B, o)["arms"] for o in range(16)]; M = lambda k: np.array([d[k] for d in D], float); best = R["cells"][f"{lab} B={B}"]["verdict_folds"]["A"]["best_arm"]; n = int(best.rsplit("_", 1)[1]); k = math.ceil(n / B)
        a, c, t = M(best)[8:], M("carry_plain")[8:], M("thr_agreement")[8:]
        print(f"{lab} B={B} | {best} | {k} | {(c - a).sum(1).mean():.1f} | {(c - a)[:, :k].sum(1).mean():.1f} | {(c - a)[:, k:].sum(1).mean():.1f} | {(c - t)[:, :k].sum(1).mean():+.1f}")
print("ALL CHECKS", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
