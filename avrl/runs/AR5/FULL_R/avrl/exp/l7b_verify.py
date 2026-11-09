"""L7b verification (CPU, run from avrl/): Usage: l7b_verify.py <runs/L7b> <runs/L6a>
(1) re-derives the regression gate from the raw JSONs (separate code from the summarizer); (2) recomputes the verdict-fold headline per cell from the raw JSONs with a different bootstrap seed (7) and compares it to result.json;
(3) re-implements amort_strat_n and amort_bnd_n from the spec in separate code (own pick logic, own fit and top-B loop) and compares per-batch hits to the job JSONs for several (judge, B, order seed);
(4) loop-level post hoc diagnostic, not in the verdict: held-out AUC and agreement coefficient of the n-label fits of the strat / bnd / rand sets (rule fitted exactly as the arms fit it)."""
import sys, os, json, glob
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

rd, r6 = sys.argv[1], sys.argv[2]; res = json.load(open(os.path.join(rd, "result.json")))
J = {"3B": "Qwen2.5-3B-Instruct", "7B-AWQ": "Qwen2.5-7B-Instruct-AWQ"}; BUDS = (62, 125); CJ = "agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std".split(",")
ld = lambda p: json.load(open(p)); DES = [f"amort_{f}_{n}" for f in ("strat", "bnd") for n in (125, 250, 500)]
# ---- (1) gate from raw files
nvals = ndiff = nvec = 0; nfiles = 0
for lab, j in J.items():
    for B in BUDS:
        for o in range(16):
            a, b = ld(f"{rd}/design_{j}_B{B}_o{o}.json"), ld(f"{r6}/amort_{j}_B{B}_o{o}.json"); nfiles += 1
            for k in b["arms"]:
                nvec += 1; nvals += len(b["arms"][k]); ndiff += sum(x != y for x, y in zip(a["arms"][k], b["arms"][k])) + abs(len(a["arms"][k]) - len(b["arms"][k]))
            nvec += 1; nvals += 12; ndiff += sum(x != y for x, y in zip(a["batch_base_errors"], b["batch_base_errors"]))
print(f"(1) gate re-derived over ALL L6a arms ({nfiles} file pairs): {nvec} vectors, {nvals} per-batch values, {ndiff} differences")
# ---- (2) headline with another bootstrap seed
rng = np.random.default_rng(7); IDX = {n: rng.integers(0, n, (10000, n)) for n in (8, 16)}
def ratio(num, den): i = IDX[len(num)]; r = num[i].mean(1) / den[i].mean(1); return num.mean() / den.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)
ok = True
for lab, j in J.items():
    for B in BUDS:
        A = {}
        for o in range(16):
            for k, v in ld(f"{rd}/design_{j}_B{B}_o{o}.json")["arms"].items(): A.setdefault(k, []).append(v)
        A = {k: np.array(v, float) for k, v in A.items()}; T = lambda k, s: A[k][s].sum(1)
        bA = max(DES, key=lambda k: T(k, slice(0, 8)).mean()); c = res["cells"][f"{lab} B={B}"]; f = c["verdict_folds"]["A"]
        r, lo, hi = ratio(T("carry_plain", slice(8, 16)), T(bA, slice(8, 16)))
        same = (bA in f["best_arm"]) and abs(r - f["carry_plain_over_best"]["ratio"]) < 1e-12 and abs(lo - f["carry_plain_over_best"]["ci_lo"]) < 0.01
        ok &= same; print(f"(2) {lab} B={B}: best {bA}, carry_plain/best {r:.3f} (CI seed 7: {lo:.3f} to {hi:.3f}); result.json {f['carry_plain_over_best']['ratio']:.3f} (CI lo {f['carry_plain_over_best']['ci_lo']:.3f}); match={same}")
        cp, bt = A["carry_plain"], A[bA]; print(f"      raw: sum carry_plain seeds 8-15 {cp[8:].sum():.0f}, best {bt[8:].sum():.0f}; per-seed wins of carry_plain over best (8-15): {int((cp[8:].sum(1) > bt[8:].sum(1)).sum())} of 8; all 16 seeds, carry_plain/best per seed min {(cp.sum(1) / bt.sum(1)).min():.3f}")
print("(2) all cells match:", ok)
# ---- (3) independent re-implementation of the two new arms
def fit(Xa, ya): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xa, ya)
def indep(kind, n, B, seed, X, y, ag, perm):
    k = -(-n // B); rng = np.random.default_rng(seed); lab = []; clf = None; out = []
    for bi in range(12):
        b = perm[bi * 250:(bi + 1) * 250]; a = ag[b]
        if bi <= k - 1:
            take = B if bi < k - 1 else n - (k - 1) * B
            if kind == "strat":
                order = np.lexsort((rng.random(250), a)); tenths = [order[i * 25:(i + 1) * 25] for i in range(10)]; cnt = [take // 10] * 10
                for e in rng.choice(10, take % 10, replace=False): cnt[e] += 1
                pick = np.concatenate([rng.choice(t, c, replace=False) for t, c in zip(tenths, cnt)])
            else:
                order = np.lexsort((np.random.default_rng(seed).random(250), a)); lo = B - take // 2; pick = order[lo:lo + take]
            assert len(set(pick.tolist())) == take
            lab += b[pick].tolist()
            if bi < k - 1: out.append(int(y[b[pick]].sum())); continue
            assert len(lab) == n and 0 < y[lab].sum() < n; clf = fit(X[lab], y[lab]); sc = clf.predict_proba(X[b])[:, 1]; sc[pick] = -1
            rest = np.argsort(-sc)[:B - take]; out.append(int(y[b[pick]].sum() + y[b[rest]].sum()))
        else: out.append(int(y[b[np.argsort(-clf.predict_proba(X[b])[:, 1])[:B]]].sum()))
    return out, np.array(lab), clf
bad = tot = 0
for lab_, j in J.items():
    df = pd.read_parquet(f"runs/E04/feats_{j}.parquet"); X = df[CJ].fillna(0).values; y = df["judge_error"].values.astype(int); ag = df["agreement"].values
    for B in BUDS:
        for o in (0, 3, 9, 15):
            d = ld(f"{rd}/design_{j}_B{B}_o{o}.json"); perm = np.random.default_rng(10000 + o).permutation(len(df))
            for kind in ("strat", "bnd"):
                for n in (125, 250, 500):
                    h, _, _ = indep(kind, n, B, o, X, y, ag, perm); tot += 1; bad += h != d["arms"][f"amort_{kind}_{n}"]
print(f"(3) independent re-implementation of amort_strat_n / amort_bnd_n: {tot} arm vectors (2 judges x 2 B x 4 order seeds x 6 arms) compared, {bad} differ")
# ---- (4) diagnostic: held-out AUC and standardized agreement coefficient of the fit sets (3B and 7B-AWQ, both B, 16 seeds)
print("(4) post hoc diagnostic: rule fitted on the n labels of each design, scored on all 3,000 prompts minus the fit set (mean over 16 order seeds)")
print("    cell | arm | held-out AUC | standardized agreement coefficient (neg = lower agreement scores higher)")
for lab_, j in J.items():
    df = pd.read_parquet(f"runs/E04/feats_{j}.parquet"); X = df[CJ].fillna(0).values; y = df["judge_error"].values.astype(int); ag = df["agreement"].values
    for B in BUDS:
        for n in (125, 250, 500):
            row = {}
            for kind in ("strat", "bnd"):
                aucs, cs = [], []
                for o in range(16):
                    perm = np.random.default_rng(10000 + o).permutation(len(df)); _, L, clf = indep(kind, n, B, o, X, y, ag, perm); rest = np.setdiff1d(np.arange(len(df)), L)
                    aucs.append(roc_auc_score(y[rest], clf.predict_proba(X[rest])[:, 1])); cs.append(clf[-1].coef_[0][0])
                row[kind] = (np.mean(aucs), np.mean(cs))
            print(f"    {lab_} B={B} | n={n} | strat AUC {row['strat'][0]:.3f} coef {row['strat'][1]:+.2f} | bnd AUC {row['bnd'][0]:.3f} coef {row['bnd'][1]:+.2f}")
# ---- (5) diagnostic: what the frozen rule selects after calibration (n = 125 fits; batches k+1..12; mean over 16 order seeds)
print("(5) post hoc diagnostic: the top-B prompts each frozen n=125 rule picks in its post-calibration batches (share of picks with agreement >= 0.875, mean global z-agreement of picks, hit rate; uniform hit rate = base rate)")
for lab_, j in J.items():
    df = pd.read_parquet(f"runs/E04/feats_{j}.parquet"); X = df[CJ].fillna(0).values; y = df["judge_error"].values.astype(int); ag = df["agreement"].values; za = (ag - ag.mean()) / ag.std()
    for B in BUDS:
        k = -(-125 // B)
        for kind in ("strat", "bnd"):
            hi, zz, hr = [], [], []
            for o in range(16):
                perm = np.random.default_rng(10000 + o).permutation(len(df)); _, L, clf = indep(kind, 125, B, o, X, y, ag, perm)
                for bi in range(k, 12):
                    b = perm[bi * 250:(bi + 1) * 250]; s = b[np.argsort(-clf.predict_proba(X[b])[:, 1])[:B]]; hi.append((ag[s] >= 0.875).mean()); zz.append(za[s].mean()); hr.append(y[s].mean())
            print(f"    {lab_} B={B} amort_{kind}_125: share agreement>=0.875 {np.mean(hi):.3f}, mean z-agreement {np.mean(zz):+.2f}, hit rate {np.mean(hr):.3f} (population error rate {y.mean():.3f}, population share agreement>=0.875 {(ag >= 0.875).mean():.3f})")
# ---- (6) post hoc horizon extrapolation (NOT a measurement): total(H) / carry_plain(H) for the n=500 and n=250 designed arms and freeze_plain_500, assuming the post-calibration mean hits per batch stay flat after batch 12
print("(6) post hoc extrapolation of total/carry_plain at H batches (calibration batches as measured, then the mean post-calibration hits per batch held flat; all 16 order seeds; B=62 / B=125 cells in order 3B B=62, 3B B=125, 7B B=62, 7B B=125)")
for arm in ("amort_strat_250", "amort_strat_500", "amort_bnd_500", "freeze_plain_250"):
    for H in (12, 29, 60, 120):
        vals = []
        for lab_, j in J.items():
            for B in BUDS:
                pb = res["cells"][f"{lab_} B={B}"]["per_batch_mean_hits"]; k = -(-int(arm.rsplit("_", 1)[1]) // B)
                tot = lambda a: sum(pb[a][:k]) + np.mean(pb[a][k:]) * (H - k)
                vals.append(tot(arm) / tot("carry_plain"))
        print(f"    {arm} H={H}: " + ", ".join(f"{v:.3f}" for v in vals))
