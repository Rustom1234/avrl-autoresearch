"""L11b second adversarial recheck (CPU). Independent of exp/l11b_oracle.py and exp/l11b_critic.py: own join, own folds (shuffled-pid GroupKFold, seeds 201 to 203), sklearn roc_auc_score.
Checks: (1) join and n_steps; (2) logistic deltas for orc_ng / steps / orc / pc over cheap_r; (3) the verdict fork: a regularised depth-3 HGB and a pairwise-interaction logistic on cheap_r vs cheap_r + orc_ng,
reported next to the baseline gap to logistic cheap_r; (4) budget and header equality across the three replay feature sets; (5) AUC of n_steps among rows by label (sanity). Usage: l11b_critic2.py (cwd = avrl)."""
import json, sys, glob, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

CR = ["judge_logit", "log_tokens", "truncated", "has_decimal", "is_negative", "log_abs_ans", "empty_answer", "log_qlen"]
PCS = [f"pc{i}" for i in range(16)]
f = pd.read_parquet("runs/p0/feats.parquet").sort_values("pid").reset_index(drop=True)
g = pd.concat([pd.read_parquet(f"data/gsm8k_{n}.parquet") for n in ("train", "test")], ignore_index=True)
print("gsm8k union rows", len(g), "dup questions", int(g.question.duplicated().sum()))
m = f[["pid", "question", "gold"]].merge(g, on="question", how="left")
print("matched", int(m.answer.notna().sum()), "of", len(f), "rows after merge", len(m))
goldp = pd.to_numeric(m.answer.str.split("####").str[-1].str.strip().str.replace(",", "", regex=False))
print("parsed gold == feats.gold:", bool((goldp.values == pd.to_numeric(m.gold).values).all()))
ns = m.answer.apply(lambda s: s.count("<<")).values.astype(float)  # '<<' count, independent of the regex used by the run
nn = m.question.str.count(r"\d[\d,]*\.?\d*").values.astype(float)
print("n_steps mean %.3f range %d-%d, zero share %.4f; n_numbers_q mean %.3f" % (ns.mean(), ns.min(), ns.max(), (ns == 0).mean(), nn.mean()))
oracle = pd.DataFrame({"pid": f.pid.values, "ns": ns, "nn": nn})

def folds(pix, seed, k=5):
    rng = np.random.default_rng(seed); perm = rng.permutation(pix.max() + 1); relabel = perm[pix]
    return list(GroupKFold(n_splits=k).split(np.zeros(len(pix)), groups=relabel))

def oof_auc(mk, X, y, fs):
    p = np.zeros(len(y))
    for tr, te in fs: p[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p)

lg = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
lg2 = lambda C: (lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), LogisticRegression(max_iter=3000, C=C, class_weight="balanced")))
hg_run = lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, early_stopping=False, random_state=0)
hg_reg = lambda it: (lambda: HistGradientBoostingClassifier(max_iter=it, learning_rate=0.05, max_depth=3, l2_regularization=1.0, min_samples_leaf=50, early_stopping=False, random_state=0))

for tag in ("3B_q_corrupted", "7B_q_corrupted"):
    t = pd.read_parquet(f"runs/L11b/tab_{tag}.parquet"); pix = t.pid.values.astype(int); y = t.corrupted.values.astype(int)
    o = oracle.set_index("pid").loc[pix]; Xc = t[CR].values.astype(float); Xp = t[PCS].values.astype(float)
    assert np.allclose(o.ns.values, t.n_steps.values) and np.allclose(o.nn.values, t.n_numbers_q.values), "n_steps / n_numbers_q differ from the run's columns"
    ONG = np.column_stack([o.ns.values, o.nn.values]); S = o.ns.values[:, None]
    print(f"\n== {tag}: rows {len(t)} pids {len(np.unique(pix))} positives {int(y.sum())} rate {y.mean():.4f} | corr(n_steps, label) {np.corrcoef(o.ns.values, y)[0, 1]:.3f}")
    res = {"cheap": [], "steps": [], "orc_ng": [], "pc": [], "hgb_cheap": [], "hgb_orc_ng": []}
    extra = {"hreg200_cheap": [], "hreg200_orc_ng": [], "hreg400_cheap": [], "hreg400_orc_ng": [], "lg2_C0.1_cheap": [], "lg2_C0.1_orc_ng": [], "lg2_C0.01_cheap": [], "lg2_C0.01_orc_ng": []}
    t0 = time.time()
    for s in (201, 202, 203):
        fs = folds(pix, s)
        res["cheap"].append(oof_auc(lg, Xc, y, fs)); res["steps"].append(oof_auc(lg, np.hstack([Xc, S]), y, fs)); res["orc_ng"].append(oof_auc(lg, np.hstack([Xc, ONG]), y, fs))
        res["pc"].append(oof_auc(lg, np.hstack([Xc, Xp]), y, fs))
        res["hgb_cheap"].append(oof_auc(hg_run, Xc, y, fs)); res["hgb_orc_ng"].append(oof_auc(hg_run, np.hstack([Xc, ONG]), y, fs))
        for it in (200, 400):
            extra[f"hreg{it}_cheap"].append(oof_auc(hg_reg(it), Xc, y, fs)); extra[f"hreg{it}_orc_ng"].append(oof_auc(hg_reg(it), np.hstack([Xc, ONG]), y, fs))
        for C in (0.1, 0.01):
            extra[f"lg2_C{C}_cheap"].append(oof_auc(lg2(C), Xc, y, fs)); extra[f"lg2_C{C}_orc_ng"].append(oof_auc(lg2(C), np.hstack([Xc, ONG]), y, fs))
    mu = {k: float(np.mean(v)) for k, v in res.items()}; mx = {k: float(np.mean(v)) for k, v in extra.items()}
    print("logistic cheap_r %.4f | +n_steps %+.4f | +orc_ng %+.4f | +16 PCs %+.4f  (per-seed orc_ng delta %s)" % (mu["cheap"], mu["steps"] - mu["cheap"], mu["orc_ng"] - mu["cheap"], mu["pc"] - mu["cheap"],
          [round(a - b, 4) for a, b in zip(res["orc_ng"], res["cheap"])]))
    print("run-config HGB (100 it, lr 0.1, unregularised): cheap %.4f (gap to logistic cheap %+.4f) | +orc_ng %.4f | delta %+.4f | HGB+orc_ng minus logistic+orc_ng %+.4f" % (
        mu["hgb_cheap"], mu["hgb_cheap"] - mu["cheap"], mu["hgb_orc_ng"], mu["hgb_orc_ng"] - mu["hgb_cheap"], mu["hgb_orc_ng"] - mu["orc_ng"]))
    for k in ("hreg200", "hreg400", "lg2_C0.1", "lg2_C0.01"):
        b, a = mx[f"{k}_cheap"], mx[f"{k}_orc_ng"]
        print(f"{k}: cheap {b:.4f} (gap to logistic cheap {b - mu['cheap']:+.4f}) | +orc_ng {a:.4f} | delta {a - b:+.4f} | delta vs LOGISTIC cheap_r {a - mu['cheap']:+.4f}")
    print(f"({time.time() - t0:.0f}s)")

# (4) header and budget equality across the replay feature sets, and identical non-feature arms
bad = 0; n = 0
for fp in sorted(glob.glob("runs/L11b/sim_*_cheap_*_s?.json")):
    a = json.load(open(fp));
    for fs in ("orc", "orc_ng"):
        b = json.load(open(fp.replace("_cheap_", f"_{fs}_"))); n += 1
        hd = ("n_fit", "n_fit_pos", "n_sim", "n_buckets", "hits", "split_seed", "label", "group_col", "n_policy_seeds")
        if any(a[k] != b[k] for k in hd): bad += 1; print("HEADER MISMATCH", fp, fs)
        for bf in ("budget_0.1", "budget_0.25", "budget_0.5"):
            for pol in ("uniform", "oracle", "thr_judge_low", "thr_boundary", "thr_tokens_long", "thr_trunc_first"):
                if a[bf][pol]["per_seed"] != b[bf][pol]["per_seed"]: bad += 1; print("FEATURE-FREE ARM DIFFERS", fp, fs, bf, pol)
            if sorted(a[bf]) != sorted(b[bf]): bad += 1; print("POLICY SET DIFFERS", fp, fs, bf)
print(f"\nreplay header / feature-free arm / policy-set comparison: {n} file pairs, {bad} mismatches")
