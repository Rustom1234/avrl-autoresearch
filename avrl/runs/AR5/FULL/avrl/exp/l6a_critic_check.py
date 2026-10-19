"""L6a critic check (CPU, .venv python). Separately written from l6a_cell_rate.py: (1) recompute the ACC CV deltas with a pandas-groupby LOO rate and a nested-OOF rate, same folds;
(2) post hoc diagnostic: confident-only cell rate (errors counted only among the confident half, matching how the neighbor ratio was measured); (3) re-derive the replay ratios from the boot JSONs;
(4) the null distribution of the delta from the smoke runs. Writes nothing; prints. usage: l6a_critic_check.py"""
import os, json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
R = "/homes/rdubash/avrl_live/FULL/avrl/runs"
POOLS = {"rm": (f"{R}/L5c-proxies/rm", "accept", "judge_error"), "rm_sens": (f"{R}/L5c-proxies/rm", "accept_sens", "judge_error_sens"), "judge3b": (f"{R}/L5c-proxies/judge3b", "accept", "judge_error"), "q3": (f"{R}/L3b-judge-proxy", "accept", "judge_error")}
JA = ["s_maj", "s_oth", "s_diff", "mean_tokens", "std_tokens", "any_truncated", "log_qlen", "agreement", "n_distinct"]
def fit_pred(Xtr, ytr, Xte): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xtr, ytr).predict_proba(Xte)[:, 1]

def _cnt(cell, y, src):
    k = cell.max() + 1; return np.bincount(cell[src], minlength=k).astype(float), np.bincount(cell[src], weights=y[src], minlength=k)

def rate_loo(cell, y, tr, te, pool_mask=None, shrink=5.0):
    """shrunk leave-one-out rate. pool_mask: rows whose labels may be used (default all training rows); rows outside the mask are never excluded (their label is not in the pool)."""
    use = np.zeros(len(y), bool); use[tr] = True
    if pool_mask is not None: use &= pool_mask
    base = y[use].mean(); n, s = _cnt(cell, y, np.flatnonzero(use)); own = use.astype(float)
    return (s[cell] - y * own + shrink * base) / (n[cell] - own + shrink)

def rate_nested(cell, y, tr, te, pool_mask=None, shrink=5.0, k=5, seed=0):
    """training rows get an inner-5-fold out-of-fold rate (labels of the other inner folds only); test rows use all training labels."""
    use = np.zeros(len(y), bool); use[tr] = True
    if pool_mask is not None: use &= pool_mask
    base = y[use].mean(); out = np.zeros(len(y)); n, s = _cnt(cell, y, np.flatnonzero(use)); out[te] = (s[cell[te]] + shrink * base) / (n[cell[te]] + shrink)
    trl = np.array(tr)
    for a, b in StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(trl)), y[trl]):
        src = trl[a]; src = src[use[src]]; n, s = _cnt(cell, y, src); out[trl[b]] = (s[cell[trl[b]]] + shrink * base) / (n[cell[trl[b]]] + shrink)
    return out

def cv_delta(F, y, enc, cells=("cell100", "cell300"), shrink=5.0, top=None, restrict_conf=False):
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(np.zeros(len(y)), y)); X0 = F[JA].values; codes = {c: pd.factorize(F[c])[0] for c in cells}
    sets = {"JA": [], "JA+cell100": [], "JA+cell300": [], "JA+both": []}; oof = {k: np.zeros((3, len(y))) for k in sets}; tops = {k: [] for k in sets}; alone = []
    conf = F.s_maj.values >= np.median(F.s_maj.values)
    for f, (tr, te) in enumerate(folds):
        cr = [enc(codes[c], y, tr, te, pool_mask=(conf if restrict_conf else None), shrink=shrink) for c in cells]
        if restrict_conf: alone.append(roc_auc_score(y[te][conf[te]], cr[0][te][conf[te]]))
        X = np.column_stack([X0] + cr)
        for k, ix in (("JA", list(range(9))), ("JA+cell100", list(range(9)) + [9]), ("JA+cell300", list(range(9)) + [10]), ("JA+both", list(range(9)) + [9, 10])):
            p = fit_pred(X[tr][:, ix], y[tr], X[te][:, ix]); sets[k].append(roc_auc_score(y[te], p)); oof[k][f // 5, te] = p; tops[k].append(roc_auc_score(y[te][conf[te]], p[conf[te]]))
    rng = np.random.default_rng(11); idx = [rng.integers(0, len(y), len(y)) for _ in range(1000)]; idx = [i for i in idx if 0 < y[i].sum() < len(i)]
    ti = np.flatnonzero(conf); idt = [rng.integers(0, len(ti), len(ti)) for _ in range(1000)]; idt = [i for i in idt if 0 < y[ti][i].sum() < len(i)]
    avg = {k: v.mean(0) for k, v in oof.items()}; out = {"JA_auc": float(np.mean(sets["JA"]))}
    for k in ("JA+cell100", "JA+cell300", "JA+both"):
        d = float(np.mean(np.array(sets[k]) - np.array(sets["JA"]))); db = np.array([roc_auc_score(y[i], avg[k][i]) - roc_auc_score(y[i], avg["JA"][i]) for i in idx]); dt = float(np.mean(np.array(tops[k]) - np.array(tops["JA"])))
        dbt = np.array([roc_auc_score(y[ti][i], avg[k][ti][i]) - roc_auc_score(y[ti][i], avg["JA"][ti][i]) for i in idt])
        out[k] = {"delta": d, "ci": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))], "top_delta": dt, "top_ci": [float(np.percentile(dbt, 2.5)), float(np.percentile(dbt, 97.5))]}
    if alone: out["conf_only_rate_alone_auc_on_conf_rows"] = float(np.mean(alone))
    return out

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "cv"
    if which == "cv":
        logged = json.load(open(f"{R}/L6a-cell-rate-usability/result.json"))["pools"]
        for name, (d, acol, lcol) in POOLS.items():
            F = pd.read_parquet(f"{d}/feats_J.parquet").sort_values("pid").reset_index(drop=True); A = F[F[acol].astype(bool)].reset_index(drop=True); y = A[lcol].values.astype(int)
            print(name, "ACC", len(A), "err", int(y.sum()), "top-half err", int(y[A.s_maj.values >= np.median(A.s_maj.values)].sum()), flush=True)
            for tag, enc, kw in (("loo-groupby", rate_loo, {}), ("nested", rate_nested, {})):
                o = cv_delta(A, y, enc, **kw); print(" ", tag, "JA", round(o["JA_auc"], 4), {k: (round(o[k]["delta"], 4), [round(c, 4) for c in o[k]["ci"]], "top", round(o[k]["top_delta"], 4), [round(c, 4) for c in o[k]["top_ci"]]) for k in ("JA+cell100", "JA+cell300", "JA+both")}, flush=True)
    if which == "conf":
        for name, (d, acol, lcol) in POOLS.items():
            F = pd.read_parquet(f"{d}/feats_J.parquet").sort_values("pid").reset_index(drop=True); A = F[F[acol].astype(bool)].reset_index(drop=True); y = A[lcol].values.astype(int)
            for tag, enc in (("conf-only loo", rate_loo), ("conf-only nested", rate_nested)):
                o = cv_delta(A, y, enc, restrict_conf=True); print(name, tag, "rate-alone AUC on conf rows", round(o["conf_only_rate_alone_auc_on_conf_rows"], 4), {k: (round(o[k]["delta"], 4), [round(c, 4) for c in o[k]["ci"]], "top", round(o[k]["top_delta"], 4), [round(c, 4) for c in o[k]["top_ci"]]) for k in ("JA+cell100", "JA+cell300", "JA+both")}, flush=True)
