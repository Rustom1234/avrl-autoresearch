"""L3b critic check (CPU, .venv python). Recomputes the L3b headline numbers from raw files (scores.parquet, groups.parquet, rollouts.parquet,
feats_T0.25.parquet for MiniLM pcs and cells, boot JSONs) with independently written code, then runs robustness checks that were NOT in the
pre-registered analysis (CV seed stability, within-fold-free cell-grouped CV, NHR controls). Nothing here changes the verdict; it only
checks it. Writes runs/L3b-judge-proxy/critic_check.json."""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
R = "/homes/rdubash/avrl_live/FULL/avrl/runs"; L = f"{R}/L3b-judge-proxy"; E = f"{R}/E21-lowT-confident-errors/T0.25"
out = {}
S = pd.read_parquet(f"{L}/scores.parquet"); G = pd.read_parquet(f"{E}/groups.parquet"); RO = pd.read_parquet(f"{E}/rollouts.parquet", columns=["pid", "k", "answer", "correct", "proxy"])
F = pd.read_parquet(f"{R}/E21-lowT-confident-errors/feats_T0.25.parquet")
M = S[S.role == "maj"].set_index("pid").sort_index(); O = S[S.role == "oth"].set_index("pid").sort_index()
G = G.set_index("pid").sort_index(); F = F.set_index("pid").sort_index()
# A. raw recompute
rk = RO.set_index(["pid", "k"])
assert (M.index == G.index).all() and (O.index == G.index).all()
gold_ok = np.array([rk.loc[(p, k), "correct"] for p, k in zip(M.index, M.k)])
out["A_maj_correct_equals_majority_correct"] = bool((gold_ok == G.majority_correct.values).all())
out["A_maj_is_proxy1"] = bool((M.proxy == 1).all()); out["A_oth_k_ne_maj_k"] = bool((O.k != M.k).all())
# label-free check of the 'oth' rule: lowest-k rollout with proxy==0 if any else lowest-k other
ok = True
for p in G.index:
    g = RO[RO.pid == p].sort_values("k"); d = g[g.proxy == 0]; mk = g[g.proxy == 1].iloc[0].k
    exp = d.iloc[0].k if len(d) else g[g.k != mk].iloc[0].k
    ok &= int(exp) == int(O.loc[p, "k"])
out["A_oth_rule_reproduced_from_proxy_only"] = bool(ok)
sm = M.s.values; acc = sm > 0; mc = G.majority_correct.values == 1; mv = ~mc
J = acc & mv
out["A_n_acc"] = int(acc.sum()); out["A_n_err_in_acc"] = int(J.sum()); out["A_n_mv_wrong"] = int(mv.sum()); out["A_jaccard"] = float((J & mv).sum() / (J | mv).sum())
out["A_judge_err_rate_overall"] = float(J.mean()); out["A_judge_err_rate_in_acc"] = float(J.sum() / acc.sum())
out["A_sanity_auc_rollouts"] = float(roc_auc_score(S.correct.values, S.s.values))
out["A_auc_smaj_alone_in_acc_direct"] = float(roc_auc_score(J[acc], -sm[acc])); out["A_auc_agreement_alone_in_acc_direct"] = float(roc_auc_score(J[acc], -G.agreement.values[acc]))
out["A_judge_rejects_right_majority"] = float(((~acc) & mc).sum() / mc.sum()); out["A_judge_accepts_wrong_majority"] = float((acc & mv).sum() / mv.sum())
out["A_auc_smaj_vs_majority_correct_all2000"] = float(roc_auc_score(mc, sm)); out["A_auc_agreement_vs_majority_correct_all2000"] = float(roc_auc_score(mc, G.agreement.values))
# B. CV
D = F.copy(); D["s_maj"] = sm; D["s_oth"] = O.s.values; D["s_diff"] = D.s_maj - D.s_oth; D["J"] = J.astype(int)
A = D[acc].reset_index()
J0 = ["s_maj", "s_oth", "s_diff", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]; JA = J0 + ["agreement", "n_distinct"]; EMB = [f"pc{i}" for i in range(16)]
y = A.J.values
def mk(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
def cv(cols_list, folds):
    res = {n: [] for n in cols_list}; oof = {n: np.zeros((folds_n_rep(folds), len(y))) for n in cols_list}
    for f, (tr, te) in enumerate(folds):
        for n, c in cols_list.items():
            p = mk().fit(A[c].values[tr], y[tr]).predict_proba(A[c].values[te])[:, 1]; res[n].append(roc_auc_score(y[te], p))
    return {n: np.array(v) for n, v in res.items()}
def folds_n_rep(folds): return 1
sets = {"JA": JA, "JA_emb": JA + EMB, "J0": J0, "J0_emb": J0 + EMB}
def run(folds, tag):
    r = cv(sets, folds); d1 = r["JA_emb"] - r["JA"]; d2 = r["J0_emb"] - r["J0"]
    out[f"B_{tag}"] = {"JA": float(r["JA"].mean()), "JA_emb": float(r["JA_emb"].mean()), "delta_JA": float(d1.mean()), "delta_JA_fold_sd": float(d1.std()), "J0": float(r["J0"].mean()), "J0_emb": float(r["J0_emb"].mean()), "delta_J0": float(d2.mean()), "n_folds": len(folds)}
run(list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(np.zeros(len(y)), y)), "registered_5x3_seed0")
run(list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1).split(np.zeros(len(y)), y)), "5x10_seed1")
run(list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=7).split(np.zeros(len(y)), y)), "5x10_seed7")
for ck in (30, 100):
    fl = []
    for rep in range(4): fl += list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, A[f"cell{ck}"].values))
    run(fl, f"cellgrouped_cell{ck}_4rep")
# C. NHR with own implementation + controls
def nhr(lab, g):
    df = pd.DataFrame({"y": lab.astype(float), "g": g}); base = df.y.mean(); s = df.groupby("g").y.agg(["sum", "count"]); df = df.join(s, on="g"); m = df[df["count"] > 1]
    loo = (m["sum"] - m["y"]) / (m["count"] - 1)
    return float(m.loc[loo > 0, "y"].mean() / base) if (loo > 0).any() and base > 0 else float("nan")
def nhr_null(lab, g, n=1000, seed=0):
    r = np.random.default_rng(seed); v = np.array([nhr(r.permutation(lab), g) for _ in range(n)]); v = v[np.isfinite(v)]
    return float(v.mean()), float(np.percentile(v, 95)), v
top = A[A.s_maj >= A.s_maj.median()]
out["C_top_half"] = {"n": int(len(top)), "n_err": int(top.J.sum())}
for ck in (30, 100, 300):
    o = nhr(top.J.values, top[f"cell{ck}"].values); nm, p95, v = nhr_null(top.J.values, top[f"cell{ck}"].values)
    out["C_top_half"][f"cell{ck}"] = {"obs": o, "null_mean": nm, "null_p95": p95, "frac_null_ge_obs": float((v >= o).mean())}
# controls at cell100: (i) whole ACC, (ii) several top fractions, (iii) majority-vote-wrong label (all 2000), (iv) same top-half rows but label = mv-wrong, (v) a random half instead of top half
def block(name, lab, g):
    o = nhr(lab, g); nm, p95, v = nhr_null(lab, g, 600); out.setdefault("C_controls", {})[name] = {"n": int(len(lab)), "n_pos": int(lab.sum()), "obs": o, "null_mean": nm, "null_p95": p95, "frac_null_ge_obs": float((v >= o).mean())}
block("J_all_ACC_cell100", A.J.values, A.cell100.values)
for q in (0.25, 0.5, 0.75):
    t = A[A.s_maj >= A.s_maj.quantile(1 - q)]; block(f"J_top{int(q*100)}pct_by_smaj_cell100", t.J.values, t.cell100.values)
t = A[A.agreement >= 0.75]; block("J_agree_ge_0.75_cell100", t.J.values, t.cell100.values)
t = D.reset_index(); block("mv_wrong_all2000_cell100", t.proxy_error.values, t.cell100.values)
t = D.reset_index(); tt = t[t.agreement >= 0.75]; block("mv_wrong_agree_ge_0.75_cell100_E21_set", tt.proxy_error.values, tt.cell100.values)
rr = np.random.default_rng(5); idx = rr.permutation(len(A))[: len(top)]; block("J_random_half_cell100", A.J.values[idx], A.cell100.values[idx])
# top-half-by-smaj rows: how many errors also in the E21 confident set, and mv-wrong NHR on exactly those rows
block("mv_wrong_on_same_top_half_rows_cell100", D.loc[top.pid.values].proxy_error.values, top.cell100.values)
# per-cell concentration at cell100 in top half: top cells by J count
c = top.groupby("cell100").J.agg(["sum", "count"]).sort_values("sum", ascending=False).head(8); out["C_top_cells_cell100"] = {int(i): [int(r["sum"]), int(r["count"])] for i, r in c.iterrows()}
out["C_top_half_cells_with_ge1_error"] = int((top.groupby("cell100").J.sum() > 0).sum()); out["C_top_half_n_cells"] = int(top.cell100.nunique())
# D. replay ratios from boot jsons
def rep(fn, frac):
    b = json.load(open(f"{L}/{fn}")); fp = b["full_pool"]; best = b["full_pool"]["best_fixed"]; rs = b["resamples"]
    rw = np.array([np.mean(r["greedy_warm"]) / np.mean(r[best]) for r in rs]); rb = np.array([np.mean(r["greedy_blend"]) / np.mean(r[best]) for r in rs])
    pw = fp["ratio_greedy_warm_over_best_fixed"]; pb = fp["ratio_greedy_blend_over_best_fixed"]; per = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    bas = lambda v, p: [float(2 * p - np.percentile(v, 97.5)), float(2 * p - np.percentile(v, 2.5))]
    return {"best_fixed": best, "plug_warm": pw, "boot_mean_warm": float(rw.mean()), "pct_warm": per(rw), "basic_warm": bas(rw, pw), "frac_warm_gt1": float((rw > 1).mean()),
            "plug_blend": pb, "boot_mean_blend": float(rb.mean()), "pct_blend": per(rb), "basic_blend": bas(rb, pb), "frac_blend_gt1": float((rb > 1).mean()),
            "means": {k: float(np.mean([np.mean(r[k]) for r in rs])) for k in ["uniform", "agree_thr", "judge_thr", "blend50", "greedy_warm", "greedy_blend", "oracle"]},
            "norefit_equals_blend50_where_present": [sum("greedy_warm_norefit" in r for r in rs), bool(all(r["greedy_warm_norefit"] == r["blend50"] for r in rs if "greedy_warm_norefit" in r))]}
out["D_f0.25"] = rep("boot_J_f0.25.json", 0.25); out["D_f0.1"] = rep("boot_J_f0.1.json", 0.1); out["D_held_f0.25"] = rep("boot_J_held_f0.25.json", 0.25)
# E. what does blend50 give vs oracle ranks on the full pool, deterministic recompute of the fixed rules at 25% and 10% (tie-break random, 20 draws)
def ranks(v): return pd.Series(v).rank(method="average").values
def fixed_hits(order_key, labels, budget, rng, nd=20):
    h = []
    for _ in range(nd):
        tb = rng.random(len(labels)); idx = np.lexsort((tb, order_key))[:budget]; h.append(labels[idx].sum())
    return float(np.mean(h))
Dd = D.reset_index(); lab = Dd.J.values; rg = np.random.default_rng(0); n = len(Dd)
# agree_thr: agreement ascending then n_distinct descending (lowest confidence first)
ag_key = ranks(Dd.agreement.values * 100 - Dd.n_distinct.values)  # ascending confidence
# judge_thr: accepted groups by s_maj ascending, rejected last
jt_key = np.where(Dd.s_maj.values > 0, Dd.s_maj.values, 1e9)
# blend50: rank blend of agreement-rank and s_maj rank (both ascending confidence); rejected groups cannot be errors under J so they sink via label anyway
bl_key = 0.5 * ranks(Dd.agreement.values * 100 - Dd.n_distinct.values) + 0.5 * ranks(Dd.s_maj.values)
out["E_recompute_fixed_hits"] = {}
for fr in (0.1, 0.25):
    b = int(fr * n); out["E_recompute_fixed_hits"][str(fr)] = {"agree_thr": fixed_hits(ag_key, lab, b, rg), "judge_thr": fixed_hits(jt_key, lab, b, rg), "blend50_my_def": fixed_hits(bl_key, lab, b, rg), "uniform": float(lab.mean() * b), "n_err": int(lab.sum())}
json.dump(out, open(f"{L}/critic_check.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
