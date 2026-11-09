"""L6a-cell-rate-usability, CV half (CPU, .venv python). PRE-REGISTERED here, in this docstring, before any real accept / judge_error / majority_correct label was read for this experiment.
Copy of the CV and bootstrap code path of exp/l5c_analyze.py (identical folds, scaler, learner, bootstrap), with the cell-rate features added. Usage: l6a_cell_rate.py --pool <dir with feats_J.parquet> --out <dir> [--sens] [--consensus d1 d2 d3]
Question. Does the cell100 clustering of proxy-accepted gold-wrong groups (neighbor ratio 2.09 Qwen3-4B, 1.67 Qwen2.5-3B, 1.24 Skywork rm at top50; null p95 about 1.23 to 1.43) carry information that 16 MiniLM PCs cannot extract linearly?
Label. J = accepted by the pool's registered accept rule AND majority completion gold-wrong (column judge_error). Population ACC = accepted groups. Registered accept rules: rm top-K, K = 1,715 (gold-calibrated, K comes from the true
majority-vote accuracy; flagged as such), judge3b s_maj > 0, Qwen3-4B s_maj > 0 (L3b pool). SENSITIVITY (--sens, rm only, never used for the verdict): accept_sens / judge_error_sens, K = 1,840 (the Qwen3-4B accept count).
Features. J0 = s_maj, s_oth, s_diff, mean_tokens, std_tokens, any_truncated, log_qlen. JA = J0 + agreement + n_distinct (PRIMARY base). Cell rate for a group in cell c (cell100 and cell300 separately, computed INSIDE each CV fold from the TRAINING fold labels only,
inside ACC): shrunk leave-one-out rate = (errors among the OTHER training-fold ACC members of c + 5 x base) / (other training-fold ACC members of c + 5), base = training-fold error rate; a training row excludes itself, a test row uses all training
members of its cell (cells with no training member get the base rate). Sets: JA, JA+cell100, JA+cell300, JA+both. CV = 5x3 RepeatedStratifiedKFold (random_state 0) on ACC, StandardScaler + balanced LogisticRegression(max_iter 1000), 15-fold mean AUC and fold SD,
delta = fold mean of (set - JA); paired bootstrap (1,000 ACC resamples, default_rng(0)) of the delta of the 3-repeat-averaged out-of-fold scores, percentile 95% CI. PRIMARY metric: AUC on ACC. SECONDARY (never in the verdict): the same AUC restricted to the top half of ACC by s_maj
(s_maj >= median of ACC), models still trained on all of ACC (fold AUCs on test rows in the top half; bootstrap over top-half rows). Model-free diagnostic (never in the verdict): AUC of the fold's test-row cell rate used alone as a score (higher rate = more suspect), which separates
"information present" from "a logistic model on leave-one-out training features can extract it" (a leave-one-out target encoding shifts an error row's own training-feature down by 1/(n_c + 4), a known attenuation).
Consensus label, POST HOC, report only: accepted by all three registered accept rules (rm K = 1,715, judge3b, Qwen3-4B) AND gold-wrong; expected 138 errors among 1,593 consensus-accepted groups (checked and reported, not a gate); CV with this pool's own JA features on the consensus-accepted rows.
Gates (exit 4): (1) the same CV code reproduces the E21 confident-subset cheap / cheap+emb / agreement-only AUCs to 1e-6 (needs proxy_error and the E21 result.json); (2) the JA AUC on ACC reproduces the JA AUC saved in <pool>/result_analyze.json to 1e-6 (primary, and sensitivity when --sens).
A gate is recorded as skipped only when its inputs do not exist (synthetic smoke pools); every real run must show both as passed.
REGISTERED BINS (the verdict is formed in exp/l6a_finalize.py from this file and exp/sim_warm_boot_j3.py; here only the CV side is defined). Per pool p: A_p = the best cell set (largest fold-mean delta among JA+cell100, JA+cell300, JA+both) has delta >= 0.03 on ACC AND paired bootstrap CI lower end > 0.
K_p = for EVERY one of the three cell sets, delta < 0.02 AND bootstrap CI upper end < 0.05. PASS (experiment) needs, on at least two of the three pools, A_p AND the replay criterion R_p (cell_rerank / agree_acc percentile 95% CI lower end > 1.0 at the 25% budget) on the SAME pool; if A holds on >= 2 pools and R holds on >= 2 pools but not
jointly on 2, the verdict is INCONCLUSIVE (loose reading, reported). KILL (literal registered bin) = K_p on all three pools; if R_p also holds on two or more pools the verdict line says the kill is discordant with the replay. Anything else is INCONCLUSIVE. The best-of-three selection is not multiplicity-adjusted; the three
pools share prompts and about 70% of their errors, so two-of-three is weaker than two independent passes; all proxies and the policy are Qwen-family. Power caveat: the clustered errors sit in the confident half (49 to 77 errors), so the ACC-wide AUC dilutes them; a kill says the clustering is unusable by this supervised prior, not that it is absent."""
import os, sys, json, argparse, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
RUNS = os.path.join(ROOT, "avrl", "runs")
ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True); ap.add_argument("--out", required=True); ap.add_argument("--sens", action="store_true")
ap.add_argument("--consensus", nargs="*", default=[os.path.join(RUNS, "L5c-proxies", "rm"), os.path.join(RUNS, "L5c-proxies", "judge3b"), os.path.join(RUNS, "L3b-judge-proxy")])
ap.add_argument("--e21", default=os.path.join(RUNS, "E21-lowT-confident-errors")); ap.add_argument("--B", type=int, default=1000)
a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out, exist_ok=True)
cheap = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]; embc = [f"pc{i}" for i in range(16)]
J0 = ["s_maj", "s_oth", "s_diff", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]; JA = J0 + ["agreement", "n_distinct"]
CELLS = ("cell100", "cell300"); SHRINK = 5.0; NAMES = JA + ["cr100", "cr300"]
SETS = {"JA": JA, "JA+cell100": JA + ["cr100"], "JA+cell300": JA + ["cr300"], "JA+both": JA + ["cr100", "cr300"]}; CELL_SETS = [k for k in SETS if k != "JA"]
COLS = {k: [NAMES.index(c) for c in v] for k, v in SETS.items()}

def auc_direct(y, s): return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")

def rate_feats(codes, y, tr, te):
    """[n, 2] shrunk cell rates (cell100, cell300): training rows leave themselves out, test rows use all training members; labels from the training fold only."""
    base = y[tr].mean(); out = np.zeros((len(y), 2))
    for j, c in enumerate(CELLS):
        cd = codes[c]; nc = cd.max() + 1; cnt = np.bincount(cd[tr], minlength=nc); sm = np.bincount(cd[tr], weights=y[tr], minlength=nc)
        out[tr, j] = (sm[cd[tr]] - y[tr] + SHRINK * base) / (cnt[cd[tr]] - 1 + SHRINK); out[te, j] = (sm[cd[te]] + SHRINK * base) / (cnt[cd[te]] + SHRINK)
    return out

def cv(G, y, top, sets):
    """E21 cv_aucs code path with the in-fold cell-rate features. Returns fold AUCs on all rows and on top-half rows per set, model-free cell-rate AUCs, and OOF [3 repeats, n] per set."""
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(np.zeros(len(y)), y)); codes = {c: pd.factorize(G[c].values)[0] for c in CELLS}; X0 = G[JA].values
    auc = {k: [] for k in sets}; auct = {k: [] for k in sets}; free = {f"{c}_alone": [] for c in CELLS}; freet = {f"{c}_alone": [] for c in CELLS}; oof = {k: np.zeros((3, len(y))) for k in sets}
    for f, (tr, te) in enumerate(folds):
        R = rate_feats(codes, y, tr, te); X = np.column_stack([X0, R]); tt = top[te]
        for j, c in enumerate(CELLS):
            free[f"{c}_alone"].append(auc_direct(y[te], R[te, j])); freet[f"{c}_alone"].append(auc_direct(y[te][tt], R[te, j][tt]))
        for k, ix in sets.items():
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X[tr][:, ix], y[tr]); p = m.predict_proba(X[te][:, ix])[:, 1]
            auc[k].append(roc_auc_score(y[te], p)); auct[k].append(auc_direct(y[te][tt], p[tt])); oof[k][f // 5, te] = p
    f2a = lambda d: {k: np.array(v) for k, v in d.items()}
    return f2a(auc), f2a(auct), f2a(free), f2a(freet), oof

def delta_block(au, avg, y, idx, base="JA"):
    """Fold mean / SD of (set - base) and the paired bootstrap of the delta of the repeat-averaged OOF scores over the resample index list idx."""
    out = {}
    for k in CELL_SETS:
        d = au[k] - au[base]; db = np.array([roc_auc_score(y[i], avg[k][i]) for i in idx]) - np.array([roc_auc_score(y[i], avg[base][i]) for i in idx])
        out[k] = {"delta": float(np.nanmean(d)), "delta_fold_sd": float(np.nanstd(d)), "oof_avg_delta": float(auc_direct(y, avg[k]) - auc_direct(y, avg[base])),
                  "boot_ci95": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))], "boot_sd": float(db.std()), "boot_mean": float(db.mean())}
    return out

def draw_idx(n, y, B):
    rng = np.random.default_rng(0); idx = [rng.integers(0, n, n) for _ in range(B)]; return [i for i in idx if 0 < y[i].sum() < len(i)]

def analyze(G, y, tag, rule):
    """CV block for one population (ACC rows G, label y). Primary AUC on ACC, secondary on the top half of ACC by s_maj."""
    y = y.astype(int); top = (G.s_maj.values >= np.median(G.s_maj.values)); n_err = int(y.sum())
    r = {"tag": tag, "rule": rule, "n_acc": int(len(y)), "n_err": n_err, "base_rate": float(y.mean()), "n_top": int(top.sum()), "n_err_top": int(y[top].sum()), "top_rate": float(y[top].mean())}
    if n_err < 10 or len(y) - n_err < 10: r["note"] = "too few errors for CV"; return r, None
    au, aut, free, freet, oof = cv(G, y, top, COLS); avg = {k: v.mean(0) for k, v in oof.items()}
    idx = draw_idx(len(y), y, a.B); r["n_boot_used"] = len(idx); r["sets"] = {k: {"auc": float(au[k].mean()), "auc_fold_sd": float(au[k].std()), "oof_avg_auc": auc_direct(y, avg[k])} for k in SETS}
    r["deltas"] = delta_block(au, avg, y, idx); r["cell_rate_alone"] = {k: {"auc_fold_mean": float(v.mean()), "auc_fold_sd": float(v.std())} for k, v in free.items()}
    ti = np.flatnonzero(top); yt = y[top]; idxt = draw_idx(len(ti), yt, a.B); avgt = {k: v[top] for k, v in avg.items()}
    r["top_half"] = {"n_boot_used": len(idxt), "sets": {k: {"auc_fold_mean": float(np.nanmean(aut[k])), "auc_fold_sd": float(np.nanstd(aut[k])), "oof_avg_auc": auc_direct(yt, avgt[k])} for k in SETS},
                     "deltas": delta_block(aut, avgt, yt, idxt), "cell_rate_alone": {k: {"auc_fold_mean": float(np.nanmean(v)), "auc_fold_sd": float(np.nanstd(v))} for k, v in freet.items()}}
    best = max(CELL_SETS, key=lambda k: r["deltas"][k]["delta"]); r["best_cell_set"] = best
    r["A_pass"] = bool(r["deltas"][best]["delta"] >= 0.03 and r["deltas"][best]["boot_ci95"][0] > 0)
    r["K_kill_side"] = bool(all(r["deltas"][k]["delta"] < 0.02 and r["deltas"][k]["boot_ci95"][1] < 0.05 for k in CELL_SETS))
    return r, au

F = pd.read_parquet(os.path.join(a.pool, "feats_J.parquet")).sort_values("pid").reset_index(drop=True); n = len(F)
assert (F.pid.values == np.sort(F.pid.values)).all() and len(F) == F.pid.nunique(); print(f"[{os.path.basename(a.pool)}] groups {n}", flush=True)
gates = {}

# gate 1: E21 confident-subset reproduction (inputs exist only for real pools)
e21p = os.path.join(a.e21, "result.json")
if "proxy_error" in F.columns and os.path.exists(e21p) and n == 2000:
    conf = F[F.agreement >= 0.75].reset_index(drop=True); yc = conf.proxy_error.values.astype(int); folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(np.zeros(len(yc)), yc)); au = {k: [] for k in ("cheap", "cheap_emb", "agree")}
    for tr, te in folds:
        au["agree"].append(roc_auc_score(yc[te], -conf.agreement.values[te]))
        for k, cols in (("cheap", cheap), ("cheap_emb", cheap + embc)):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(conf[cols].values[tr], yc[tr]); au[k].append(roc_auc_score(yc[te], m.predict_proba(conf[cols].values[te])[:, 1]))
    e = json.load(open(e21p))["datasets"]["T0.25"]["confident"]; ref = {"cheap": e["auc_cheap"], "cheap_emb": e["auc_cheap_emb"], "agree": e["auc_agreement_only"]}
    g = {"here": {k: float(np.mean(v)) for k, v in au.items()}, "e21": ref}; g["max_abs_diff"] = max(abs(g["here"][k] - ref[k]) for k in ref); g["pass"] = bool(g["max_abs_diff"] < 1e-6); gates["e21_reproduction"] = g
else: gates["e21_reproduction"] = {"skipped": "no proxy_error column / E21 result.json / n != 2000 (synthetic pool)", "pass": None}

def rule_columns(sens):
    acc = F["accept_sens" if sens else "accept"].values.astype(bool); lab = F["judge_error_sens" if sens else "judge_error"].values.astype(int); assert (lab == (acc & (F.majority_correct.values == 0)).astype(int)).all() if "majority_correct" in F else True
    return acc, lab

def ref_ja(sens):
    p = os.path.join(a.pool, "result_analyze.json")
    if not os.path.exists(p): return None
    r = json.load(open(p)); blk = r.get("sensitivity") if sens else (r.get("primary") or r.get("primary_tau0"))
    return blk["sets"]["JA"]["auc"] if blk and "sets" in blk else None

res = {"pool": a.pool, "n_groups": n, "gates": gates, "cell_rate_shrink": SHRINK, "B": a.B}; aucs = {}
runs = [("primary", False, "registered accept rule (column accept / judge_error)")] + ([("sensitivity", True, "accept_sens / judge_error_sens (rm: K = 1,840, the Qwen3-4B accept count)")] if a.sens else [])
for tag, sens, rule in runs:
    acc, lab = rule_columns(sens); ACC = F[acc].reset_index(drop=True); r, au = analyze(ACC, lab[acc], tag, rule); res[tag] = r
    ref = ref_ja(sens)
    if au is not None and ref is not None:
        g = {"here": float(au["JA"].mean()), "reference": float(ref)}; g["abs_diff"] = abs(g["here"] - g["reference"]); g["pass"] = bool(g["abs_diff"] < 1e-6); gates[f"ja_auc_reproduction_{tag}"] = g
    else: gates[f"ja_auc_reproduction_{tag}"] = {"skipped": "no result_analyze.json reference or too few errors", "pass": None}
    print(f"[{tag}] ACC {r['n_acc']}, n_err {r['n_err']} (top half {r.get('n_err_top')}), JA AUC {r['sets']['JA']['auc']:.4f}" if "sets" in r else f"[{tag}] {r.get('note')}", flush=True)

# consensus label, post hoc
try:
    P = [pd.read_parquet(os.path.join(d, "feats_J.parquet")).sort_values("pid").reset_index(drop=True) for d in a.consensus]
    assert all((p.pid.values == F.pid.values).all() for p in P); cacc = np.logical_and.reduce([p.accept.values.astype(bool) for p in P]); clab = (cacc & (F.majority_correct.values == 0)).astype(int)
    r, _ = analyze(F[cacc].reset_index(drop=True), clab[cacc], "consensus_post_hoc", "accepted by all three registered accept rules AND gold-wrong (post hoc)"); r["matches_spec_1593_138"] = bool(int(cacc.sum()) == 1593 and int(clab.sum()) == 138); res["consensus_post_hoc"] = r
    print(f"[consensus] ACC {r['n_acc']} err {r['n_err']} (spec 1593 / 138: {r['matches_spec_1593_138']})", flush=True)
except Exception as ex: res["consensus_post_hoc"] = {"status": "failed", "message": repr(ex)}; print("consensus failed:", repr(ex), flush=True)

res["gates_all_pass"] = bool(all(g.get("pass") is not False for g in gates.values())); res["gates_any_skipped"] = bool(any(g.get("pass") is None for g in gates.values())); res["wall_s"] = time.time() - t0
json.dump(res, open(os.path.join(a.out, "result_cv.json"), "w"), indent=1, default=float)

f4 = lambda x: f"{x:.4f}"; L = [f"## CV, pool {os.path.basename(a.pool)}; gates: {json.dumps(gates)}", ""]
for tag in ("primary", "sensitivity", "consensus_post_hoc"):
    r = res.get(tag)
    if not r or "sets" not in r:
        if r: L.append(f"### {tag}: {json.dumps(r)}")
        continue
    L += [f"### {tag}: {r['rule']}", f"ACC {r['n_acc']}, n_err {r['n_err']} (rate {r['base_rate']:.4f}); top half by s_maj {r['n_top']}, n_err {r['n_err_top']} (rate {r['top_rate']:.4f}); boot used {r['n_boot_used']}; best cell set {r['best_cell_set']}; A_pass {r['A_pass']}; K_kill_side {r['K_kill_side']}", "",
          "| set | AUC ACC (fold SD) | AUC top half (fold SD) |", "|---|---|---|"]
    L += [f"| {k} | {f4(v['auc'])} ({f4(v['auc_fold_sd'])}) | {f4(r['top_half']['sets'][k]['auc_fold_mean'])} ({f4(r['top_half']['sets'][k]['auc_fold_sd'])}) |" for k, v in r["sets"].items()]
    L += ["", "| delta vs JA | ACC fold mean (SD) | ACC boot 95% CI | top-half fold mean (SD) | top-half boot 95% CI |", "|---|---|---|---|---|"]
    for k in CELL_SETS:
        d, t = r["deltas"][k], r["top_half"]["deltas"][k]; L.append(f"| {k} | {d['delta']:+.4f} ({f4(d['delta_fold_sd'])}) | [{d['boot_ci95'][0]:+.4f}, {d['boot_ci95'][1]:+.4f}] | {t['delta']:+.4f} ({f4(t['delta_fold_sd'])}) | [{t['boot_ci95'][0]:+.4f}, {t['boot_ci95'][1]:+.4f}] |")
    L += ["", "model-free cell-rate-alone AUC (ACC | top half): " + "; ".join(f"{k} {v['auc_fold_mean']:.4f} | {r['top_half']['cell_rate_alone'][k]['auc_fold_mean']:.4f}" for k, v in r["cell_rate_alone"].items()), ""]
open(os.path.join(a.out, "cv_table.txt"), "w").write("\n".join(L)); print("\n".join(L), flush=True); print(f"cv done in {time.time()-t0:.0f}s", flush=True)
if not res["gates_all_pass"]: print("GATE FAILED", json.dumps(gates), flush=True); sys.exit(4)
