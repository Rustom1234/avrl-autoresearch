"""L11b oracle-covariate test (CPU). Usage: l11b_oracle.py <runs/L1a> <feats.parquet> <tag> [tag ...]   (one process per tag; outputs under runs/L11b)

Question: do GSM8K prompts carry difficulty structure that the allocator's features (cheap_r, 16 embedding PCs) miss? Oracle prompt-level covariates the allocator would NOT have:
  n_steps = number of <<...>> calculation annotations in the GSM8K gold answer, n_numbers_q = count of numbers in the question, log_abs_gold = log10(|gold answer| + 1).
Steps: (1) join runs/p0/feats.parquet to data/gsm8k_{train,test}.parquet on question (assert 3,000 matched, 0 duplicates), write runs/L11b/oracle_pid.parquet and augmented COPIES of the L1a tables
(tab_<tag>.parquet + 3 oracle columns; the L1a tables are untouched). (2) Same CV / bootstrap / permutation-null machinery as exp/l1a_structure.py (imported, not copied): pid-grouped 5 folds x 3 CV seeds,
balanced logistic on standardized features, same 200 pid-bootstrap resamples (seed 12345), 20-permutation null at CV seed 0 where the oracle vector of a pid is replaced by that of a random donor pid.
Models: cheap_r; +orc (all 3); +orc_ng (n_steps, n_numbers_q only: the leak-free subset, see below); +n_steps; +n_numbers_q; +log_abs_gold; +16 PCs; +PCs+orc; HistGradientBoosting on cheap_r vs +orc vs +orc_ng (3 CV seeds).
LEAK CAVEAT: cheap_r contains log_abs_ans (the rollout's own answer magnitude). log_abs_gold next to it lets a model detect answer == gold, i.e. read the label (corrupted = accepted and wrong). A linear logistic cannot
form the equality test, a tree model can partly. The orc_ng variant (no gold magnitude) is therefore the clean difficulty-only reading; the share of rows with log_abs_ans == log_abs_gold is reported.
(3) OOF ridge / target-encoding / HGB R^2 of the 16 PCs, log_qlen and the cell ids for n_steps, n_numbers_q, log_abs_gold (pid level, 3000 prompts, pid folds). (4) Gate: cheap_r OOF AUC must reproduce 0.7045 (3B) / 0.7090 (7B-AWQ)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from l1a_structure import CHEAP_R, PCS, CELLS, CV_SEEDS, NBOOT, NPERM, auc_w, fold_ids, oof, hgb

OUT = "runs/L11b"
ORC = ["n_steps", "n_numbers_q", "log_abs_gold"]
GATE = {"3B_q_corrupted": 0.7045, "7B_q_corrupted": 0.7090}

def build_oracle(feats_path):
    f = pd.read_parquet(feats_path).sort_values("pid").reset_index(drop=True); assert (f.pid.values == np.arange(len(f))).all()
    g = pd.concat([pd.read_parquet(f"data/gsm8k_{n}.parquet") for n in ("train", "test")], ignore_index=True)
    m = f[["pid", "question", "gold"]].merge(g, on="question", how="left", validate="one_to_one")  # one_to_one: 0 duplicate questions on either side
    assert len(m) == len(f) == 3000 and m.answer.notna().all(), (len(m), int(m.answer.notna().sum()))
    gold = pd.to_numeric(m.answer.str.split("####").str[-1].str.strip().str.replace(",", "", regex=False))
    assert (gold.values == pd.to_numeric(m.gold).values).all(), "parsed gold disagrees with feats.gold"
    o = pd.DataFrame({"pid": m.pid.values, "n_steps": m.answer.str.count(r"<<.*?>>").values.astype(float),
                      "n_numbers_q": m.question.str.count(r"\d[\d,]*\.?\d*").values.astype(float), "log_abs_gold": np.log10(gold.abs().values + 1)})
    return o

def r2(y, p): return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())

def r2_block(feats_path, orc):
    f = pd.read_parquet(feats_path).sort_values("pid").reset_index(drop=True); n = len(f); pix = np.arange(n)
    P = f[PCS].values.astype(float); Q = f[["log_qlen"]].values.astype(float); T = {c: orc[c].values for c in ORC}
    ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    hgr = lambda: HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, early_stopping=False, random_state=0)
    res = {"n_pids": n, "target_sd": {c: float(T[c].std()) for c in ORC}}
    for tn, y in T.items():
        r = {}
        for name, X, mk in [("ridge_pcs", P, ridge), ("ridge_qlen", Q, ridge), ("ridge_pcs_qlen", np.hstack([P, Q]), ridge), ("hgb_pcs", P, hgr), ("hgb_pcs_qlen", np.hstack([P, Q]), hgr)]:
            v = []
            for s in CV_SEEDS:
                fo = fold_ids(pix, n, s); p = np.zeros(n)
                for k in range(5):
                    tr, te = fo != k, fo == k; p[te] = mk().fit(X[tr], y[tr]).predict(X[te])
                v.append(r2(y, p))
            r[name] = {"mean": float(np.mean(v)), "per_seed": [float(x) for x in v]}
        for c in CELLS:  # OOF target encoding of the cell id, shrinkage 10 prompts toward the training mean
            cc = f[c].values.astype(int); nc = cc.max() + 1; v = []
            for s in CV_SEEDS:
                fo = fold_ids(pix, n, s); p = np.zeros(n)
                for k in range(5):
                    tr, te = fo != k, fo == k; sm = np.bincount(cc[tr], weights=y[tr], minlength=nc); cn = np.bincount(cc[tr], minlength=nc)
                    p[te] = (sm[cc[te]] + 10 * y[tr].mean()) / (cn[cc[te]] + 10)
                v.append(r2(y, p))
            r[f"te_{c}"] = {"mean": float(np.mean(v)), "per_seed": [float(x) for x in v]}
        res[tn] = r
    return res

def run_rung(tag):
    t0 = time.time(); lab = "corrupted"
    t = pd.read_parquet(os.path.join(OUT, f"tab_{tag}.parquet")); orc = pd.read_parquet(os.path.join(OUT, "oracle_pid.parquet")).sort_values("pid")
    n_pid = len(orc); pix = t.pid.values.astype(int); y = t[lab].values.astype(int); O = orc[ORC].values.astype(float)
    assert np.allclose(t[ORC].values, O[pix]); Oi = O[pix]
    Xc = t[CHEAP_R].values.astype(float); Xp = t[PCS].values.astype(float)
    X = {"cheap": Xc, "orc": np.hstack([Xc, Oi]), "orc_ng": np.hstack([Xc, Oi[:, :2]]), "steps": np.hstack([Xc, Oi[:, :1]]), "nnum": np.hstack([Xc, Oi[:, 1:2]]), "gold": np.hstack([Xc, Oi[:, 2:3]]),
         "pc": np.hstack([Xc, Xp]), "pc_orc": np.hstack([Xc, Xp, Oi]), "pc_orc_ng": np.hstack([Xc, Xp, Oi[:, :2]])}
    XH = {"hgb_cheap": Xc, "hgb_orc": X["orc"], "hgb_orc_ng": X["orc_ng"]}
    folds = {s: fold_ids(pix, n_pid, s) for s in CV_SEEDS}; preds = {}
    for s in CV_SEEDS:
        for m, M in X.items(): preds[(m, s)] = oof(M, y, folds[s])
        for m, M in XH.items(): preds[(m, s)] = oof(M, y, folds[s], model=hgb)
    grp = {k: (lambda u: (u[1], len(u[0])))(np.unique(v, return_inverse=True)) for k, v in preds.items()}
    models = sorted({k[0] for k in preds}); auc = {}
    for m in models: auc[m] = {"per_seed": [auc_w(preds[(m, s)], y, grp=grp[(m, s)]) for s in CV_SEEDS]}; auc[m]["mean"] = float(np.mean(auc[m]["per_seed"]))
    pairs = {"orc": ("cheap", "orc"), "orc_ng": ("cheap", "orc_ng"), "steps": ("cheap", "steps"), "nnum": ("cheap", "nnum"), "gold": ("cheap", "gold"), "pc": ("cheap", "pc"),
             "pc_orc_vs_pc": ("pc", "pc_orc"), "pc_orc_ng_vs_pc": ("pc", "pc_orc_ng"), "hgb_orc": ("hgb_cheap", "hgb_orc"), "hgb_orc_ng": ("hgb_cheap", "hgb_orc_ng")}
    rng = np.random.default_rng(12345); present = np.unique(pix); idx_of = np.full(n_pid, -1); idx_of[present] = np.arange(len(present)); W = []
    for b in range(NBOOT):
        cnt = np.bincount(rng.integers(0, len(present), len(present)), minlength=len(present)); W.append(cnt[idx_of[pix]].astype(float))
    delta = {}
    for name, (base, test) in pairs.items():
        pt = [auc[test]["per_seed"][i] - auc[base]["per_seed"][i] for i in range(len(CV_SEEDS))]
        bs = np.array([np.mean([auc_w(preds[(test, s)], y, w, grp[(test, s)]) - auc_w(preds[(base, s)], y, w, grp[(base, s)]) for s in CV_SEEDS]) for w in W])
        delta[name] = {"base": base, "test": test, "mean": float(np.mean(pt)), "per_seed": [float(v) for v in pt], "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)), "boot_sd": float(bs.std(ddof=1))}
    # permutation null, CV seed 0: the oracle vector of a pid replaced by that of a random donor pid
    rngp = np.random.default_rng(777); null = {k: [] for k in ("orc", "orc_ng", "hgb_orc", "hgb_orc_ng")}
    b_lin, b_hgb = auc["cheap"]["per_seed"][0], auc["hgb_cheap"]["per_seed"][0]
    for _ in range(NPERM):
        Od = O[rngp.permutation(n_pid)[pix]]; Xd = {"orc": np.hstack([Xc, Od]), "orc_ng": np.hstack([Xc, Od[:, :2]])}
        for k, M in Xd.items():
            null[k].append(auc_w(oof(M, y, folds[0]), y) - b_lin); null["hgb_" + k].append(auc_w(oof(M, y, folds[0], model=hgb), y) - b_hgb)
    for k, v in null.items(): delta[k]["null_mean"] = float(np.mean(v)); delta[k]["null_sd"] = float(np.std(v, ddof=1)); delta[k]["null_max"] = float(np.max(v)); delta[k]["null_values"] = [float(x) for x in v]
    # leak diagnostic: how often the rollout's answer magnitude equals the gold magnitude (answer == gold) and the label rate there
    eq = np.abs(t.log_abs_ans.values - Oi[:, 2]) < 1e-9
    leak = {"rows_log_abs_ans_equals_log_abs_gold": int(eq.sum()), "share": float(eq.mean()), "label_rate_when_equal": float(y[eq].mean()) if eq.any() else None, "label_rate_when_not_equal": float(y[~eq].mean())}
    # reproduction of the L1a numbers
    l1 = json.load(open(os.path.join(sys.argv[1], f"structure_{tag}.json")))
    repro = {"cheap_auc_l1a": l1["auc"]["cheap"]["mean"], "cheap_auc_here": auc["cheap"]["mean"], "pc_delta_l1a": l1["delta"]["pc"]["mean"], "pc_delta_here": delta["pc"]["mean"],
             "hgb_cheap_l1a": l1["auc"]["hgb_cheap"]["mean"], "hgb_cheap_here": auc["hgb_cheap"]["per_seed"][0]}
    gate_ok = round(auc["cheap"]["mean"], 4) == GATE[tag] and abs(repro["cheap_auc_l1a"] - repro["cheap_auc_here"]) < 1e-6
    out = {"tag": tag, "label": lab, "rows": int(len(t)), "pids": int(len(present)), "positives": int(y.sum()), "rate": float(y.mean()), "auc": auc, "delta": delta, "leak_diagnostic": leak, "repro": repro,
           "gate": {"expected_cheap_auc_4dp": GATE[tag], "got": auc["cheap"]["mean"], "passed": bool(gate_ok)}, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(OUT, f"oracle_{tag}.json"), "w"), indent=1)
    d = delta
    print(f"{tag}: n {len(t)} pids {len(present)} pos {int(y.sum())} | cheap {auc['cheap']['mean']:.4f} (gate {GATE[tag]} {'OK' if gate_ok else 'FAIL'}) | " + " | ".join(
        f"{k} {d[k]['mean']:+.4f} [{d[k]['ci_lo']:+.4f},{d[k]['ci_hi']:+.4f}]" + (f" null {d[k]['null_mean']:+.4f}±{d[k]['null_sd']:.4f} max {d[k]['null_max']:+.4f}" if "null_mean" in d[k] else "")
        for k in ("orc", "orc_ng", "steps", "nnum", "gold", "pc", "pc_orc_vs_pc", "hgb_orc", "hgb_orc_ng")) + f" | leak share {leak['share']:.4f} | {out['seconds']}s", flush=True)
    return gate_ok

if __name__ == "__main__":
    ld, fp, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(OUT, exist_ok=True)
    orc = build_oracle(fp); orc.to_parquet(os.path.join(OUT, "oracle_pid.parquet"), index=False)
    print(f"join OK: 3000 of 3000 matched, 0 duplicates; n_steps mean {orc.n_steps.mean():.3f} range {orc.n_steps.min():.0f}-{orc.n_steps.max():.0f}; n_numbers_q mean {orc.n_numbers_q.mean():.3f}; log_abs_gold mean {orc.log_abs_gold.mean():.3f}", flush=True)
    for tag in tags:
        t0 = pd.read_parquet(os.path.join(ld, f"tab_{tag}.parquet")); t = t0.merge(orc, on="pid", how="left", validate="many_to_one")
        assert len(t) == len(t0) and (t.pid.values == t0.pid.values).all() and t[ORC].notna().all().all() and list(t.columns[:len(t0.columns)]) == list(t0.columns)
        t.to_parquet(os.path.join(OUT, f"tab_{tag}.parquet"), index=False)
    r2res = r2_block(fp, orc); json.dump(r2res, open(os.path.join(OUT, "r2.json"), "w"), indent=1)
    print("R2 n_steps:", {k: round(v["mean"], 4) for k, v in r2res["n_steps"].items()}, flush=True)
    with Pool(len(tags)) as pool: oks = pool.map(run_rung, tags)
    json.dump({t: bool(o) for t, o in zip(tags, oks)}, open(os.path.join(OUT, "gate_auc.json"), "w"))
    sys.exit(0 if all(oks) else 3)
