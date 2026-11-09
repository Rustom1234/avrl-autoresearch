"""L3a stage 1 (CPU): does the policy's own hidden state add cross-fit AUC for proxy error over the cheap rollout features?
Reads <run_dir>/hs.npz (l3a_hidden_probe.py) and <e21>/feats_T0.25.parquet; standardize + PCA(16, random_state=0) per (layer, position) on all 2000 groups
(label-free, same dimension as the MiniLM pc0..pc15); writes <run_dir>/feats_T0.25_hs.parquet (a copy of the E21 feats plus hs columns hs{P|A}{layer}_{0..15}),
<run_dir>/result_stage1.json and stage1_table.md. CV is the E21 code path exactly (RepeatedStratifiedKFold 5x3 random_state=0, StandardScaler + balanced LogisticRegression).
Reproduction gate: confident-subset cheap and cheap+emb AUCs must equal E21 result.json (tolerance 1e-6) or the script exits 4."""
import os, sys, json, argparse, subprocess, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("--run_dir", required=True); ap.add_argument("--e21", required=True); ap.add_argument("--no_sim", action="store_true")
a = ap.parse_args(); Q = a.run_dir; t0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cheap = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]; embc = [f"pc{i}" for i in range(16)]
CONF, B = 0.75, 1000
F = pd.read_parquet(os.path.join(a.e21, "feats_T0.25.parquet")); z = np.load(os.path.join(Q, "hs.npz")); hs, layers = z["hs"].astype(np.float64), [int(x) for x in z["layers"]]
assert (z["pid"] == F.pid.values).all() and hs.shape[0] == len(F) == 2000 and (F.pid.values == np.arange(2000)).all()
pca_info = {}
for li, L in enumerate(layers):
    for pi, pos in enumerate("PA"):
        pcs = PCA(n_components=16, random_state=0); Zp = pcs.fit_transform(StandardScaler().fit_transform(hs[:, li, pi, :]))
        for i in range(16): F[f"hs{pos}{L}_{i}"] = Zp[:, i]
        pca_info[f"hs{pos}{L}"] = float(pcs.explained_variance_ratio_.sum())
F.to_parquet(os.path.join(Q, "feats_T0.25_hs.parquet"), index=False); print("pca explained variance (16 comps):", json.dumps(pca_info), flush=True)
hc = lambda tag: [f"hs{tag}_{i}" for i in range(16)]
SETS = {"cheap": cheap, "cheap_emb": cheap + embc, "emb_only": embc, "cheap_hsA20": cheap + hc("A20"), "cheap_hsP20": cheap + hc("P20"), "hsA20": hc("A20"), "hsP20": hc("P20"),
        "cheap_hsA14": cheap + hc("A14"), "cheap_hsA28": cheap + hc("A28"), "cheap_hsP14": cheap + hc("P14"), "cheap_hsP28": cheap + hc("P28"), "hsA14": hc("A14"), "hsA28": hc("A28")}
PRIMARY, EXPLORATORY = "cheap_hsA20", ["cheap_hsP20", "cheap_hsA14", "cheap_hsA28", "cheap_hsP14", "cheap_hsP28"]

def cv(G):
    """E21 cv_aucs plus out-of-fold predictions: fold AUCs [15] per set and OOF scores [3 repeats, n] per set (identical folds for every set)."""
    y = G.proxy_error.values.astype(int); folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(np.zeros(len(y)), y))
    auc = {k: [] for k in list(SETS) + ["agree"]}; oof = {k: np.zeros((3, len(y))) for k in SETS}
    for f, (tr, te) in enumerate(folds):
        auc["agree"].append(roc_auc_score(y[te], -G.agreement.values[te]))
        for k, cols in SETS.items():
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(G[cols].values[tr], y[tr])
            p = m.predict_proba(G[cols].values[te])[:, 1]; auc[k].append(roc_auc_score(y[te], p)); oof[k][f // 5, te] = p
    return y, {k: np.array(v) for k, v in auc.items()}, oof

def summarize(G, name):
    y, auc, oof = cv(G); n = len(y); rng = np.random.default_rng(0); idx = [rng.integers(0, n, n) for _ in range(B)]; idx = [i for i in idx if 0 < y[i].sum() < len(i)]
    avg = {k: v.mean(0) for k, v in oof.items()}; base_b = np.array([roc_auc_score(y[i], avg["cheap"][i]) for i in idx]); res = {"n": n, "n_err": int(y.sum()), "n_boot_used": len(idx), "auc_agreement_only": float(auc["agree"].mean()), "sets": {}}
    for k in SETS:
        d = auc[k] - auc["cheap"]; r = {"auc": float(auc[k].mean()), "auc_fold_sd": float(auc[k].std()), "delta": float(d.mean()), "delta_fold_sd": float(d.std()), "oof_avg_auc": float(roc_auc_score(y, avg[k]))}
        if k != "cheap":
            db = np.array([roc_auc_score(y[i], avg[k][i]) for i in idx]) - base_b; r.update(oof_avg_delta=float(r["oof_avg_auc"] - roc_auc_score(y, avg["cheap"])), boot_ci95=[float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))], boot_sd=float(db.std()), boot_mean=float(db.mean()))
        res["sets"][k] = r
    print(name, json.dumps({k: (round(v["auc"], 4), round(v["delta"], 4), [round(x, 4) for x in v.get("boot_ci95", [])]) for k, v in res["sets"].items()}), flush=True); return res

conf = F[F.agreement >= CONF].reset_index(drop=True)
assert len(conf) == 1582 and int(conf.proxy_error.sum()) == 94, (len(conf), int(conf.proxy_error.sum()))
res = {"pca_explained_variance_16": pca_info}
res["confident"] = summarize(conf, "confident"); res["overall"] = summarize(F, "overall")

# reproduction gate against the stored E21 numbers (full precision)
e = json.load(open(os.path.join(a.e21, "result.json")))["datasets"]["T0.25"]["confident"]; c = res["confident"]["sets"]
rep = {"e21_auc_cheap": e["auc_cheap"], "here_auc_cheap": c["cheap"]["auc"], "e21_auc_cheap_emb": e["auc_cheap_emb"], "here_auc_cheap_emb": c["cheap_emb"]["auc"],
       "e21_delta": e["delta_auc_emb_over_cheap"], "here_delta": c["cheap_emb"]["delta"], "e21_auc_agreement": e["auc_agreement_only"], "here_auc_agreement": res["confident"]["auc_agreement_only"]}
rep["max_abs_diff"] = float(max(abs(rep["e21_auc_cheap"] - rep["here_auc_cheap"]), abs(rep["e21_auc_cheap_emb"] - rep["here_auc_cheap_emb"]), abs(rep["e21_delta"] - rep["here_delta"]), abs(rep["e21_auc_agreement"] - rep["here_auc_agreement"])))
rep["pass"] = bool(rep["max_abs_diff"] < 1e-6); res["reproduction_gate"] = rep; print("reproduction gate:", json.dumps(rep), flush=True)
if not rep["pass"]:
    json.dump(res, open(os.path.join(Q, "result_stage1.json"), "w"), indent=1); print("GATE FAILED: E21 confident-subset AUCs not reproduced", flush=True); sys.exit(4)

# sanity: hs-only overall AUC
res["sanity"] = {"hsA20_alone_overall_auc": res["overall"]["sets"]["hsA20"]["auc"], "threshold": 0.70, "ok": bool(res["overall"]["sets"]["hsA20"]["auc"] >= 0.70)}

# stage 1 reading (pre-registered in the spec)
p = c[PRIMARY]; ex = {k: c[k]["delta"] for k in EXPLORATORY}
clears = bool(p["delta"] >= 0.05 and p["boot_ci95"][0] > 0)
kill = bool(p["delta"] < 0.02 and p["boot_ci95"][1] < 0.05 and all(v < 0.05 for v in ex.values()))
res["stage1"] = {"primary": PRIMARY, "primary_delta": p["delta"], "primary_delta_fold_sd": p["delta_fold_sd"], "primary_boot_ci95": p["boot_ci95"], "primary_boot_sd": p["boot_sd"], "clears": clears,
                 "kill_conditions": {"primary_delta_lt_0.02": bool(p["delta"] < 0.02), "boot_ci_upper_lt_0.05": bool(p["boot_ci95"][1] < 0.05), "no_exploratory_ge_0.05": bool(all(v < 0.05 for v in ex.values()))},
                 "kill_candidate": kill, "exploratory_deltas": ex, "exploratory_ge_0.05": [k for k, v in ex.items() if v >= 0.05],
                 "bin": "clears" if clears else ("kill" if kill else "inconclusive")}

# free-label ceiling (sim_e21.py, policy `static`), cheap + hsA20 columns, cell100, errors caught at 10/25% vs agree_thr and oracle
if not a.no_sim:
    out = os.path.join(Q, "sim_T0.25_cell100_cheap_hsA20.json"); cols = ",".join(cheap + hc("A20")); t1 = time.time()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "avrl", "exp", "sim_e21.py"), os.path.join(Q, "feats_T0.25_hs.parquet"), "proxy_error", "cell100", cols, out], capture_output=True, text=True)
    open(os.path.join(Q, "sim_T0.25_cell100_cheap_hsA20.log"), "w").write(r.stdout + r.stderr)
    if r.returncode != 0: raise RuntimeError("sim_e21.py failed: " + r.stderr[-800:])
    ceil = {}; ref = {f: json.load(open(os.path.join(a.e21, f"sim_T0.25_cell100_{f}.json"))) for f in ("cheap", "cheap_emb")}; hsj = json.load(open(out))
    for bf in ("0.1", "0.25"):
        thr, orc = hsj[f"budget_{bf}"]["agree_thr"]["mean"], hsj[f"budget_{bf}"]["oracle"]["mean"]; row = {"n_sim": hsj["n_sim"], "pool_errors": hsj["hits"], "uniform": hsj[f"budget_{bf}"]["uniform"]["mean"], "agree_thr": thr, "oracle": orc}
        for nm, j in (("cheap", ref["cheap"]), ("cheap_emb", ref["cheap_emb"]), ("cheap_hsA20", hsj)): s = j[f"budget_{bf}"]["static"]["mean"]; row[f"static_{nm}"] = s; row[f"static_{nm}_over_thr"] = s / thr; row[f"static_{nm}_gap_closed"] = (s - thr) / (orc - thr)
        assert abs(ref["cheap"][f"budget_{bf}"]["agree_thr"]["mean"] - thr) < 1e-9, "agree_thr should be identical across feature sets"; ceil[bf] = row
    res["free_label_ceiling"] = ceil; print("ceiling:", json.dumps(ceil), f"({time.time()-t1:.0f}s)", flush=True)

json.dump(res, open(os.path.join(Q, "result_stage1.json"), "w"), indent=1)
f3 = lambda x: f"{x:.4f}"; L = []
for sub in ("confident", "overall"):
    s = res[sub]; L += [f"## {sub}: n={s['n']}, errors={s['n_err']}, AUC of -agreement alone {f3(s['auc_agreement_only'])}", "", "| feature set | AUC (15-fold mean, fold SD) | delta vs cheap (15-fold mean, fold SD) | delta on averaged OOF | bootstrap 95% CI of OOF delta | boot SD |", "|---|---|---|---|---|---|"]
    for k, v in s["sets"].items():
        L.append(f"| {k} | {f3(v['auc'])} ({f3(v['auc_fold_sd'])}) | " + ("-" if k == "cheap" else f"{v['delta']:+.4f} ({f3(v['delta_fold_sd'])}) | {v['oof_avg_delta']:+.4f} | [{v['boot_ci95'][0]:+.4f}, {v['boot_ci95'][1]:+.4f}] | {f3(v['boot_sd'])}") + (" |" if k == "cheap" else " |"))
    L.append("")
L += ["## Stage 1 reading", "", "```", json.dumps(res["stage1"], indent=1), "```", "", f"Sanity (hsA20 alone, overall AUC): {json.dumps(res['sanity'])}", ""]
open(os.path.join(Q, "stage1_table.md"), "w").write("\n".join(L)); print("\n".join(L), flush=True); print(f"stage 1 done in {time.time()-t0:.0f}s", flush=True)
