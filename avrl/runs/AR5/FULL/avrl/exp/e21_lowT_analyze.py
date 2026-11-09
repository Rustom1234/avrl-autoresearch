"""E21 analysis (CPU). Copy of p0_analyze.py adapted to: --run_dir (T0.25, T0.5 groups from e21_lowT_rollouts.py), the paired T=1.0 reference (first
n_prompts pids of --ref_run), CPU MiniLM, repeated stratified 5-fold x 3 CV with identical folds across feature sets, and the replay in exp/sim_e21.py
(which adds the label-free agreement-threshold policy). Writes <run_dir>/result.json, <run_dir>/table.md, feats_*.parquet, sim_*.json."""
import os, sys, json, subprocess, argparse, glob
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from aa import data as D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True); ap.add_argument("--ref_run", required=True); ap.add_argument("--n_prompts", type=int, default=2000)
a = ap.parse_args(); Q = a.run_dir
cheap = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]; embc = [f"pc{i}" for i in range(16)]
CONF = 0.75; CELLS = (30, 100, 300); BUD = (0.10, 0.25, 0.50)

# datasets: name -> groups frame (pid 0..n-1, same prompts in every dataset)
ds = {}
for T in ("0.25", "0.5"):
    p = os.path.join(Q, f"T{T}", "groups.parquet")
    if os.path.exists(p): ds[f"T{T}"] = pd.read_parquet(p).sort_values("pid").reset_index(drop=True)
ref = pd.read_parquet(os.path.join(a.ref_run, "groups.parquet")); ref = ref[ref.pid < a.n_prompts].sort_values("pid").reset_index(drop=True)
ds["T1.0"] = ref[[c for c in ref.columns if not c.startswith(("cell", "pc"))]]
for k, g in ds.items():
    assert len(g) == a.n_prompts and (g.pid.values == np.arange(a.n_prompts)).all() and (g.question.values == ref.question.values).all() and (g.gold.values == ref.gold.values).all(), f"{k} not paired with E01 prompts"
print("datasets:", {k: len(v) for k, v in ds.items()}, flush=True)

# one embedding / PCA / KMeans on the shared prompts, reused by every temperature (identical cells and PCs across datasets)
epath = os.path.join(Q, "q_emb_minilm.npy")
if os.path.exists(epath): emb = np.load(epath)
else:
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu").encode(ref.question.tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    np.save(epath, emb)
cells = {k: KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(emb) for k in CELLS}
pcs = PCA(n_components=16, random_state=0).fit_transform(emb)
for name, G in ds.items():
    for k in CELLS: G[f"cell{k}"] = cells[k]
    for i in range(16): G[f"pc{i}"] = pcs[:, i]
    G["log_qlen"] = np.log1p(G.q_len); G["split"] = (G.agreement < 1).astype(int)
    G.to_parquet(os.path.join(Q, f"feats_{name}.parquet"), index=False)

def cv_aucs(G, sets, y):
    """Fold-level AUCs, identical (repeated stratified 5-fold x 3) folds for every feature set; 'agree' = label-free score -agreement, no fit."""
    out = {k: [] for k in list(sets) + ["agree"]}
    if y.sum() < 5 or (len(y) - y.sum()) < 5: return {k: np.full(1, np.nan) for k in out}
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(np.zeros(len(y)), y):
        out["agree"].append(roc_auc_score(y[te], -G.agreement.values[te]))
        for k, cols in sets.items():
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(G[cols].values[tr], y[tr])
            out[k].append(roc_auc_score(y[te], m.predict_proba(G[cols].values[te])[:, 1]))
    return {k: np.array(v) for k, v in out.items()}
sets = {"cheap": cheap, "cheap_emb": cheap + embc, "emb_only": embc}
def summarize(G):
    y = G.proxy_error.values.astype(int); cv = cv_aucs(G, sets, y)
    d = {"n": int(len(G)), "n_err": int(y.sum()), "error_rate": float(y.mean()),
         "auc_agreement_only": float(np.mean(cv["agree"])), "auc_agreement_only_direct": float(roc_auc_score(y, -G.agreement.values)) if 0 < y.sum() < len(y) else float("nan"),
         "auc_cheap": float(np.mean(cv["cheap"])), "auc_cheap_emb": float(np.mean(cv["cheap_emb"])), "auc_emb_only": float(np.mean(cv["emb_only"])),
         "delta_auc_emb_over_cheap": float(np.mean(cv["cheap_emb"] - cv["cheap"])), "delta_auc_fold_sd": float(np.std(cv["cheap_emb"] - cv["cheap"])),
         "delta_auc_cheap_over_agree": float(np.mean(cv["cheap"] - cv["agree"])), "auc_cheap_fold_sd": float(np.std(cv["cheap"])), "n_cv_folds": int(len(cv["cheap"])),
         "neighbor_hit_ratio": {f"cell{k}": D.neighbor_hit_ratio(y, G[f"cell{k}"].values)["ratio"] for k in CELLS}}
    # permutation null for the cell100 ratio (how big is the ratio by chance at this n_err); not part of the pre-registered verdict
    rng = np.random.default_rng(0); nl = [D.neighbor_hit_ratio(rng.permutation(y), G["cell100"].values)["ratio"] for _ in range(200)] if y.sum() >= 2 else [float("nan")]
    d["neighbor_hit_ratio_cell100_null_mean"] = float(np.nanmean(nl)); d["neighbor_hit_ratio_cell100_null_p95"] = float(np.nanpercentile(nl, 95))
    return d
res = {"datasets": {}}
for name, G in ds.items():
    conf = G[G.agreement >= CONF].reset_index(drop=True)
    res["datasets"][name] = {"overall": summarize(G), "confident": summarize(conf), "confident_frac": float(len(conf) / len(G)),
                             "rollout_accuracy": float(G.pass_rate.mean()), "majority_accuracy": float(G.majority_correct.mean()),
                             "frac_unanimous": float((G.agreement == 1).mean()), "mean_tokens": float(G.mean_tokens.mean()), "frac_groups_any_truncated": float(G.any_truncated.mean())}
    print(name, json.dumps({k: res["datasets"][name][k] for k in ("overall", "confident")}, indent=1), flush=True)

# allocation replay: dataset x cell x feature set, sim_e21.py (label-free agreement threshold added)
jobs = [(n, g, fn) for n in ds for g in ("cell100", "cell30") for fn in ("cheap", "cheap_emb")]
def runsim(j):
    n, g, fn = j; cols = cheap if fn == "cheap" else cheap + embc; o = os.path.join(Q, f"sim_{n}_{g}_{fn}.json")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "avrl", "exp", "sim_e21.py"), os.path.join(Q, f"feats_{n}.parquet"), "proxy_error", g, ",".join(cols), o], capture_output=True, text=True)
    open(os.path.join(Q, f"sim_{n}_{g}_{fn}.log"), "w").write(r.stdout + r.stderr)
    if r.returncode != 0: raise RuntimeError(f"sim failed {j}: {r.stderr[-800:]}")
    return j, json.load(open(o))
with ThreadPoolExecutor(4) as ex: sims = dict(ex.map(runsim, jobs))
def replay(name, g, fn):
    s = sims[(name, g, fn)]; rows = {}
    for bf in BUD:
        r = s[f"budget_{bf}"]; thr, orc, uni = r["agree_thr"], r["oracle"], r["uniform"]
        lp = max(("greedy_online", "ts_online"), key=lambda p: r[p]["mean"]); L = r[lp]
        diff = np.array(L["seeds"]) - np.array(thr["seeds"])
        rows[f"{bf:.2f}"] = {"uniform": uni["mean"], "agree_thr": thr["mean"], "agree_thr_sd": thr["sd"], "static_free_labels": r["static"]["mean"], "greedy_online": r["greedy_online"]["mean"], "greedy_online_sd": r["greedy_online"]["sd"],
                             "ts_online": r["ts_online"]["mean"], "ts_online_sd": r["ts_online"]["sd"], "learned_policy": lp, "learned": L["mean"], "learned_sd": L["sd"], "oracle": orc["mean"],
                             "learned_over_thr": L["mean"] / max(thr["mean"], 1e-9), "gap_learned_minus_thr": float(L["mean"] - thr["mean"]), "gap_paired_sd": float(diff.std()),
                             "gap_over_2sd": bool((L["mean"] - thr["mean"]) > 2 * max(L["sd"], diff.std())), "H": float((orc["mean"] - thr["mean"]) / max(orc["mean"], 1e-9)),
                             "n_sim": s["n_sim"], "hits_in_sim": s["hits"]}
    return rows
for name in ds:
    res["datasets"][name]["replay"] = {f"{g}_{fn}": replay(name, g, fn) for g in ("cell100", "cell30") for fn in ("cheap", "cheap_emb")}

# verdict against the pre-registered thresholds (primary replay variant: cell100 + cheap_emb; sensitivity: best learned over all four variants)
def best_over_variants(name, b):
    return max(res["datasets"][name]["replay"][v][b]["learned_over_thr"] for v in res["datasets"][name]["replay"])
V = {"per_temp": {}}
for T in ("T0.25", "T0.5"):
    if T not in ds: continue
    d = res["datasets"][T]; c = d["confident"]; rp = d["replay"]["cell100_cheap_emb"]
    V["per_temp"][T] = {"n_conf_err": c["n_err"], "powered": c["n_err"] >= 100, "conf_delta_auc": c["delta_auc_emb_over_cheap"], "conf_nhr_cell100": c["neighbor_hit_ratio"]["cell100"],
                        "routeA": bool(c["n_err"] >= 100 and c["delta_auc_emb_over_cheap"] >= 0.05 and c["neighbor_hit_ratio"]["cell100"] >= 1.5),
                        "learned_over_thr_10": rp["0.10"]["learned_over_thr"], "learned_over_thr_25": rp["0.25"]["learned_over_thr"], "learned_over_thr_50": rp["0.50"]["learned_over_thr"],
                        "routeB": bool(rp["0.25"]["learned_over_thr"] >= 1.10 and rp["0.25"]["gap_over_2sd"]), "H25": rp["0.25"]["H"], "H25_ok": bool(rp["0.25"]["H"] >= 0.10),
                        "best_learned_over_thr_25_any_variant": best_over_variants(T, "0.25"), "best_learned_over_thr_10_any_variant": best_over_variants(T, "0.10"),
                        "kill_cond": bool(c["delta_auc_emb_over_cheap"] < 0.03 and c["neighbor_hit_ratio"]["cell100"] < 1.3 and rp["0.10"]["learned_over_thr"] <= 1.03 and rp["0.25"]["learned_over_thr"] <= 1.03)}
pt = V["per_temp"]
V["pass"] = any((v["routeA"] or v["routeB"]) and v["H25_ok"] for v in pt.values())
V["kill"] = bool("T0.25" in pt and pt["T0.25"]["powered"] and len(pt) == 2 and all(v["kill_cond"] for v in pt.values()))
V["underpowered"] = bool("T0.25" not in pt or not pt["T0.25"]["powered"])
V["verdict"] = "pass (claim 3 revived)" if V["pass"] else ("kill (claim 3 dead with adequate power; also kills claim 2 offline replay)" if V["kill"] else ("underpowered, not a kill" if V["underpowered"] else "ambiguous"))
res["verdict"] = V
json.dump(res, open(os.path.join(Q, "result.json"), "w"), indent=1, default=float)

# table
L = ["| dataset | subset | n | n_err | err rate | AUC agree | AUC cheap | AUC cheap+emb | AUC emb | delta emb/cheap (fold sd) | NHR cell30 / 100 / 300 | NHR100 null mean / p95 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for n in ds:
    for s in ("overall", "confident"):
        c = res["datasets"][n][s]; h = c["neighbor_hit_ratio"]
        L.append(f"| {n} | {s} | {c['n']} | {c['n_err']} | {c['error_rate']:.3f} | {c['auc_agreement_only']:.3f} | {c['auc_cheap']:.3f} | {c['auc_cheap_emb']:.3f} | {c['auc_emb_only']:.3f} | {c['delta_auc_emb_over_cheap']:+.3f} ({c['delta_auc_fold_sd']:.3f}) | {h['cell30']:.2f} / {h['cell100']:.2f} / {h['cell300']:.2f} | {c['neighbor_hit_ratio_cell100_null_mean']:.2f} / {c['neighbor_hit_ratio_cell100_null_p95']:.2f} |")
L += ["", "Replay (cell100, cheap+emb features; errors caught, mean over 8 seeds; learned = better of greedy_online / ts_online):", "",
      "| dataset | budget | uniform | label-free agree thr | static free labels | greedy_online | ts_online | learned | learned/thr | oracle | H=(orc-thr)/orc |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for n in ds:
    for b, r in res["datasets"][n]["replay"]["cell100_cheap_emb"].items():
        L.append(f"| {n} | {float(b):.0%} | {r['uniform']:.0f} | {r['agree_thr']:.1f} | {r['static_free_labels']:.0f} | {r['greedy_online']:.1f} | {r['ts_online']:.1f} | {r['learned']:.1f} ({r['learned_policy']}) | {r['learned_over_thr']:.3f} | {r['oracle']:.0f} | {r['H']:.3f} |")
L += ["", "Verdict: " + V["verdict"], "", "```", json.dumps(V, indent=1, default=float), "```"]
open(os.path.join(Q, "table.md"), "w").write("\n".join(L)); print("\n".join(L), flush=True)
