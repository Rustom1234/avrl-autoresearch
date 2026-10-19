"""E04 step 2 (CPU only): per judge rung, does an answer-only LLM judge proxy have prompt-space error structure that cheap features miss,
and does the learned allocator beat the best label-free threshold? Same 3,000 E01 groups, reuses runs/p0/feats.parquet (MiniLM PCs and cells).
Usage: e04_analyze.py --feats runs/p0/feats.parquet --judges runs/E04 --out runs/E04

Pre-registered choices (fixed before any judge result existed):
  * primary rung = Qwen2.5-3B-Instruct if valid (rollout AUC >= 0.65 and judge_error count >= 100), else Qwen2.5-7B-Instruct-AWQ if valid, else uninformative.
  * primary replay configuration = groups cell100, features cheap_j+emb, 25 percent budget; the other three (cell30, cheap_j only) are reported as robustness.
  * "best label-free threshold" = whichever of thr_agreement (agreement ascending) and thr_margin (judge_margin ascending) catches more at that budget.
  * gap significance = (ts_online mean - best threshold mean) > 2 * sqrt(sd_online^2 + sd_threshold^2) over the 8 replay seeds.
"""
import os, sys, json, glob, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from aa import data as D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--feats", required=True); ap.add_argument("--judges", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--rollouts", default=None); ap.add_argument("--e01", default=None); ap.add_argument("--no_sim", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
P0 = os.path.dirname(os.path.abspath(a.feats))
R0 = pd.read_parquet(a.rollouts or os.path.join(P0, "rollouts.parquet")); F = pd.read_parquet(a.feats)
E01_PUB = json.load(open(a.e01 or os.path.join(P0, "result.json")))
assert (F.pid.values == np.arange(len(F))).all()
CHEAP = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
JF = ["judge_top", "judge_margin", "judge_mean", "judge_std"]; EMBC = [f"pc{i}" for i in range(16)]
SEEDS = (0, 1, 2); BF = (0.10, 0.25, 0.50)

def cv_auc(X, y):
    """5-fold stratified CV logistic regression (balanced), repeated over 3 shuffle seeds. Returns mean and SD across seeds."""
    s = [cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")), X, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=sd), scoring="roc_auc").mean() for sd in SEEDS]
    return float(np.mean(s)), float(np.std(s))
def conc(yv, g):
    agg = pd.DataFrame({"y": yv, "g": g}).groupby("g")["y"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    cd = agg["count"].cumsum() / agg["count"].sum(); ch = agg["sum"].cumsum() / max(agg["sum"].sum(), 1)
    return float(ch[cd >= 0.10].iloc[0])

def cheap_vs_emb(Fd, y, cheap_cols):
    o = {}
    o["auc_cheap_x"], o["auc_cheap_x_sd"] = cv_auc(Fd[cheap_cols].values, y)
    o["auc_cheap_x_emb"], o["auc_cheap_x_emb_sd"] = cv_auc(Fd[cheap_cols + EMBC].values, y)
    o["delta_auc"] = o["auc_cheap_x_emb"] - o["auc_cheap_x"]
    o["auc_emb_only"], o["auc_emb_only_sd"] = cv_auc(Fd[EMBC].values, y)
    o["neighbor_ratio"] = {f"cell{k}": D.neighbor_hit_ratio(y, Fd[f"cell{k}"].values)["ratio"] for k in (30, 100, 300)}
    o["top10pct_cells_hold"] = {f"cell{k}": conc(y, Fd[f"cell{k}"].values) for k in (30, 100, 300)}
    return o

# ---- simulator: sim_generic.py plus two explicit label-free threshold policies, budgets 10/25/50 percent, K=1 as in E01
def build_sim(path):
    s = open(os.path.join(ROOT, "runs", "sim_generic.py")).read()
    def rep(old, new):
        nonlocal s; assert s.count(old) == 1, old; s = s.replace(old, new)
    rep("BUDGET_FRACS = 3, 4, 8, (0.01, 0.03, 0.10)", "BUDGET_FRACS = 3, 1, 8, (0.10, 0.25, 0.50)")
    rep('    elif policy == "oracle": order = np.argsort(-(y + 1e-6 * rng.random(len(y))))',
        '    elif policy == "oracle": order = np.argsort(-(y + 1e-6 * rng.random(len(y))))\n'
        '    elif policy == "thr_agreement": order = np.lexsort((rng.random(len(y)), sim["agreement"].values))  # low agreement first, random ties, no labels\n'
        '    elif policy == "thr_margin": order = np.lexsort((rng.random(len(y)), sim["judge_margin"].values))  # low judge margin first, random ties, no labels\n'
        '    elif policy == "thr_margin_split": order = np.lexsort((rng.random(len(y)), np.where(sim["n_distinct"].values == 1, 1.0, sim["judge_margin"].values)))  # margin, unanimous groups last (robustness)\n'
        '    elif policy == "thr_agree_margin": order = np.lexsort((rng.random(len(y)), np.where(sim["n_distinct"].values == 1, 1.0, sim["judge_margin"].values), sim["agreement"].values))  # agreement, ties by margin (robustness)')
    rep('if policy in ("uniform", "static", "static_matched", "oracle"):', 'if policy in ("uniform", "static", "static_matched", "oracle", "thr_agreement", "thr_margin", "thr_margin_split", "thr_agree_margin"):')
    rep('    for pol in ["uniform", "static", "static_matched", "ts_flat", "greedy_online", "ts_online", "ts_static", "neighbor", "oracle"]:',
        '    for pol in POLS:')
    rep('out = {"n_sim": int(len(sim))', 'POLS = ["uniform", "static", "static_matched", "ts_flat", "greedy_online", "ts_online", "ts_static", "neighbor", "oracle", "thr_agreement"] + (["thr_margin", "thr_margin_split", "thr_agree_margin"] if "judge_margin" in sim else [])\n'
        'out = {"n_sim": int(len(sim))')
    rep('row[pol] = {"mean": float(np.mean(r)), "sd": float(np.std(r))}', 'row[pol] = {"mean": float(np.mean(r)), "sd": float(np.std(r)), "per_seed": [int(v) for v in r]}')
    rep('for p in ["static", "static_matched", "ts_flat", "greedy_online", "ts_online", "ts_static", "neighbor", "oracle"]', 'for p in POLS[1:]')
    open(path, "w").write(s)

def run_sims(tag, feats_path, label, cheap_cols):
    sim_py = os.path.join(a.out, "sim_alloc.py")
    jobs = [(g, nm, cols) for g in ("cell100", "cell30") for nm, cols in (("cheap", cheap_cols), ("cheap+emb", cheap_cols + EMBC))]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    def one(j):
        g, nm, cols = j; outj = os.path.join(a.out, f"sim_{tag}_{g}_{nm}.json")
        with open(os.path.join(a.out, f"sim_{tag}_{g}_{nm}.log"), "w") as lf:
            subprocess.run([sys.executable, sim_py, feats_path, label, g, ",".join(cols), outj], stdout=lf, stderr=subprocess.STDOUT, env=env, check=True)
        return (g, nm), json.load(open(outj))
    with ThreadPoolExecutor(4) as ex: return dict(ex.map(one, jobs))

def replay_summary(sims):
    """Per configuration and budget: learned (ts_online) vs best label-free threshold, ratio and SD, plus the other arms."""
    out = {}
    for (g, nm), s in sims.items():
        cfg = {}
        for bf in BF:
            row = s[f"budget_{bf}"]; thr = {p: row[p] for p in ("thr_agreement", "thr_margin") if p in row}
            bp = max(thr, key=lambda p: thr[p]["mean"]); L, T = row["ts_online"], thr[bp]
            ratio = L["mean"] / max(T["mean"], 1e-9); rsd = ratio * float(np.sqrt((L["sd"] / max(L["mean"], 1e-9)) ** 2 + (T["sd"] / max(T["mean"], 1e-9)) ** 2))
            gap = L["mean"] - T["mean"]; gsd = float(np.sqrt(L["sd"] ** 2 + T["sd"] ** 2))
            aug = {p: row[p] for p in ("thr_agreement", "thr_margin", "thr_margin_split", "thr_agree_margin") if p in row}; ba = max(aug, key=lambda p: aug[p]["mean"]); TA = aug[ba]
            agap = L["mean"] - TA["mean"]; agsd = float(np.sqrt(L["sd"] ** 2 + TA["sd"] ** 2))
            cfg[str(bf)] = {"ts_online": L["mean"], "ts_online_sd": L["sd"], "best_threshold": bp, "best_threshold_mean": T["mean"], "best_threshold_sd": T["sd"],
                            "ratio_learned_over_best_threshold": ratio, "ratio_sd": rsd, "gap": gap, "gap_over_2sd": bool(gap > 2 * gsd), "gap_in_pooled_sd": float(gap / gsd) if gsd > 0 else None,
                            "thr_agreement": thr["thr_agreement"]["mean"], "thr_margin": thr["thr_margin"]["mean"] if "thr_margin" in thr else None,
                            "static_matched": row["static_matched"]["mean"], "static_free_labels": row["static"]["mean"], "greedy_online": row["greedy_online"]["mean"],
                            "best_threshold_aug": ba, "best_threshold_aug_mean": TA["mean"], "ratio_aug": L["mean"] / max(TA["mean"], 1e-9), "gap_over_2sd_aug": bool(agap > 2 * agsd),
                            "thr_margin_split": aug["thr_margin_split"]["mean"] if "thr_margin_split" in aug else None, "thr_agree_margin": aug["thr_agree_margin"]["mean"] if "thr_agree_margin" in aug else None,
                            "ts_flat": row["ts_flat"]["mean"], "neighbor": row["neighbor"]["mean"], "uniform": row["uniform"]["mean"], "oracle": row["oracle"]["mean"],
                            "n_sim": s["n_sim"], "n_errors_in_sim": s["hits"]}
        out[f"{g}|{nm}"] = cfg
    return out

if not a.no_sim: build_sim(os.path.join(a.out, "sim_alloc.py"))
G = F.copy(); G["split"] = (G.agreement < 1).astype(int)
result = {"floor_oracle_best_of_8_miss": float((G.pass_rate == 0).mean()), "majority_vote_error": float(G.proxy_error.mean()), "e01_published": E01_PUB, "rungs": {}}

# ---- E01 reference, recomputed under this script's CV scheme and replayed with the explicit threshold policy
y0 = G.proxy_error.values.astype(int)
e01 = {"n_groups": len(G), "error_rate": float(y0.mean()), "n_errors": int(y0.sum()), "auc_agreement_only": float(roc_auc_score(y0, -G.agreement.values))}
e01.update(cheap_vs_emb(G, y0, CHEAP)); e01["jaccard_with_e01_proxy_error"] = 1.0
if not a.no_sim:
    G.to_parquet(os.path.join(a.out, "feats_E01.parquet"), index=False)
    e01["replay"] = replay_summary(run_sims("E01", os.path.join(a.out, "feats_E01.parquet"), "proxy_error", CHEAP))
result["E01_reference"] = e01; print("E01 reference done", flush=True)

# ---- judge rungs
ORDER = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct-AWQ"]
files = {os.path.basename(f)[len("judge_"):-len(".parquet")]: f for f in glob.glob(os.path.join(a.judges, "judge_*.parquet"))}
for name in [n for n in ORDER if n in files] + sorted(n for n in files if n not in ORDER):
    J = pd.read_parquet(files[name]); Rj = R0.merge(J, on=["pid", "answer"], how="left")
    assert len(Rj) == len(R0) and Rj.judge_p.notna().all(), f"{name}: judge_p missing for some rollouts"
    rung = {"judge": name, "n_pairs_scored": int(len(J))}
    # validity: rollout-level, unique-pair-level, and within-split-groups rollout-level (where selection actually matters)
    rung["validity_auc_rollout"] = float(roc_auc_score(Rj.correct, Rj.judge_p))
    pr = Rj.drop_duplicates(["pid", "answer"]); rung["validity_auc_unique_pair"] = float(roc_auc_score(pr.correct, pr.judge_p)) if pr.correct.nunique() > 1 else None
    sp = Rj[Rj.pid.isin(G.pid[G.agreement < 1])]; rung["validity_auc_rollout_split_groups"] = float(roc_auc_score(sp.correct, sp.judge_p)) if sp.correct.nunique() > 1 else None
    rung["judge_p_mean"] = float(Rj.judge_p.mean()); rung["judge_p_frac_gt_half"] = float((Rj.judge_p > 0.5).mean()); rung["judge_p_n_distinct"] = int(Rj.judge_p.round(6).nunique())
    # per-group judge features
    gagg = Rj.groupby(["pid", "answer"]).agg(n=("k", "size"), p=("judge_p", "first")).reset_index()
    rows = []
    for pid, d in gagg.groupby("pid"):
        d = d.sort_values(["p", "n", "answer"], ascending=[False, False, True]); p = d.p.values; n = d.n.values; tot = n.sum()
        mean = float((p * n).sum() / tot); std = float(np.sqrt((n * (p - mean) ** 2).sum() / tot))
        rows.append(dict(pid=pid, judge_sel=d.answer.values[0], judge_top=float(p[0]), judge_margin=float(p[0] - p[1]) if len(p) > 1 else 0.0, judge_mean=mean, judge_std=std))
    Gj = G.merge(pd.DataFrame(rows), on="pid", how="left"); assert Gj.judge_sel.notna().all()
    Gj["judge_error"] = (Gj.judge_sel != Gj.gold).astype(int); y = Gj.judge_error.values
    rung.update({"judge_error_rate": float(y.mean()), "judge_error_count": int(y.sum()), "n_groups": int(len(Gj)),
                 "gate_valid": bool(rung["validity_auc_rollout"] >= 0.65 and y.sum() >= 100),
                 "jaccard_with_e01_proxy_error": float((y & y0).sum() / max(((y | y0)).sum(), 1)), "n_both_wrong": int((y & y0).sum()),
                 "judge_error_given_majority_wrong": float(y[y0 == 1].mean()), "judge_error_given_majority_right": float(y[y0 == 0].mean()),
                 "auc_agreement_only": float(roc_auc_score(y, -Gj.agreement.values))})
    cheap_j = CHEAP + JF
    rung["auc_judge_only"], rung["auc_judge_only_sd"] = cv_auc(Gj[JF].values, y)
    rung["auc_cheap"], rung["auc_cheap_sd"] = cv_auc(Gj[CHEAP].values, y)
    rung["auc_cheap_j"], rung["auc_cheap_j_sd"] = cv_auc(Gj[cheap_j].values, y)
    rung["auc_cheap_j_emb"], rung["auc_cheap_j_emb_sd"] = cv_auc(Gj[cheap_j + EMBC].values, y)
    rung["delta_auc"] = rung["auc_cheap_j_emb"] - rung["auc_cheap_j"]
    rung["auc_emb_only"], rung["auc_emb_only_sd"] = cv_auc(Gj[EMBC].values, y)
    rung["neighbor_ratio"] = {f"cell{k}": D.neighbor_hit_ratio(y, Gj[f"cell{k}"].values)["ratio"] for k in (30, 100, 300)}
    rung["top10pct_cells_hold"] = {f"cell{k}": conc(y, Gj[f"cell{k}"].values) for k in (30, 100, 300)}
    Gj.to_parquet(os.path.join(a.out, f"feats_{name}.parquet"), index=False)
    if not a.no_sim: rung["replay"] = replay_summary(run_sims(name, os.path.join(a.out, f"feats_{name}.parquet"), "judge_error", cheap_j))
    result["rungs"][name] = rung; print(f"{name}: valid={rung['gate_valid']} auc_roll={rung['validity_auc_rollout']:.3f} err={rung['judge_error_rate']:.4f} delta={rung['delta_auc']:.4f} nbr100={rung['neighbor_ratio']['cell100']:.3f}", flush=True)

# ---- pre-registered verdict
PRIM = ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct-AWQ"); primary = next((n for n in PRIM if n in result["rungs"] and result["rungs"][n]["gate_valid"]), None)
v = {"primary_rung": primary, "fallback_used": bool(primary is not None and primary != PRIM[0]), "primary_config": "cell100|cheap+emb @ 25 percent budget"}
if primary is None:
    v["verdict"] = "uninformative: neither the 3B nor the 7B-AWQ rung passed the validity gate (rollout AUC >= 0.65 and judge_error count >= 100) or both are missing"
else:
    r = result["rungs"][primary]; rp = r.get("replay", {}).get("cell100|cheap+emb", {}).get("0.25")
    v["structure_present"] = bool(r["delta_auc"] >= 0.05 or r["neighbor_ratio"]["cell100"] >= 1.5)
    if rp:
        ratio = rp["ratio_learned_over_best_threshold"]; v["ratio_25"] = ratio; v["gap_over_2sd"] = rp["gap_over_2sd"]
        v["ratio_25_augmented_thresholds"] = rp["ratio_aug"]; v["pass"] = bool(v["structure_present"] and ratio >= 1.10 and rp["gap_over_2sd"])
        v["kill"] = bool(r["delta_auc"] < 0.05 and r["neighbor_ratio"]["cell100"] < 1.5 and ratio <= 1.05)
        v["any_config_ratio_ge_1.10_at_25"] = {k: c["0.25"]["ratio_learned_over_best_threshold"] for k, c in r["replay"].items()}
        v["verdict"] = "PASS (first positive point for claim 3)" if v["pass"] else ("KILL (claim 3 dead for majority-vote and answer-only judge proxies on GSM8K)" if v["kill"] else "neither pass nor kill (mixed)")
    else: v["verdict"] = "replay missing"
    v["jaccard_with_e01"] = r["jaccard_with_e01_proxy_error"]
    v["jaccard_read"] = "proxy choice is a lever (<0.5)" if r["jaccard_with_e01_proxy_error"] < 0.5 else ("judge adds nothing beyond E01 (>0.8)" if r["jaccard_with_e01_proxy_error"] > 0.8 else "between 0.5 and 0.8")
result["verdict"] = v

# ---- tentative G03 point: structure statistic versus (learned minus best threshold) at 25 percent, primary config, E01 plus every judge rung
pts = []
if not a.no_sim:
    def pt(nm, d):
        c = d["replay"]["cell100|cheap+emb"]["0.25"]; return dict(name=nm, delta_auc=d["delta_auc"], neighbor_ratio_cell100=d["neighbor_ratio"]["cell100"], gain_vs_threshold=c["ratio_learned_over_best_threshold"] - 1, gain_vs_threshold_aug=c["ratio_aug"] - 1)
    pts = [pt("E01", e01)] + [pt(n, result["rungs"][n]) for n in result["rungs"]]
g03 = {"points": pts, "note": f"n={len(pts)}, tentative first point for G03, NOT significant (Spearman on so few points cannot reach significance in any useful sense)"}
if len(pts) >= 3:
    for st in ("delta_auc", "neighbor_ratio_cell100"):
        rho, pv = spearmanr([p[st] for p in pts], [p["gain_vs_threshold"] for p in pts]); g03[f"spearman_{st}_vs_gain"] = {"rho": None if np.isnan(rho) else float(rho), "p": None if np.isnan(pv) else float(pv)}
result["G03_tentative"] = g03
json.dump(result, open(os.path.join(a.out, "result.json"), "w"), indent=1)
print(json.dumps({"verdict": v, "G03": g03}, indent=1), flush=True)
