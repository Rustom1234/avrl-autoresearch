"""L1a independent verification (CPU). Usage: l1a_verify.py <runs/L1a>. Recomputes the headline numbers from the raw sim_*.json / tab_*.parquet / p0 files with code that does not import or reuse
l1a_summarize.py (plain dict arithmetic, sklearn GroupKFold folds instead of the script's own pid folds) and compares them with result.json. Writes verify.json; exit 1 on any mismatch."""
import sys, os, json, glob
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

rd = sys.argv[1]; res = json.load(open(os.path.join(rd, "result.json"))); out = {"checks": {}}; bad = []
def chk(name, ours, theirs, tol=1e-9):
    ok = (ours == theirs) if isinstance(ours, (bool, str)) else abs(ours - theirs) <= tol
    out["checks"][name] = {"recomputed": ours, "reported": theirs, "ok": bool(ok)}
    if not ok: bad.append(name)
FORB = {"agreement", "n_distinct", "pass_rate", "majority", "majority_correct", "proxy", "proxy_error", "mean_tokens", "std_tokens", "any_truncated", "judge_top", "judge_margin", "judge_mean", "judge_std", "judge_sel", "judge_error"}
CR = ["judge_logit", "log_tokens", "truncated", "has_decimal", "is_negative", "log_abs_ans", "empty_answer", "log_qlen"]; PC = [f"pc{i}" for i in range(16)]

# 1 provenance: no forbidden column anywhere, every sim file is rollout mode with the right feature list and 8 policy seeds
n_files = 0
for f in sorted(glob.glob(os.path.join(rd, "sim_*.json"))):
    d = json.load(open(f)); n_files += 1; fs = "pc" if "_pc_" in os.path.basename(f) else "cheap"
    assert d["mode"] == "rollout" and d["n_policy_seeds"] == 8 and d["features"] == (CR + PC if fs == "pc" else CR) and not (set(d["features"]) & FORB), f
    for b in ("0.1", "0.25", "0.5"):
        for p, v in d[f"budget_{b}"].items(): assert len(v["per_seed"]) == 8 and abs(np.mean(v["per_seed"]) - v["mean"]) < 1e-9
for f in glob.glob(os.path.join(rd, "tab_*.parquet")): assert not (set(pd.read_parquet(f).columns) & FORB), f
chk("n_sim_files", n_files, 72)

# 2 gate quantities straight from the raw p0 / E04 files
r = pd.read_parquet("runs/p0/rollouts.parquet")
for jk, nm, want_auc in (("3B", "Qwen2.5-3B-Instruct", 0.654), ("7B", "Qwen2.5-7B-Instruct-AWQ", 0.702)):
    j = pd.read_parquet(f"runs/E04/judge_{nm}.parquet"); m = r.merge(j, on=["pid", "answer"], how="left"); assert m.judge_p.notna().all()
    chk(f"{jk} rollout AUC rounds to {want_auc}", round(roc_auc_score(1 - m.correct, 1 - m.judge_p), 3), want_auc)
    thr = np.quantile(m.judge_p, 1 - 0.6421); acc = m.judge_p >= thr
    chk(f"{jk}_q corrupted positives", int((acc & (m.correct == 0)).sum()), res["structure"][f"{jk}_q_corrupted"]["positives"])
    chk(f"{jk}_q judge_wrong positives", int((acc.astype(int) != m.correct).sum()), res["structure"][f"{jk}_q_judgewrong"]["positives"])
    if jk == "7B": chk("7B_p05 corrupted positives", int(((m.judge_p > 0.5) & (m.correct == 0)).sum()), res["structure"]["7B_p05_corrupted"]["positives"])

# 3 headline replay numbers from per_seed lists, primary label, cell100, 25% budget
def per(tag, fs, sp, b, p): return np.array(json.load(open(os.path.join(rd, f"sim_{tag}_{fs}_cell100_s{sp}.json")))[f"budget_{b}"][p]["per_seed"], float)
THR = ["thr_judge_low", "thr_boundary", "thr_tokens_long", "thr_trunc_first"]
for rg in ("3B", "7B"):
    tag = f"{rg}_q_corrupted"; H = {}
    for fs in ("cheap", "pc"):
        ts = np.stack([per(tag, fs, s, "0.25", "ts_online") for s in range(4)]); gr = np.stack([per(tag, fs, s, "0.25", "greedy_online") for s in range(4)])
        thr = np.stack([max((per(tag, fs, s, "0.25", p) for p in THR), key=lambda a: a.mean()) for s in range(4)])
        H[fs] = (ts, gr, thr)
        R = res["replay"][tag]["cell100"]["0.25"]["ratios"][fs]
        chk(f"{rg} {fs} pooled ts/greedy @25%", float(ts.mean() / gr.mean()), R["ts_vs_greedy"]["ratio"], 1e-9)
        chk(f"{rg} {fs} pooled greedy/best_thr @25%", float(gr.mean() / thr.mean()), R["greedy_vs_best_thr"]["ratio"], 1e-9)
        chk(f"{rg} {fs} pooled ts/best_thr @25%", float(ts.mean() / thr.mean()), R["ts_vs_best_thr"]["ratio"], 1e-9)
        sp_n = sum(1 for s in range(4) if ts[s].mean() / gr[s].mean() >= 1.05 and (ts[s].mean() - gr[s].mean()) / np.hypot(ts[s].std(), gr[s].std()) > 2)
        chk(f"{rg} {fs} splits with ts>=1.05 greedy and >2 SD", sp_n, res["verdict"]["rungs"][rg]["b_detail"][fs]["n_splits"])
    chk(f"{rg} pooled greedy(pc)/greedy(cheap) @25%", float(H["pc"][1].mean() / H["cheap"][1].mean()), res["verdict"]["rungs"][rg]["pooled_greedy_pc_vs_greedy_cheap"], 1e-9)
    rec = np.stack([max((per(tag, "cheap", s, "0.25", p) for p in ("recal_charged_100", "recal_charged_200", "recal_charged_400")), key=lambda a: a.mean()) for s in range(4)])
    chk(f"{rg} cheap greedy/best_recal @25%", float(H["cheap"][1].mean() / rec.mean()), res["replay"][tag]["cell100"]["0.25"]["ratios"]["cheap"]["greedy_vs_best_recal"]["ratio"], 1e-9)

# 4 structure: independent sklearn GroupKFold OOF AUC for cheap_r and +PCs (different fold assignment than the script, so agreement is to ~0.003, not exact)
for rg in ("3B", "7B"):
    tag = f"{rg}_q_corrupted"; t = pd.read_parquet(os.path.join(rd, f"tab_{tag}.parquet")); y = t.corrupted.values; g = t.pid.values; a = {}
    for nm, cols in (("cheap", CR), ("pc", CR + PC)):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(t, y, g):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(t.iloc[tr][cols].values, y[tr]); p[te] = m.predict_proba(t.iloc[te][cols].values)[:, 1]
        a[nm] = roc_auc_score(y, p)
    S = res["structure"][tag]
    chk(f"{rg} cheap_r OOF AUC (GroupKFold vs script, tol 0.003)", float(a["cheap"]), S["cheap_r_auc"], 0.003)
    chk(f"{rg} delta AUC +PCs (GroupKFold vs script, tol 0.004)", float(a["pc"] - a["cheap"]), S["delta"]["pc"]["mean"], 0.004)

# 5 verdict flags recomputed from the stored deltas and ratios
V = res["verdict"]; gate = bool(res["build"]["gate_passed"] and res["regression_gate"]["passed"] and res["regression_gate"]["differences"] == 0)
ak = {}; bk = {}
for rg in ("3B", "7B"):
    S = res["structure"][f"{rg}_q_corrupted"]["delta"]; x = V["rungs"][rg]
    ak[rg] = S["pc"]["mean"] < 0.02 and S["pc"]["ci_hi"] < 0.03 and all(S[c]["mean"] < 0.01 for c in ("cell30", "cell100", "cell300"))
    bk[rg] = all(x["b_detail"][fs]["pooled_ts_vs_greedy"] <= 1.03 for fs in ("cheap", "pc")) and x["pooled_greedy_pc_vs_greedy_cheap"] <= 1.03
chk("KILL verdict recomputed", bool(gate and all(ak.values()) and all(bk.values())), V["KILL"])
chk("PASS verdict recomputed", False, V["PASS"])
out["bad"] = bad; json.dump(out, open(os.path.join(rd, "verify.json"), "w"), indent=1)
for k, v in out["checks"].items(): print(("OK  " if v["ok"] else "BAD ") + k, v["recomputed"], v["reported"])
print("MISMATCHES:", bad); sys.exit(1 if bad else 0)
