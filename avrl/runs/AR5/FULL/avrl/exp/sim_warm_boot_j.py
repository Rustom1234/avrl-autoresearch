"""L3b judge-proxy pool bootstrap. Copy of sim_warm_boot.py (L2a) with: allocator label J (column given as argv[2], here `judge_error` = judge accepts the
majority completion AND gold wrong); pool = ALL groups of feats_J.parquet (2,000; env POOL=held restores the E21 held-out-cells quarter split, 1,546 groups, as a robustness read);
label-free fixed policies agree_thr, judge_thr, blend50; learner greedy_warm / greedy_blend warm-started from the blend50 ordering; bootstrap only (no main table, no TS arms).
Usage: BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 sim_warm_boot_j.py <feats_J.parquet> <label> <group_col> <feature_cols,...> <out.json>
Fixed policies (ordering = suspicion, audit first; every draw of randomness is fixed so the learner's warm start reproduces blend50 exactly per seed):
  agree_thr: agreement ascending, n_distinct descending, random tie-break.   judge_thr: judge-accepted groups by s_maj ascending, judge-rejected groups last in random order.
  blend50: score = 0.5 * rank(agree_thr ordering) + 0.5 * rank(judge_thr ordering) (rank = (n - position) / n, higher = audit earlier), random tie-break.
best fixed rule = highest full-pool mean hits at 25% budget over the three fixed rules (8 seeds). Learner: greedy_warm refits balanced logistic on audited rows every
R = max(budget // 8, 40) labels and ranks by it; greedy_blend averages the warm rank and the refit rank (as L2a). Gates (exit 2/3): greedy_warm_norefit == blend50 per seed
on the full pool (8 seeds) and on the first 20 resamples."""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
MIN, SEEDS = 3, 8
path, LABEL, G, cheap = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","); out_path = sys.argv[5] if len(sys.argv) > 5 else None
df = pd.read_parquet(path); cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
if os.environ.get("POOL", "all") == "held":  # same held-out-cells split as sim_e21.py / sim_warm_boot.py
    rng0 = np.random.default_rng(0); groups = df[G].unique(); rng0.shuffle(groups); fit_g = set(groups[: len(groups) // 4]); sim0 = df[~df[G].isin(fit_g)].reset_index(drop=True)
else: sim0 = df

def set_pool(d):
    global sim, Xs, y, agree, ndist, smaj, acc
    sim = d.reset_index(drop=True); Xs = sim[cheap].fillna(0).values; y = sim[LABEL].values.astype(int)
    agree, ndist, smaj, acc = sim.agreement.values, sim.n_distinct.values, sim.s_maj.values, sim.accept.values.astype(bool)
set_pool(sim0)
print(f"pool {len(sim)} groups, label {LABEL}, base {y.mean():.4f}, hits {int(y.sum())}, accepted {int(acc.sum())}, features {cheap}", flush=True)

def rank01(x): return np.argsort(np.argsort(x)) / len(x)

def rs(o):  # ordering (best first) -> score in (0, 1], top = 1
    sc = np.empty(len(o)); sc[o] = (len(o) - np.arange(len(o))) / len(o); return sc

def orders(rng):
    n = len(y); u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)  # fixed draw order: the learner and the fixed rules see the same tie-breaks
    o_agree = np.lexsort((u1, -ndist, agree)); o_judge = np.lexsort((u2, np.where(acc, smaj, np.inf)))
    blend = 0.5 * rs(o_agree) + 0.5 * rs(o_judge); o_blend = np.lexsort((u3, -blend))
    return {"agree_thr": o_agree, "judge_thr": o_judge, "blend50": o_blend}

def run(policy, budget, seed):
    rng = np.random.default_rng(seed)
    if policy == "uniform": order = rng.permutation(len(y))
    elif policy == "oracle": order = np.argsort(-(y + 1e-6 * rng.random(len(y))))
    else: order = orders(rng)[policy]
    return int(y[order[:budget]].sum())

def run_greedy_warm(budget, seed, R=None, blend=False):
    rng = np.random.default_rng(seed); R = R or max(budget // 8, 40); seen = np.zeros(len(y), bool)
    score = rs(orders(rng)["blend50"]); warm_score = score.copy(); hits = spent = 0; next_fit = R
    while spent < budget:
        mu = np.where(~seen)[0]; c = mu[np.argsort(-score[mu])][:1]; seen[c] = True; hits += int(y[c].sum()); spent += len(c)
        if spent >= next_fit and 0 < y[seen].sum() < seen.sum():
            score = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")).fit(Xs[seen], y[seen]).predict_proba(Xs)[:, 1]
            if blend: score = 0.5 * rank01(warm_score) + 0.5 * rank01(score)
            next_fit += R
    return hits

POL = {p: (lambda b, s, p=p: run(p, b, s)) for p in ("uniform", "agree_thr", "judge_thr", "blend50", "oracle")}
POL["greedy_warm_norefit"] = lambda b, s: run_greedy_warm(b, s, R=10**9)  # sanity: refit never fires, must equal blend50 per seed
POL["greedy_warm"] = lambda b, s: run_greedy_warm(b, s); POL["greedy_blend"] = lambda b, s: run_greedy_warm(b, s, blend=True)
ALL = ("uniform", "agree_thr", "judge_thr", "blend50", "greedy_warm", "greedy_blend", "oracle"); FIXED = ("agree_thr", "judge_thr", "blend50")

B, NS = int(os.environ["BOOT_B"]), int(os.environ.get("BOOT_SEEDS", 4)); FRAC = float(os.environ.get("BOOT_FRAC", 0.25)); brng = np.random.default_rng(12345); rows = []
# full pool, unresampled, SEEDS (=8) seeds: budget at FRAC for every policy, and the 25% budget for the three fixed rules (best fixed rule selection)
budget0 = int(FRAC * len(sim0)); fp = {"budget": budget0, "seeds": SEEDS, "n_pool": int(len(sim0)), "n_err": int(y.sum())}
for pol in ALL: fp[pol] = [POL[pol](budget0, s) for s in range(SEEDS)]
fp["greedy_warm_norefit"] = [POL["greedy_warm_norefit"](budget0, s) for s in range(SEEDS)]
b25 = int(0.25 * len(sim0)); fp["budget_25"] = b25; fp["fixed_mean_at_25"] = {p: float(np.mean([POL[p](b25, s) for s in range(SEEDS)])) for p in FIXED}
fp["best_fixed"] = max(FIXED, key=lambda p: fp["fixed_mean_at_25"][p])  # first listed wins ties
fp["fixed_mean_at_frac"] = {p: float(np.mean(fp[p])) for p in FIXED}
fp["ratio_greedy_warm_over_best_fixed"] = float(np.mean(fp["greedy_warm"]) / np.mean(fp[fp["best_fixed"]])); fp["ratio_greedy_blend_over_best_fixed"] = float(np.mean(fp["greedy_blend"]) / np.mean(fp[fp["best_fixed"]]))
san = {"gate_norefit_equals_blend50_fullpool": bool(fp["greedy_warm_norefit"] == fp["blend50"]), "gate_norefit_equals_blend50_first20": None}
print("full pool", json.dumps(fp), flush=True)
if not san["gate_norefit_equals_blend50_fullpool"]: print("GATE FAILED: greedy_warm_norefit != blend50 on the full pool", flush=True); sys.exit(2)
for i in range(B):
    set_pool(sim0.iloc[brng.integers(0, len(sim0), len(sim0))]); budget = int(FRAC * len(sim)); r = {"boot": i, "n_err": int(y.sum()), "budget": budget}
    for pol in ALL: r[pol] = [POL[pol](budget, s) for s in range(NS)]
    if i < 20:
        r["greedy_warm_norefit"] = [POL["greedy_warm_norefit"](budget, s) for s in range(NS)]
        if r["greedy_warm_norefit"] != r["blend50"]: print("GATE FAILED: norefit != blend50 at resample", i, r, flush=True); sys.exit(3)
    rows.append(r); print(i, json.dumps(r), flush=True)
san["gate_norefit_equals_blend50_first20"] = True
if out_path: json.dump({"n_pool": len(sim0), "pool_mode": os.environ.get("POOL", "all"), "label": LABEL, "features": cheap, "budget_frac": FRAC, "seeds_per_resample": NS, "n_boot": B,
                        "sanity": san, "full_pool": fp, "resamples": rows}, open(out_path, "w"), indent=1)
