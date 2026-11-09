"""L6a-cell-rate-usability, replay half. Copy of sim_warm_boot_j2.py (L5c) with ONE added arm, cell_rerank, and an ACCEPT_COL env var. PRE-REGISTERED here, before any real label was read for this experiment.
NEW arm cell_rerank (fixed order plus online): start from the agree_acc order and audit greedily. The proxy-confident half C = accepted groups with s_maj >= median of s_maj over the accepted groups of the (resampled) pool (label-free). The agree_acc order is read as a
sequence of slots; a slot that agree_acc fills with a group of C is filled instead with the unaudited group of C that has the highest count of audited errors in its own cell100 (ties by agree_acc order); a slot filled with a group outside C is filled with the next unaudited group outside C in
agree_acc order, exactly as agree_acc does. So the share of the budget spent inside C equals agree_acc's by construction and only the choice WITHIN C changes. Error count of a cell = number of DISTINCT audited pids with label 1 in that cell (audit-time labels only, no gold before audit),
excluding the candidate's own pid, so duplicate copies of one group in a bootstrap resample cannot leak a label to each other (on the unresampled pool pids are unique and this is the plain count). cell_rerank is NOT a candidate for the best fixed rule (BEST is selected over the five fixed rules exactly as in j2, so every j2 arm reproduces j2).
Usage: ACCEPT_COL=accept BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 sim_warm_boot_j3.py <feats_J.parquet> <label> <group_col> <feature_cols,...> <out.json>; the file must hold pid, the group column, agreement, n_distinct, s_maj, the accept column (ACCEPT_COL, default accept) and the label column.
REPLAY CRITERION R_p (per pool, verdict interval = percentile): ratio = mean over seeds of cell_rerank / mean over seeds of agree_acc per resample (as l5c_finalize.py), 95% percentile CI over the 200 resamples at BOOT_FRAC = 0.25 (500 audits); R_p = CI lower end > 1.0. The basic CI is reported beside it, never used. Context arms reported: greedy_warm, greedy_blend,
best fixed, oracle headroom H = (oracle - best fixed) / oracle. Pool-level and experiment-level bins (PASS / KILL / INCONCLUSIVE) are registered in the docstring of exp/l6a_cell_rate.py and formed in exp/l6a_finalize.py. B = 200 puts about 0.01 of Monte Carlo noise on the percentile CI ends.
Added gates (exit 6 / 7): cell_rerank with the audit-time labels forced to zero (no audited error ever seen) equals agree_acc per seed on the full pool (8 seeds, exit 6) and on the first 20 resamples (exit 7); all j2 gates unchanged (exit 2/3).
---- j2 text follows ----
L5c / L5a corrected-baseline pool bootstrap. Copy of sim_warm_boot_j.py (L3b) with two extra label-free fixed rules and the learner warm-started from the BEST fixed rule instead of always blend50.
Allocator label J (column given as argv[2], here `judge_error` = proxy accepts the majority completion AND gold wrong); pool = ALL groups of feats_J.parquet (env POOL=held restores the E21 held-out-cells quarter split as a robustness read).
Usage: BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 sim_warm_boot_j2.py <feats_J.parquet> <label> <group_col> <feature_cols,...> <out.json>
Fixed policies (ordering = suspicion, audit first; every draw of randomness is fixed so the learner's warm start reproduces the best fixed rule exactly per seed; u1..u3 are drawn first so agree_thr, judge_thr and
blend50 are identical to sim_warm_boot_j.py for the same seed, the new draws u4, u5 come after):
  agree_thr: agreement ascending, n_distinct descending, random tie-break.   judge_thr: accepted groups by s_maj ascending, rejected groups last in random order.
  blend50: 0.5 * rank(agree_thr ordering) + 0.5 * rank(judge_thr ordering) (rank = (n - position) / n, higher = audit earlier), random tie-break.
  NEW agree_acc: accepted groups by agreement ascending, n_distinct descending (random tie-break), rejected groups last in random order.
  NEW blend50_acc: 0.5 * rank(agree_acc ordering) + 0.5 * rank(judge_thr ordering), random tie-break.
best fixed rule = highest full-pool mean hits at 25% budget over the FIVE fixed rules (8 seeds, first listed wins ties), selected once on the unresampled pool and used for every resample.
Learner: greedy_warm is warm-started from the best fixed rule, refits a balanced logistic on audited rows every R = max(budget // 8, 40) labels and ranks by it; greedy_blend averages the warm rank and the refit rank (as L2a).
Gates (exit 2/3): greedy_warm_norefit == best fixed per seed on the full pool (8 seeds) and on the first 20 resamples. Resample generator default_rng(12345)."""
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
    global sim, Xs, y, agree, ndist, smaj, acc, conf, cellc, pidc, ncell, npid
    sim = d.reset_index(drop=True); Xs = sim[cheap].fillna(0).values; y = sim[LABEL].values.astype(int)
    agree, ndist, smaj, acc = sim.agreement.values, sim.n_distinct.values, sim.s_maj.values, sim[os.environ.get("ACCEPT_COL", "accept")].values.astype(bool)
    conf = acc & (smaj >= np.median(smaj[acc])); cellc, pidc = pd.factorize(sim[G])[0], pd.factorize(sim.pid)[0]; ncell, npid = cellc.max() + 1, pidc.max() + 1
set_pool(sim0)
print(f"pool {len(sim)} groups, label {LABEL}, base {y.mean():.4f}, hits {int(y.sum())}, accepted {int(acc.sum())}, features {cheap}", flush=True)

def rank01(x): return np.argsort(np.argsort(x)) / len(x)

def rs(o):  # ordering (best first) -> score in (0, 1], top = 1
    sc = np.empty(len(o)); sc[o] = (len(o) - np.arange(len(o))) / len(o); return sc

def orders(rng):
    n = len(y); u1, u2, u3, u4, u5 = (rng.random(n) for _ in range(5))  # fixed draw order: the learner and the fixed rules see the same tie-breaks
    o_agree = np.lexsort((u1, -ndist, agree)); o_judge = np.lexsort((u2, np.where(acc, smaj, np.inf)))
    blend = 0.5 * rs(o_agree) + 0.5 * rs(o_judge); o_blend = np.lexsort((u3, -blend))
    o_agree_acc = np.lexsort((u4, np.where(acc, -ndist, 0), np.where(acc, agree, 0), (~acc).astype(int)))  # primary key last: accepted first, rejected random
    blend_acc = 0.5 * rs(o_agree_acc) + 0.5 * rs(o_judge); o_blend_acc = np.lexsort((u5, -blend_acc))
    return {"agree_thr": o_agree, "judge_thr": o_judge, "blend50": o_blend, "agree_acc": o_agree_acc, "blend50_acc": o_blend_acc}

def run(policy, budget, seed):
    rng = np.random.default_rng(seed)
    if policy == "uniform": order = rng.permutation(len(y))
    elif policy == "oracle": order = np.argsort(-(y + 1e-6 * rng.random(len(y))))
    else: order = orders(rng)[policy]
    return int(y[order[:budget]].sum())

def run_cell_rerank(budget, seed, y_aud=None):
    """cell_rerank (see docstring). y_aud = labels the audit reveals for the cell counts (default y); the gate passes zeros to check the arm collapses to agree_acc. Hits are always scored with the real y."""
    ya = y if y_aud is None else y_aud; o = orders(np.random.default_rng(seed))["agree_acc"]; isc = conf[o]; cl = o[isc]; oth = o[~isc]
    cell_err = np.zeros(ncell); pid_err = np.zeros(npid, bool); unseen = np.ones(len(cl), bool); oi = hits = 0
    for t in range(budget):
        if isc[t]:
            j = np.flatnonzero(unseen); cand = cl[j]; k = np.argmax(cell_err[cellc[cand]]); g = cand[k]; unseen[j[k]] = False  # argmax = first max = agree_acc order
        else: g = oth[oi]; oi += 1
        hits += int(y[g])
        if ya[g] and not pid_err[pidc[g]]: pid_err[pidc[g]] = True; cell_err[cellc[g]] += 1
    return hits

BEST = None  # set from the full-pool 25% means before any learner arm runs

def run_greedy_warm(budget, seed, R=None, blend=False):
    rng = np.random.default_rng(seed); R = R or max(budget // 8, 40); seen = np.zeros(len(y), bool)
    score = rs(orders(rng)[BEST]); warm_score = score.copy(); hits = spent = 0; next_fit = R
    while spent < budget:
        mu = np.where(~seen)[0]; c = mu[np.argsort(-score[mu])][:1]; seen[c] = True; hits += int(y[c].sum()); spent += len(c)
        if spent >= next_fit and 0 < y[seen].sum() < seen.sum():
            score = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")).fit(Xs[seen], y[seen]).predict_proba(Xs)[:, 1]
            if blend: score = 0.5 * rank01(warm_score) + 0.5 * rank01(score)
            next_fit += R
    return hits

FIXED = ("agree_thr", "judge_thr", "blend50", "agree_acc", "blend50_acc")
ALL = ("uniform",) + FIXED + ("cell_rerank", "greedy_warm", "greedy_blend", "oracle")
POL = {p: (lambda b, s, p=p: run(p, b, s)) for p in ("uniform", "oracle") + FIXED}
POL["cell_rerank"] = lambda b, s: run_cell_rerank(b, s); POL["cell_rerank_noerr"] = lambda b, s: run_cell_rerank(b, s, y_aud=np.zeros(len(y), int))
POL["greedy_warm_norefit"] = lambda b, s: run_greedy_warm(b, s, R=10**9)  # sanity: refit never fires, must equal the best fixed rule per seed
POL["greedy_warm"] = lambda b, s: run_greedy_warm(b, s); POL["greedy_blend"] = lambda b, s: run_greedy_warm(b, s, blend=True)

B, NS = int(os.environ["BOOT_B"]), int(os.environ.get("BOOT_SEEDS", 4)); FRAC = float(os.environ.get("BOOT_FRAC", 0.25)); brng = np.random.default_rng(12345); rows = []
# full pool, unresampled, SEEDS (=8) seeds: the 25% budget for the fixed rules (best fixed rule selection), then the FRAC budget for every policy
budget0 = int(FRAC * len(sim0)); fp = {"budget": budget0, "seeds": SEEDS, "n_pool": int(len(sim0)), "n_err": int(y.sum())}
b25 = int(0.25 * len(sim0)); fp["budget_25"] = b25; fp["fixed_mean_at_25"] = {p: float(np.mean([POL[p](b25, s) for s in range(SEEDS)])) for p in FIXED}
BEST = max(FIXED, key=lambda p: fp["fixed_mean_at_25"][p]); fp["best_fixed"] = BEST  # first listed wins ties
for pol in ALL: fp[pol] = [POL[pol](budget0, s) for s in range(SEEDS)]
fp["greedy_warm_norefit"] = [POL["greedy_warm_norefit"](budget0, s) for s in range(SEEDS)]
fp["fixed_mean_at_frac"] = {p: float(np.mean(fp[p])) for p in FIXED}
fp["ratio_greedy_warm_over_best_fixed"] = float(np.mean(fp["greedy_warm"]) / np.mean(fp[BEST])); fp["ratio_greedy_blend_over_best_fixed"] = float(np.mean(fp["greedy_blend"]) / np.mean(fp[BEST]))
fp["cell_rerank_noerr"] = [POL["cell_rerank_noerr"](budget0, s) for s in range(SEEDS)]
san = {"gate_norefit_equals_best_fixed_fullpool": bool(fp["greedy_warm_norefit"] == fp[BEST]), "gate_norefit_equals_best_fixed_first20": None, "gate_cellrerank_noerr_equals_agree_acc_fullpool": bool(fp["cell_rerank_noerr"] == fp["agree_acc"]), "gate_cellrerank_noerr_equals_agree_acc_first20": None, "best_fixed": BEST}
print("full pool", json.dumps(fp), flush=True)
if not san["gate_norefit_equals_best_fixed_fullpool"]: print("GATE FAILED: greedy_warm_norefit != best fixed on the full pool", flush=True); sys.exit(2)
if not san["gate_cellrerank_noerr_equals_agree_acc_fullpool"]: print("GATE FAILED: cell_rerank with no audited error != agree_acc on the full pool", fp["cell_rerank_noerr"], fp["agree_acc"], flush=True); sys.exit(6)
for i in range(B):
    set_pool(sim0.iloc[brng.integers(0, len(sim0), len(sim0))]); budget = int(FRAC * len(sim)); r = {"boot": i, "n_err": int(y.sum()), "budget": budget}
    for pol in ALL: r[pol] = [POL[pol](budget, s) for s in range(NS)]
    if i < 20:
        r["greedy_warm_norefit"] = [POL["greedy_warm_norefit"](budget, s) for s in range(NS)]
        if r["greedy_warm_norefit"] != r[BEST]: print("GATE FAILED: norefit != best fixed at resample", i, r, flush=True); sys.exit(3)
        r["cell_rerank_noerr"] = [POL["cell_rerank_noerr"](budget, s) for s in range(NS)]
        if r["cell_rerank_noerr"] != r["agree_acc"]: print("GATE FAILED: cell_rerank with no audited error != agree_acc at resample", i, r, flush=True); sys.exit(7)
    rows.append(r); print(i, json.dumps(r), flush=True)
san["gate_norefit_equals_best_fixed_first20"] = True; san["gate_cellrerank_noerr_equals_agree_acc_first20"] = True
if out_path: json.dump({"n_pool": len(sim0), "pool_mode": os.environ.get("POOL", "all"), "accept_col": os.environ.get("ACCEPT_COL", "accept"), "label": LABEL, "features": cheap, "budget_frac": FRAC, "seeds_per_resample": NS, "n_boot": B,
                        "sanity": san, "full_pool": fp, "resamples": rows}, open(out_path, "w"), indent=1)
