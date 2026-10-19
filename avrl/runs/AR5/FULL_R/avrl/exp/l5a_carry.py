"""L5a: cross-step label carry-over replay (CPU only). run(), charged_order(), run_online(), ridge_logit() and run_offset() are copied verbatim from exp/l4a_sim_alloc.py
(that file executes at import, so the functions are copied, never imported); the module globals they read (Xs, y, agree, ndist, Zoff, Zoth) are set by bind().
Usage: l5a_carry.py <feats.parquet> <label> <cheap_col1,...> <budget B> <order_seed> <out.json>
       l5a_carry.py --gate <feats.parquet> <label> <cheap_col1,...> <budget B> <out.json>     (regression gate vs runs/L4a, see l5a_gate_check.py)
Design: permute the 3,000 prompts with default_rng(10000 + order_seed) and cut 12 consecutive disjoint batches of m = 250 (each prompt used once).
Per batch the budget is B (62 = 25%, 125 = 50% of the batch). Policy seed = order seed for every arm and batch (each arm builds default_rng(seed) itself, as in L4a).
Arms (fixed in advance, none chosen on held-out data):
  thr_agreement, uniform                  free, no labels
  restart_offset  = offset_n0_R5_lam10 from scratch every batch (L4a code, batch-standardized Zs)
  restart_plain   = greedy_n3_R1 from scratch every batch (L4a code)
  carry_plain     = greedy_n3_R1 logic with every earlier label kept (rule = StandardScaler + balanced logistic on the labels in hand, refit before every pick), rank only the current batch;
                    batch 1 equals restart_plain exactly
  carry_offset_lam{10,1}  = offset learner (n0 = 0, R = 5) whose ridge fit uses all carried labels; w refit at the start of every batch and after every 5 new labels
  carry_window{W}_lam{10,1}, W in 125, 250, 500: same, fitted on the last W labels only (drift proxy); both lam are run, the summary picks the better lam on order seeds 0..3
  calibrate_once  = recal_charged_50 in batch 1 (50 random labels charged inside the budget), the fitted rule reused free in batches 2..12 (top B by score, no further labels)
  static_out      = balanced logistic rule fit with free labels on the 3,000 - 250 prompts outside the batch (optimistic ceiling, never a policy)
Standardization: restart arms standardize on the current batch exactly as L4a (Zs from the 250 batch prompts). carry_offset / carry_window use ONE global label-free z-score over all 3,000 prompts
(Zg) so carried rows and current rows live on one scale (offset = -2 * global z agreement). carry_plain keeps L4a's rule, whose StandardScaler is fit on the labels in hand (the carried set); that
composition is invariant to any extra affine transform, so a global transform would change nothing for it."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

K, SEEDS, M, NB, OFFSET_C = 1, 8, 250, 12, 2.0
WINDOWS, LAMS, N_CAL = (125, 250, 500), (10, 1), 50
args = sys.argv[1:]; GATE = bool(args) and args[0] == "--gate"; args = args[1:] if GATE else args
path, LABEL, cheap = args[0], args[1], args[2].split(","); BUDGET = int(args[3])
ORDER = None if GATE else int(args[4]); out_path = args[4] if GATE else args[5]
df = pd.read_parquet(path); Xall = df[cheap].fillna(0).values; yall = df[LABEL].values.astype(int); N = len(df)
AG_ALL, ND_ALL = df["agreement"].values, df["n_distinct"].values
IA = cheap.index("agreement")
_mu_g, _sd_g = Xall.mean(0), Xall.std(0); Zg = (Xall - _mu_g) / np.where(_sd_g > 0, _sd_g, 1.0)  # global label-free transform (carry arms)
Zoff_g = -OFFSET_C * Zg[:, IA]; Zoth_g = np.delete(Zg, IA, axis=1)

def bind(idx):
    """Point the copied L4a functions at the prompt set idx, standardized on that set exactly as L4a does (no labels used)."""
    global Xs, y, agree, ndist, Zoff, Zoth
    Xs = Xall[idx]; y = yall[idx]; agree = AG_ALL[idx]; ndist = ND_ALL[idx]
    mu, sd = Xs.mean(0), Xs.std(0); Zs = (Xs - mu) / np.where(sd > 0, sd, 1.0); Zoff = -OFFSET_C * Zs[:, IA]; Zoth = np.delete(Zs, IA, axis=1)

# ---------------- copied verbatim from exp/l4a_sim_alloc.py (only the branches L5a uses) ----------------
def rule(Xa, ya, max_iter=1000): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, class_weight="balanced")).fit(Xa, ya)

def charged_order(cal, fallback):
    """cal labels are verified first (charged, counted as hits); a rule fit on them ranks the rest. fallback must start with cal."""
    if y[cal].sum() == 0 or y[cal].sum() == len(cal): return fallback
    sc = rule(Xs[cal], y[cal]).predict_proba(Xs)[:, 1]; sc[cal] = -1
    return np.concatenate([cal, np.argsort(-sc)[: len(y) - len(cal)]])

def run(policy, budget, seed):
    rng = np.random.default_rng(seed)
    if policy == "uniform": order = rng.permutation(len(y))
    elif policy.startswith("recal_charged_"):  # n_cal random verified docs
        n_cal = int(policy.rsplit("_", 1)[1]); perm = rng.permutation(len(y)); order = charged_order(perm[:n_cal], perm)
    elif policy == "thr_agreement": order = np.lexsort((rng.random(len(y)), agree))  # low agreement first, random ties, no labels
    else: raise ValueError(policy)
    return int(y[order[:budget]].sum())

def run_online(budget, seed, n0=None, R=None, seeded=False):
    """greedy_online: inspect the top-scored unseen docs anywhere; score is random (or lowest-agreement first when seeded) until the first fit after n0 labels,
    then a classifier is refit on all labels collected every R labels. n0 = R = max(budget // 8, 40) is the L1b reference."""
    rng = np.random.default_rng(seed); R = R or max(budget // 8, 40); n0 = n0 or R
    seen = np.zeros(len(y), bool); hits = spent = 0; next_fit = n0
    if seeded:
        ag = np.lexsort((rng.random(len(y)), agree)); score = np.empty(len(y)); score[ag] = -np.arange(len(y), dtype=float)
    else: score = rng.random(len(y)) * 1e-3
    while spent < budget:
        take = min(K, budget - spent)
        mu = np.where(~seen)[0]; c = mu[np.argsort(-score[mu])][:take]; seen[c] = True; hits += int(y[c].sum()); spent += len(c)
        if spent >= next_fit and 0 < y[seen].sum() < seen.sum():
            score = rule(Xs[seen], y[seen], 500).predict_proba(Xs)[:, 1]; next_fit += R
    return hits

def ridge_logit(Z, yy, off, lam, iters=100):
    """Ridge logistic regression with a fixed offset, by Newton/IRLS with backtracking: eta = off + b + Z w, loss = sum(log(1+e^eta) - y eta) + lam/2 |w|^2,
    intercept b unpenalized, no class weights. lam = inf returns w = 0. Returns w (len Z.shape[1])."""
    p = Z.shape[1]
    if np.isinf(lam): return np.zeros(p)
    A = np.hstack([np.ones((len(yy), 1)), Z]); pen = np.r_[0.0, np.full(p, float(lam))]; beta = np.zeros(p + 1)
    nll = lambda b: float(np.sum(np.logaddexp(0.0, off + A @ b) - yy * (off + A @ b)) + 0.5 * np.sum(pen * b * b))
    f = nll(beta)
    for _ in range(iters):
        pr = expit(off + A @ beta); g = A.T @ (pr - yy) + pen * beta
        H = (A * (pr * (1 - pr))[:, None]).T @ A + np.diag(pen) + 1e-10 * np.eye(p + 1)
        step = np.linalg.solve(H, g); t = 1.0
        while t > 1e-10:
            nb = beta - t * step; fn = nll(nb)
            if fn <= f: break
            t /= 2
        else: break
        done = f - fn < 1e-10 * (1 + abs(f)); beta, f = nb, fn
        if done: break
    return beta[1:]

def run_offset(budget, seed, n0, R, lam):
    """Warm-start learner. score = -OFFSET_C * z_agreement + w.z_other; w = 0 at the start, so the first labels are the lowest-agreement prompts with random ties
    (identical to thr_agreement's prefix). The first refit is attempted after n0 labels (after R labels when n0 = 0), then every R labels: w is refit by ridge-logistic IRLS
    (offset = -OFFSET_C * z_agreement, intercept unpenalized) on all labels so far. A refit is skipped while the labels are all one class (retried at every label until it can run)."""
    rng = np.random.default_rng(seed); tb = rng.random(len(y))  # same first draw as thr_agreement, so ties break identically
    score = Zoff.copy(); seen = np.zeros(len(y), bool); hits = spent = 0; next_fit = n0 or R
    while spent < budget:
        mu = np.where(~seen)[0]; c = mu[np.lexsort((tb[mu], -score[mu]))[0]]
        seen[c] = True; hits += int(y[c]); spent += 1
        if spent >= next_fit and 0 < y[seen].sum() < seen.sum():
            score = Zoff + Zoth @ ridge_logit(Zoth[seen], y[seen].astype(float), Zoff[seen], lam); next_fit = spent + R
    return hits
# ---------------- end of verbatim copy ----------------

def _diag(L, ya, za):
    """selection-bias diagnostics of the labels in hand: count, error rate, mean global z-agreement (population mean is 0)."""
    L = np.asarray(L); return {"n": int(len(L)), "err_rate": float(ya[L].mean()) if len(L) else None, "mean_z_agree": float(za[L].mean()) if len(L) else None}

def carry_plain(Xa, ya, batches, budget, seed, za):
    """greedy_n3_R1 with every earlier label kept. Labels in hand L (global indices, label order). Before every pick, if >= 3 labels are in hand with both classes and they changed since the
    last fit, the L4a rule is refit on all of L and scores the current batch; otherwise the random-start score (rng.random * 1e-3, as run_online) is used. Equals run_online on batch 1."""
    L, hits_b, diag = [], [], []
    for b in batches:
        rng = np.random.default_rng(seed); score = rng.random(len(b)) * 1e-3; seen = np.zeros(len(b), bool); hits = spent = 0; fitted_n = -1
        while spent < budget:
            n = len(L)
            if n != fitted_n and n >= 3 and 0 < ya[L].sum() < n:
                score = rule(Xa[L], ya[L], 500).predict_proba(Xa[b])[:, 1]; fitted_n = n
            mu = np.where(~seen)[0]; c = mu[np.argsort(-score[mu])][:1]; seen[c] = True; hits += int(ya[b[c]].sum()); spent += 1; L.append(int(b[c[0]]))
        hits_b.append(hits); diag.append(_diag(L, ya, za))
    return hits_b, diag

def carry_offset(Zo, Zt, ya, batches, budget, seed, lam, za, R=5, window=None):
    """Offset learner (n0 = 0, R = 5) whose ridge fit uses all carried labels (window = None) or the last `window` labels. Zo, Zt: offset and 9-feature matrices over ALL rows (global z).
    w = 0 at the start of batch 1 (so batch 1 follows thr_agreement until the first refit); at the start of every later batch w is refit on the labels in hand, then after every R new labels.
    A refit is skipped while the fitted label set is one class (retried at every label). tb = default_rng(seed).random(len(batch)) so ties break as in thr_agreement."""
    L, hits_b, diag = [], [], []; w = np.zeros(Zt.shape[1]); inf = np.isinf(lam)
    def refit():
        Lw = np.asarray(L[-window:] if window else L)
        if 0 < ya[Lw].sum() < len(Lw): return ridge_logit(Zt[Lw], ya[Lw].astype(float), Zo[Lw], lam), True
        return None, False
    for bi, b in enumerate(batches):
        tb = np.random.default_rng(seed).random(len(b)); seen = np.zeros(len(b), bool); hits = spent = 0; since = 0
        if bi > 0:
            wn, ok = refit()
            if ok: w = wn
        base, Zb = Zo[b], Zt[b]
        while spent < budget:
            score = base + Zb @ w; mu = np.where(~seen)[0]; c = mu[np.lexsort((tb[mu], -score[mu]))[0]]
            seen[c] = True; hits += int(ya[b[c]]); spent += 1; L.append(int(b[c])); since += 1
            if since >= R:
                wn, ok = refit()
                if ok: w = wn; since = 0
        hits_b.append(hits); diag.append(_diag(L, ya, za))
    return hits_b, diag

if GATE:
    # regression gate: the L4a prompt set for (m = 250, subset seed 0); thr_agreement, uniform, greedy_n3_R1, offset_n0_R5_lam10 through the copied functions, per policy seed 0..7
    idx = np.sort(np.random.default_rng(0).choice(len(df), 250, replace=False)); bind(idx)
    pol = {"thr_agreement": [run("thr_agreement", BUDGET, s) for s in range(SEEDS)], "uniform": [run("uniform", BUDGET, s) for s in range(SEEDS)],
           "greedy_n3_R1": [run_online(BUDGET, s, n0=3, R=1, seeded=False) for s in range(SEEDS)], "offset_n0_R5_lam10": [run_offset(BUDGET, s, 0, 5, 10) for s in range(SEEDS)]}
    one = [np.arange(len(idx))]; za1 = (Xs[:, IA] - Xs[:, IA].mean()) / Xs[:, IA].std()  # diagnostics only
    sc = {"carry_plain_1batch_vs_run_online": ([carry_plain(Xs, y, one, BUDGET, s, za1)[0][0] for s in range(SEEDS)], pol["greedy_n3_R1"]),
          "carry_offset_lam10_1batch_vs_run_offset": ([carry_offset(Zoff, Zoth, y, one, BUDGET, s, 10, za1)[0][0] for s in range(SEEDS)], pol["offset_n0_R5_lam10"]),
          "carry_window125_1batch_vs_run_offset": ([carry_offset(Zoff, Zoth, y, one, BUDGET, s, 10, za1, window=125)[0][0] for s in range(SEEDS)], pol["offset_n0_R5_lam10"]),
          "carry_offset_lam_inf_1batch_vs_thr": ([carry_offset(Zoff, Zoth, y, one, BUDGET, s, np.inf, za1)[0][0] for s in range(SEEDS)], pol["thr_agreement"])}
    # multi-batch path: carry_offset with lam = inf (w = 0) over the real 12-batch layout of order seed 0 must equal thr_agreement on every batch
    perm0 = np.random.default_rng(10000).permutation(N); bat0 = [perm0[i * M:(i + 1) * M] for i in range(NB)]
    ci = carry_offset(Zoff_g, Zoth_g, yall, bat0, BUDGET, 0, np.inf, Zg[:, IA])[0]; ti = []
    for b in bat0: bind(b); ti.append(run("thr_agreement", BUDGET, 0))
    sc["carry_offset_lam_inf_12batch_vs_thr"] = (ci, ti)
    selfcheck = {k: {"carry": [int(x) for x in a], "ref": [int(x) for x in r], "max_abs_diff": int(np.max(np.abs(np.array(a) - np.array(r))))} for k, (a, r) in sc.items()}
    res = {"feats": path, "label": LABEL, "budget": BUDGET, "m": 250, "subset_seed": 0, "policies": {k: [int(v) for v in vs] for k, vs in pol.items()}, "selfcheck": selfcheck}
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"gate {os.path.basename(path)} B {BUDGET}: " + " ".join(f"{k}={np.mean(v):.2f}" for k, v in pol.items()) + " selfcheck max diffs " + str({k: v["max_abs_diff"] for k, v in selfcheck.items()}), flush=True)
    sys.exit(0)

perm = np.random.default_rng(10000 + ORDER).permutation(N); batches = [perm[i * M:(i + 1) * M] for i in range(NB)]
assert len(perm) == NB * M and len(np.unique(np.concatenate(batches))) == N
arms = {k: [] for k in ("thr_agreement", "uniform", "restart_offset", "restart_plain", "calibrate_once", "static_out")}; clf_cal = None; base_errors = []
for bi, b in enumerate(batches):
    bind(b); base_errors.append(int(y.sum()))
    arms["thr_agreement"].append(run("thr_agreement", BUDGET, ORDER)); arms["uniform"].append(run("uniform", BUDGET, ORDER))
    arms["restart_offset"].append(run_offset(BUDGET, ORDER, 0, 5, 10)); arms["restart_plain"].append(run_online(BUDGET, ORDER, n0=3, R=1, seeded=False))
    if bi == 0:  # calibrate_once: batch 1 is exactly recal_charged_50 (cal labels charged inside the budget); keep the fitted rule for batches 2..12
        arms["calibrate_once"].append(run(f"recal_charged_{N_CAL}", BUDGET, ORDER)); cal = np.random.default_rng(ORDER).permutation(len(y))[:N_CAL]
        clf_cal = rule(Xs[cal], y[cal]) if 0 < y[cal].sum() < N_CAL else None
    else:  # fitted rule reused free; a one-class calibration set (not seen at 50 random labels) would fall back to a random order
        order = np.argsort(-clf_cal.predict_proba(Xs)[:, 1]) if clf_cal is not None else np.random.default_rng(ORDER).permutation(len(y))
        arms["calibrate_once"].append(int(y[order[:BUDGET]].sum()))
    outside = np.setdiff1d(np.arange(N), b); sc_s = rule(Xall[outside], yall[outside]).predict_proba(Xall[b])[:, 1]; arms["static_out"].append(int(y[np.argsort(-sc_s)[:BUDGET]].sum()))
za = Zg[:, IA]; diag = {}
arms["carry_plain"], diag["carry_plain"] = carry_plain(Xall, yall, batches, BUDGET, ORDER, za)
for lam in LAMS:
    arms[f"carry_offset_lam{lam}"], diag[f"carry_offset_lam{lam}"] = carry_offset(Zoff_g, Zoth_g, yall, batches, BUDGET, ORDER, lam, za)
    for W in WINDOWS: arms[f"carry_window{W}_lam{lam}"], diag[f"carry_window{W}_lam{lam}"] = carry_offset(Zoff_g, Zoth_g, yall, batches, BUDGET, ORDER, lam, za, window=W)
chk = {"batch1_carry_plain_eq_restart_plain": bool(arms["carry_plain"][0] == arms["restart_plain"][0]),
       "batch1_windows_eq_carry_offset": bool(all(arms[f"carry_window{W}_lam{lam}"][0] == arms[f"carry_offset_lam{lam}"][0] for lam in LAMS for W in WINDOWS))}
out = {"feats": path, "label": LABEL, "features": cheap, "budget": BUDGET, "m": M, "n_batches": NB, "order_seed": ORDER, "policy_seed": ORDER, "batch_base_errors": base_errors,
       "arms": {k: [int(v) for v in vs] for k, vs in arms.items()}, "diag": diag, "checks": chk, "offset_c": OFFSET_C, "windows": list(WINDOWS), "lams": list(LAMS)}
json.dump(out, open(out_path, "w"), indent=1)
pool = lambda k: np.sum(arms[k][3:])
print(f"{os.path.basename(path)} B {BUDGET} order {ORDER} checks {chk}; hits batches 4..12 " + " ".join(f"{k}={pool(k)}" for k in arms), flush=True)
