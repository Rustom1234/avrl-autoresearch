"""L7a: persistent learner under a mid-stream proxy change (judge swap), CPU only. Copied from exp/l6a_amort.py (functions copied, never imported; l6a files untouched).
Usage: l7a_drift.py <featsA.parquet> <featsB.parquet|same> <label> <cheap_col1,...> <budget B> <order_seed> <mode> <out.json>
Design (the L5a/L6a order layout): permute the 3,000 prompts with default_rng(10000 + order_seed), 12 consecutive disjoint batches of m = 250, per-step budget B (62 or 125), policy seed = order seed.
Modes:
  drift         : batches 1..6 use table A (labels + judge features of featsA), batches 7..12 use table B (featsB). Both tables are row-aligned (same pid; agreement, n_distinct, mean_tokens, std_tokens,
                  any_truncated, log_qlen identical, asserted below), so row i is the same prompt; the judge swap changes only judge_error and the four judge_* features. Labels carried across the switch keep
                  the label and the z they had when collected. Per-batch label-free z (each batch standardized on its own rows).
  control_exact : every batch from table A, arms in exactly their L6a form (carry_plain on raw features with the StandardScaler fitted on the labels, carry_offset_lam1 on global z). Regression gate vs runs/L6a.
  control_z     : every batch from table A, arms in the drift (per-batch z) form: the no-drift like-for-like reference for the drift arms.
Arms (fixed in advance, none chosen on held-out data): thr_agreement, restart_offset (verbatim L6a); carry_plain (verbatim L6a, unbounded); carry_plain_window_W for W in 125, 250, 500 (rule refit on the last W labels
  in hand); carry_plain_decay (rule refit on all labels in hand with exponential sample weights, 0.5 ** (age / 250) with age = number of labels collected after it, in the logistic step only, StandardScaler unweighted);
  carry_offset_lam1 (verbatim L6a, unbounded); freeze_plain_k3 / freeze_plain_k6 (L6a freeze_plain with k batches: carry_plain hits through batch k, then the rule refit on the k*B labels in hand frozen and reused free in
  batches k+1..12; k <= 6 so the freeze precedes the switch); reset_at_switch (carry_plain with the label store cleared at batch 7: an upper bound that knows the switch; in the control modes it is the same reset without a drift).
The module globals the copied functions read (Xs, y, agree, ndist, Zoff, Zoth, Xall, yall) are set by bind() / below."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

K, M, NB, OFFSET_C, SWITCH = 1, 250, 12, 2.0, 6  # batches with 0-based index < SWITCH use table A
WINDOWS, HALF_LIFE, FREEZE_K = (125, 250, 500), 250, (3, 6)
args = sys.argv[1:]
pathA, pathB, LABEL, cheap = args[0], args[1], args[2], args[3].split(","); BUDGET, ORDER, MODE, out_path = int(args[4]), int(args[5]), args[6], args[7]
assert MODE in ("drift", "control_exact", "control_z") and (pathB != "same") == (MODE == "drift"), (MODE, pathB)
dfA = pd.read_parquet(pathA); N = len(dfA); XA = dfA[cheap].fillna(0).values; yA = dfA[LABEL].values.astype(int)
perm = np.random.default_rng(10000 + ORDER).permutation(N); batches = [perm[i * M:(i + 1) * M] for i in range(NB)]
assert len(perm) == NB * M and len(np.unique(np.concatenate(batches))) == N and BUDGET < M
if MODE == "drift":
    dfB = pd.read_parquet(pathB); XB = dfB[cheap].fillna(0).values; yB = dfB[LABEL].values.astype(int)
    shared = [c for c in ("agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen") if c in cheap]
    assert len(dfB) == N and np.array_equal(dfA["pid"].values, dfB["pid"].values), "tables are not row-aligned"
    assert all(np.array_equal(dfA[c].fillna(0).values, dfB[c].fillna(0).values) for c in shared), "judge-independent columns differ between tables"
    inB = np.zeros(N, bool); inB[np.concatenate(batches[SWITCH:])] = True  # rows whose batch comes after the switch
    Xall = np.where(inB[:, None], XB, XA); yall = np.where(inB, yB, yA)  # the stream the learner sees: row i from table A before the switch, from table B after
else: Xall, yall = XA, yA
AG_ALL, ND_ALL = dfA["agreement"].values, dfA["n_distinct"].values
IA = cheap.index("agreement")
_mu_g, _sd_g = Xall.mean(0), Xall.std(0); Zg = (Xall - _mu_g) / np.where(_sd_g > 0, _sd_g, 1.0)  # global label-free transform (exact form only, as L6a)
Zoff_g = -OFFSET_C * Zg[:, IA]; Zoth_g = np.delete(Zg, IA, axis=1)
def batch_z(X):  # per-batch label-free z: every batch standardized on its own rows; a row keeps the z it got in its own batch
    Z = np.empty(X.shape, float)
    for b in batches: mu, sd = X[b].mean(0), X[b].std(0); Z[b] = (X[b] - mu) / np.where(sd > 0, sd, 1.0)
    return Z
if MODE == "control_exact": FORM, Xa, Zo, Zt, za = "exact", Xall, Zoff_g, Zoth_g, Zg[:, IA]
else: Zb_ = batch_z(Xall); FORM, Xa, Zo, Zt, za = "perbatch_z", Zb_, -OFFSET_C * Zb_[:, IA], np.delete(Zb_, IA, axis=1), Zb_[:, IA]

def bind(idx):
    """Point the copied L4a functions at the prompt set idx, standardized on that set exactly as L4a does (no labels used)."""
    global Xs, y, agree, ndist, Zoff, Zoth
    Xs = Xall[idx]; y = yall[idx]; agree = AG_ALL[idx]; ndist = ND_ALL[idx]
    mu, sd = Xs.mean(0), Xs.std(0); Zs = (Xs - mu) / np.where(sd > 0, sd, 1.0); Zoff = -OFFSET_C * Zs[:, IA]; Zoth = np.delete(Zs, IA, axis=1)

# ---------------- copied from exp/l6a_amort.py (verbatim except: rule() takes optional sample weights, carry_plain() takes win / hl / reset_at which default to the verbatim behaviour, freeze_plain() takes the feature matrix and k) ----------------
def rule(Xa_, ya, max_iter=1000, w=None):
    p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, class_weight="balanced"))
    return p.fit(Xa_, ya) if w is None else p.fit(Xa_, ya, logisticregression__sample_weight=w)

def run(policy, budget, seed):
    rng = np.random.default_rng(seed)
    if policy == "thr_agreement": order = np.lexsort((rng.random(len(y)), agree))  # low agreement first, random ties, no labels
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

def _diag(L, ya, za_):
    """selection-bias diagnostics of the labels in hand: count, error rate, mean z-agreement."""
    L = np.asarray(L); return {"n": int(len(L)), "err_rate": float(ya[L].mean()) if len(L) else None, "mean_z_agree": float(za_[L].mean()) if len(L) else None}

def carry_plain(Xa_, ya, batches_, budget, seed, za_, win=None, hl=None, reset_at=None):
    """greedy_n3_R1 with every earlier label kept (win = None, hl = None, reset_at = None: verbatim L6a). Labels in hand L (global indices, label order). Before every pick, if >= 3 labels are in hand with both classes and they
    changed since the last fit, the L4a rule is refit on all of L and scores the current batch; otherwise the random-start score (rng.random * 1e-3, as run_online) is used. Equals run_online on batch 1.
    New in L7a: win = W refits on the last W labels in hand (both-class test on that window); hl = h refits on all of L with sample weights 0.5 ** (age / h); reset_at = j clears L at the start of batch j."""
    L, hits_b, diag = [], [], []
    for bi, b in enumerate(batches_):
        if reset_at is not None and bi == reset_at: L = []
        rng = np.random.default_rng(seed); score = rng.random(len(b)) * 1e-3; seen = np.zeros(len(b), bool); hits = spent = 0; fitted_n = -1
        while spent < budget:
            n = len(L); Lw = L[-win:] if win else L
            if n != fitted_n and n >= 3 and 0 < ya[Lw].sum() < len(Lw):
                w = None if hl is None else 0.5 ** (np.arange(len(Lw))[::-1] / hl)
                score = rule(Xa_[Lw], ya[Lw], 500, w).predict_proba(Xa_[b])[:, 1]; fitted_n = n
            mu = np.where(~seen)[0]; c = mu[np.argsort(-score[mu])][:1]; seen[c] = True; hits += int(ya[b[c]].sum()); spent += 1; L.append(int(b[c[0]]))
        hits_b.append(hits); diag.append(_diag(L, ya, za_))
    return hits_b, diag, L

def carry_offset(Zo_, Zt_, ya, batches_, budget, seed, lam, za_, R=5, window=None):
    """Offset learner (n0 = 0, R = 5) whose ridge fit uses all carried labels (window = None) or the last `window` labels. Zo_, Zt_: offset and 9-feature matrices over ALL rows.
    w = 0 at the start of batch 1 (so batch 1 follows thr_agreement until the first refit); at the start of every later batch w is refit on the labels in hand, then after every R new labels.
    A refit is skipped while the fitted label set is one class (retried at every label). tb = default_rng(seed).random(len(batch)) so ties break as in thr_agreement."""
    L, hits_b, diag = [], [], []; w = np.zeros(Zt_.shape[1]); inf = np.isinf(lam)
    def refit():
        Lw = np.asarray(L[-window:] if window else L)
        if 0 < ya[Lw].sum() < len(Lw): return ridge_logit(Zt_[Lw], ya[Lw].astype(float), Zo_[Lw], lam), True
        return None, False
    for bi, b in enumerate(batches_):
        tb = np.random.default_rng(seed).random(len(b)); seen = np.zeros(len(b), bool); hits = spent = 0; since = 0
        if bi > 0:
            wn, ok = refit()
            if ok: w = wn
        base, Zb = Zo_[b], Zt_[b]
        while spent < budget:
            score = base + Zb @ w; mu = np.where(~seen)[0]; c = mu[np.lexsort((tb[mu], -score[mu]))[0]]
            seen[c] = True; hits += int(ya[b[c]]); spent += 1; L.append(int(b[c])); since += 1
            if since >= R:
                wn, ok = refit()
                if ok: w = wn; since = 0
        hits_b.append(hits); diag.append(_diag(L, ya, za_))
    return hits_b, diag

def freeze_plain(Xa_, carry_hits, L, k, batches_, budget, seed):
    """carry_plain hits through batches 1..k, then the rule refit on the k*B labels in hand (carry_plain's own fit for the next pick) frozen and reused free (top B by score) in batches k+1..12. Verbatim L6a freeze_plain with k given."""
    Lk = L[: k * budget]; clf = rule(Xa_[Lk], yall[Lk], 500) if 0 < yall[Lk].sum() < len(Lk) else None; hits = list(carry_hits[:k])
    for b in batches_[k:]:
        bind(b); hits.append(run("thr_agreement", budget, seed) if clf is None else int(y[np.argsort(-clf.predict_proba(Xa_[b])[:, 1])[:budget]].sum()))
    return hits, {"k": k, "n_fit": len(Lk), "fit_err_rate": float(yall[Lk].mean()), "fallback_thr": clf is None}

# ---------------- run ----------------
arms = {k: [] for k in ("thr_agreement", "restart_offset")}; base_errors = []
for bi, b in enumerate(batches):
    bind(b); base_errors.append(int(y.sum()))
    arms["thr_agreement"].append(run("thr_agreement", BUDGET, ORDER)); arms["restart_offset"].append(run_offset(BUDGET, ORDER, 0, 5, 10))
diag, info = {}, {}
arms["carry_plain"], diag["carry_plain"], Lplain = carry_plain(Xa, yall, batches, BUDGET, ORDER, za)
for W in WINDOWS: arms[f"carry_plain_window_{W}"], diag[f"carry_plain_window_{W}"], _ = carry_plain(Xa, yall, batches, BUDGET, ORDER, za, win=W)
arms["carry_plain_decay"], diag["carry_plain_decay"], _ = carry_plain(Xa, yall, batches, BUDGET, ORDER, za, hl=HALF_LIFE)
arms["carry_offset_lam1"], diag["carry_offset_lam1"] = carry_offset(Zo, Zt, yall, batches, BUDGET, ORDER, 1, za)
for k in FREEZE_K: arms[f"freeze_plain_k{k}"], info[f"freeze_plain_k{k}"] = freeze_plain(Xa, arms["carry_plain"], Lplain, k, batches, BUDGET, ORDER)
arms["reset_at_switch"], diag["reset_at_switch"], _ = carry_plain(Xa, yall, batches, BUDGET, ORDER, za, reset_at=SWITCH)
bind(batches[0]); chk = {"batch1_carry_plain_eq_run_online": bool(arms["carry_plain"][0] == run_online(BUDGET, ORDER, n0=3, R=1, seeded=False))}
for k in FREEZE_K: chk[f"freeze_plain_k{k}_eq_carry_plain_batches_1_to_{k}"] = bool(arms[f"freeze_plain_k{k}"][:k] == arms["carry_plain"][:k])
chk[f"reset_at_switch_eq_carry_plain_batches_1_to_{SWITCH}"] = bool(arms["reset_at_switch"][:SWITCH] == arms["carry_plain"][:SWITCH])
for W in WINDOWS:  # a window of W labels cannot bind before more than W labels are in hand: labels in hand before any pick of batch c are <= c*B - 1
    nb_eq = (W + 1) // BUDGET; chk[f"window_{W}_eq_carry_plain_batches_1_to_{nb_eq}"] = bool(arms[f"carry_plain_window_{W}"][:nb_eq] == arms["carry_plain"][:nb_eq])
out = {"feats_A": pathA, "feats_B": pathB, "label": LABEL, "features": cheap, "mode": MODE, "form": FORM, "budget": BUDGET, "m": M, "n_batches": NB, "switch_after_batch": SWITCH, "order_seed": ORDER, "policy_seed": ORDER,
       "batch_base_errors": base_errors, "arms": {k: [int(v) for v in vs] for k, vs in arms.items()}, "diag": diag, "arm_info": info, "checks": chk, "offset_c": OFFSET_C, "windows": list(WINDOWS), "half_life": HALF_LIFE, "freeze_k": list(FREEZE_K)}
json.dump(out, open(out_path, "w"), indent=1)
tot = lambda k: int(np.sum(arms[k])); post = lambda k: int(np.sum(arms[k][SWITCH:]))
print(f"{os.path.basename(pathA)} -> {os.path.basename(pathB)} B {BUDGET} order {ORDER} mode {MODE} checks_all_true={all(chk.values())}; hits batches 7-12 " + " ".join(f"{k}={post(k)}" for k in arms), flush=True)
