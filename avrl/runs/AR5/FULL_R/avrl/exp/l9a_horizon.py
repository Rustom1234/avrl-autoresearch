"""L9a: long-horizon replay (CPU only). Copied from exp/l7b_design.py (functions copied, never imported; l7b files untouched). Every copied function is verbatim; only the layout, the arm set and the outputs change.
Usage: l9a_horizon.py <feats.parquet> <label> <cheap_col1,...> <budget B> <seed> <mode> <H> <out.json>
  mode "iid"        : H batches, each of m = 250 prompts drawn uniformly WITH replacement from the 3,000-prompt table, stream rng = default_rng(30000 + seed) (batch i is the i-th draw, so an H = 12 prefix
                      of the H = 60 stream is the H = 12 stream), policy seed = stream seed. Arms: thr_agreement, uniform, carry_plain, amort_{seed,split,strat}_n and freeze_plain_n for n in 125, 250, 500, 1000
                      (definitions as L7b below; n = 1000 gives k = ceil(n / B) up to 17). Also records the memorisation control (final carry_plain fit, AUC on labelled vs never-labelled prompts, plus a split-half
                      in-sample vs out-of-sample AUC control on the labelled prompts) and carry_plain's repeat-payment statistics.
  mode "iid_nr"     : the pre-registered memorisation remedy (run only for the control): as "iid" but prompts that carry_plain has already paid for are removed from every later candidate pool. Batch i is m = 250
                      draws with replacement from the table minus the prompts in carry_plain's label list so far (default_rng(30000 + seed), so batch 1 equals the iid batch 1); the stream ends when fewer than 250
                      unpaid prompts remain (the 3,000-prompt table cannot supply 60 fresh batches: carry_plain pays B per batch). One common stream for all arms, defined by the learner's payments; H is the cap.
  mode "disjoint12" : the L7b permutation layout (12 disjoint batches, default_rng(10000 + seed)), the full L7b arm set; the hits must equal runs/L7b/design_*_o<seed>.json exactly (regression gate in l9a_summarize.py).
Below, the L7b docstring is kept for the references.
L7b docstring (first paragraph names the copy chain): L7b: designed (non-adaptive) calibration sets vs persistent carry (CPU only). Copied from exp/l6a_amort.py (functions copied, never imported; l6a files untouched). Everything from L6a is verbatim
(rule, run, run_online, ridge_logit, run_offset, _diag, carry_plain, carry_offset, freeze_plain, the five L6a arms and their random streams) except: amortized() gains two pick kinds ("strat", "bnd"), the
calibration fit sets record coverage statistics (cov()), and the arm loop / in-job checks include the new arms. The L6a arms must reproduce runs/L6a exactly (checked by l7b_summarize.py).
The module globals the copied functions read (Xs, y, agree, ndist, Zoff, Zoth) are set by bind().
Usage: l7b_design.py <feats.parquet> <label> <cheap_col1,...> <budget B> <order_seed> <out.json>
New in L7b (arms fixed in advance, none chosen on held-out data), same calibration structure as amort_rand_n (k = ceil(n / B); batches 1..k-1 are B calibration picks each; in batch k the remaining
n - (k-1)*B calibration labels are picks, ONE rule() fit on exactly the n labels, the rest of the batch-k budget goes to the highest-scored unlabeled prompts of batch k, the fit is reused free as top-B by score
in batches k+1..12; calibration labels are charged and counted as hits; a one-class fit set falls back to thr_agreement):
  amort_strat_n : each calibration pick set takes equal numbers from each of NBIN = 10 quantile bins of the batch's label-free agreement. Bins are the 10 equal-size rank groups of the batch ordered by
                  (agreement, policy-seeded random tie-break), i.e. the thr_agreement order cut into tenths; take // 10 picks per bin, the take % 10 remainder goes to bins drawn without replacement by the
                  policy-seeded rng; inside a bin picks are uniform without replacement (agreement takes only 8 distinct values, so ties are broken at random, never by label).
  amort_bnd_n   : picks are the contiguous within-batch lowest-agreement rank window (thr_agreement order, same tie-break stream) of size take centred on the thr_agreement cutoff rank B, i.e. ranks
                  B - take//2 .. B - take//2 + take - 1 (a full batch: ranks B/2 .. 3B/2, B = 62: 31..92, B = 125: 63..187; the last calibration batch gets the smaller centred window).
Below, the L6a docstring is kept for the references.
L6a docstring:
Design (identical to L5a): permute the 3,000 prompts with default_rng(10000 + order_seed), 12 consecutive disjoint batches of m = 250, per-step budget B (62 or 125), policy seed = order seed.
Reference arms (verbatim L5a, must reproduce runs/L5a on order seeds 0..7, checked by l6a_summarize.py): thr_agreement, uniform, restart_offset, carry_plain, carry_offset_lam1, static_out (+ restart_plain, self-check only).
New arms, fixed in advance, none chosen on held-out data. k = ceil(n / B) calibration batches, n in 125, 250, 500. The rule is fit ONCE and then reused free (top B by score) in batches k+1..12:
  amort_seed_n  : batches 1..k are thr_agreement picks (lowest agreement, random ties; hits identical to thr_agreement); every label paid in those k batches (k*B >= n, see n_fit) is a calibration label,
                  one rule() (StandardScaler + balanced logistic) fit on them at the end of batch k.
  amort_rand_n  : batches 1..k-1 are B uniformly random picks each; in batch k the remaining n - (k-1)*B calibration labels are random picks, the rule is fit on exactly the n random labels, and the rest of the
                  batch-k budget goes to the highest-scored unlabeled prompts of batch k (recal_charged style: calibration labels are charged and counted as hits).
  amort_split_n : as amort_rand_n but every calibration pick is random among the prompts with n_distinct > 1 of its batch (the L2a range-restriction guard).
  A one-class calibration set falls back to thr_agreement (remainder of batch k and all later batches).
Diagnostic only (not in the verdict): freeze_plain_n = carry_plain through batches 1..k, then the rule refit on all k*B labels in hand at the end of batch k (the fit carry_plain itself would use for the first pick of batch k+1)
frozen and reused free in batches k+1..12."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

K, M, OFFSET_C = 1, 250, 2.0
NBIN = 10
args = sys.argv[1:]
path, LABEL, cheap = args[0], args[1], args[2].split(","); BUDGET = int(args[3]); ORDER = int(args[4]); MODE = args[5]; H = int(args[6]); out_path = args[7]
assert MODE in ("iid", "iid_nr", "disjoint12")
if MODE == "disjoint12": NB, CAL_N, KINDS = 12, (125, 250, 500), ("seed", "rand", "split", "strat", "bnd")   # exactly the L7b configuration
else: NB, CAL_N, KINDS = H, (125, 250, 500, 1000), ("seed", "split", "strat")
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

# ---------------- copied from exp/l5a_carry.py (verbatim, except: run() keeps only its uniform / thr_agreement branches, charged_order() dropped, carry_plain() also returns L) ----------------
def rule(Xa, ya, max_iter=1000): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, class_weight="balanced")).fit(Xa, ya)

def run(policy, budget, seed):
    rng = np.random.default_rng(seed)
    if policy == "uniform": order = rng.permutation(len(y))
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

def _diag(L, ya, za):
    """selection-bias diagnostics of the labels in hand: count, error rate, mean global z-agreement (population mean is 0)."""
    L = np.asarray(L); return {"n": int(len(L)), "err_rate": float(ya[L].mean()) if len(L) else None, "mean_z_agree": float(za[L].mean()) if len(L) else None}

def carry_plain(Xa, ya, batches, budget, seed, za, L=None):
    """greedy_n3_R1 with every earlier label kept. Labels in hand L (global indices, label order). Before every pick, if >= 3 labels are in hand with both classes and they changed since the
    last fit, the L4a rule is refit on all of L and scores the current batch; otherwise the random-start score (rng.random * 1e-3, as run_online) is used. Equals run_online on batch 1."""
    L, hits_b, diag = ([] if L is None else L), [], []  # L9a: an external list may be passed so a lazy `batches` (iid_nr) can read the labels in hand; None = verbatim L7b behaviour
    for b in batches:
        rng = np.random.default_rng(seed); score = rng.random(len(b)) * 1e-3; seen = np.zeros(len(b), bool); hits = spent = 0; fitted_n = -1
        while spent < budget:
            n = len(L)
            if n != fitted_n and n >= 3 and 0 < ya[L].sum() < n:
                score = rule(Xa[L], ya[L], 500).predict_proba(Xa[b])[:, 1]; fitted_n = n
            mu = np.where(~seen)[0]; c = mu[np.argsort(-score[mu])][:1]; seen[c] = True; hits += int(ya[b[c]].sum()); spent += 1; L.append(int(b[c[0]]))
        hits_b.append(hits); diag.append(_diag(L, ya, za))
    return hits_b, diag, L

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

# ---------------- new in L6a ----------------
def cov(L):
    """coverage of a fit set L (global indices): global label-free z-agreement (population mean 0, SD 1) min / max / SD, label error rate, share with n_distinct > 1. New in L7b."""
    L = np.asarray(L); z = Zg[L, IA]
    return {"fit_z_min": float(z.min()), "fit_z_max": float(z.max()), "fit_z_sd": float(z.std()), "fit_err_rate": float(yall[L].mean()), "fit_nd_gt1_share": float((ND_ALL[L] > 1).mean())}

def amortized(kind, n, batches, budget, seed):
    """Amortized fixed rule (see module docstring). Returns per-batch hits and a small info dict. Uses bind() on every batch. L7b adds kinds "strat" and "bnd" and the cov() fields."""
    k = -(-n // budget); rng = np.random.default_rng(seed); hits, Lc, clf = [], [], None; spread, wins = [], []
    def cal_pick(take):  # calibration picks (local indices) inside the bound batch
        if kind == "seed": return np.lexsort((np.random.default_rng(seed).random(len(y)), agree))[:take]  # == run("thr_agreement") order
        if kind == "rand": return rng.permutation(len(y))[:take]
        if kind == "strat":  # equal picks from each of NBIN rank-quantile bins of the thr_agreement order (random ties), remainder to rng-drawn bins, uniform inside a bin
            order = np.lexsort((rng.random(len(y)), agree)); bins = np.array_split(order, NBIN); cnt = np.full(NBIN, take // NBIN); cnt[rng.choice(NBIN, take % NBIN, replace=False)] += 1
            spread.append(int(cnt.max() - cnt.min())); return np.concatenate([rng.choice(bn, c, replace=False) for bn, c in zip(bins, cnt)])
        if kind == "bnd":  # contiguous lowest-agreement rank window of size take centred on the thr_agreement cutoff rank B (thr_agreement order and tie-break stream)
            order = np.lexsort((np.random.default_rng(seed).random(len(y)), agree)); s = budget - take // 2; wins.append((s, s + take)); return order[s:s + take]
        pool = np.where(ndist > 1)[0]; assert len(pool) >= take, (len(pool), take); return rng.choice(pool, take, replace=False)
    def fit(L): return rule(Xall[L], yall[L]) if 0 < yall[L].sum() < len(L) else None  # None = one-class set, fall back to thr_agreement
    info = {"k": k, "n_nominal": n}
    for bi, b in enumerate(batches):
        bind(b)
        if bi < k - 1 or (kind == "seed" and bi == k - 1):  # whole batch is calibration (seed: thr_agreement picks, so hits equal thr_agreement)
            c = cal_pick(budget); hits.append(int(y[c].sum())); Lc += b[c].tolist()
            if bi == k - 1: clf = fit(Lc); info.update(n_fit=len(Lc), cal_err_rate=float(yall[Lc].mean()), fallback_thr=clf is None, **cov(Lc))
        elif bi == k - 1:  # rand / split: last n - (k-1)*B calibration labels, rule fit on exactly n labels, the rest of the batch budget ranked by it
            c = cal_pick(n - bi * budget); Lc += b[c].tolist(); clf = fit(Lc); info.update(n_fit=len(Lc), cal_err_rate=float(yall[Lc].mean()), fallback_thr=clf is None, **cov(Lc))
            if clf is None: o = np.lexsort((np.random.default_rng(seed).random(len(y)), agree)); rest = o[~np.isin(o, c)]
            else: sc = clf.predict_proba(Xs)[:, 1]; sc[c] = -1; rest = np.argsort(-sc)
            hits.append(int(y[np.concatenate([c, rest[: budget - len(c)]])].sum()))
        elif clf is None: hits.append(run("thr_agreement", budget, seed))
        else: hits.append(int(y[np.argsort(-clf.predict_proba(Xs)[:, 1])[:budget]].sum()))  # frozen rule reused free, top B by score
    if kind == "strat": info["bin_count_spread_max"] = max(spread)
    if kind == "bnd": info["cal_windows"] = wins
    return hits, info

def freeze_plain(carry_hits, L, n, batches, budget, seed):
    """Diagnostic: carry_plain hits through batches 1..k, then the rule refit on the k*B labels in hand (carry_plain's own fit for the next pick) frozen and reused free."""
    k = -(-n // budget); Lk = L[: k * budget]; clf = rule(Xall[Lk], yall[Lk], 500) if 0 < yall[Lk].sum() < len(Lk) else None; hits = list(carry_hits[:k])
    for b in batches[k:]:
        bind(b); hits.append(run("thr_agreement", budget, seed) if clf is None else int(y[np.argsort(-clf.predict_proba(Xs)[:, 1])[:budget]].sum()))
    return hits, {"k": k, "n_fit": len(Lk), **cov(Lk), "fallback_thr": clf is None}

def memo(L):
    """Memorisation control (new in L9a, iid mode). The final carry_plain fit (rule on ALL paid labels L, with multiplicities) scores the whole table; AUC against judge_error on the distinct prompts it labelled (S)
    and on the prompts it never labelled (U); primary gap = AUC_S - AUC_U (pre-registered trigger: gap > 0.02 in any cell). S is selected (the learner picks high-score prompts), so a range-restriction effect can
    push the primary gap either way; the split-half control isolates memorisation from selection: S is split at random into halves A and B, the rule is refit on the labels of A only, gap_split = AUC on A (in-sample)
    minus AUC on B (prompts the fit never saw, same selected population)."""
    Lc = np.asarray(L); S = np.unique(Lc); U = np.setdiff1d(np.arange(N), S)
    auc = lambda sc, ix: float(roc_auc_score(yall[ix], sc[ix])) if len(ix) and 0 < yall[ix].sum() < len(ix) else None
    sub = lambda a, b: None if a is None or b is None else a - b
    p = rule(Xall[Lc], yall[Lc], 500).predict_proba(Xall)[:, 1]
    h = np.random.default_rng(40000 + ORDER).permutation(S); Ah, Bh = h[: len(S) // 2], h[len(S) // 2:]; inA = np.isin(Lc, Ah)
    pA = rule(Xall[Lc[inA]], yall[Lc[inA]], 500).predict_proba(Xall)[:, 1]
    a_s, a_u, a_a, a_b = auc(p, S), auc(p, U), auc(pA, Ah), auc(pA, Bh)
    return {"n_labels": int(len(Lc)), "n_distinct_labelled": int(len(S)), "n_never_labelled": int(len(U)), "auc_labelled": a_s, "auc_never_labelled": a_u, "gap_labelled_minus_never": sub(a_s, a_u),
            "auc_split_in_sample": a_a, "auc_split_held_out": a_b, "gap_split_in_minus_held_out": sub(a_a, a_b), "err_rate_labelled": float(yall[S].mean()), "err_rate_never_labelled": float(yall[U].mean()) if len(U) else None}

if MODE == "disjoint12":
    perm = np.random.default_rng(10000 + ORDER).permutation(N); batches = [perm[i * M:(i + 1) * M] for i in range(NB)]
    assert len(perm) == NB * M and len(np.unique(np.concatenate(batches))) == N and BUDGET < M
else:  # iid with replacement: batch i is the i-th draw of m prompts from default_rng(30000 + seed)
    srng = np.random.default_rng(30000 + ORDER); batches = [srng.integers(0, N, M) for _ in range(NB)] if MODE == "iid" else None; assert BUDGET < M
za = Zg[:, IA]; diag = {}
if MODE == "iid_nr":  # carry_plain first: its payments define the common no-repeat stream
    class NRStream:
        def __init__(self, L, seed, cap): self.L, self.rng, self.cap, self.out = L, np.random.default_rng(30000 + seed), cap, []
        def __iter__(self):
            while len(self.out) < self.cap:
                unpaid = np.setdiff1d(np.arange(N), np.asarray(self.L, dtype=int))
                if len(unpaid) < M: return
                b = unpaid[self.rng.integers(0, len(unpaid), M)]; self.out.append(b); yield b
    Lnr = []; nrs = NRStream(Lnr, ORDER, H); arms = {}
    arms["carry_plain"], diag["carry_plain"], Lplain = carry_plain(Xall, yall, nrs, BUDGET, ORDER, za, L=Lnr); batches = nrs.out; NB = len(batches); assert NB >= 2
arms = {**({"carry_plain": arms["carry_plain"]} if MODE == "iid_nr" else {}), **{k: [] for k in ("thr_agreement", "uniform") + (("restart_offset", "restart_plain", "static_out") if MODE == "disjoint12" else ())}}; base_errors = []
for bi, b in enumerate(batches):
    bind(b); base_errors.append(int(y.sum()))
    arms["thr_agreement"].append(run("thr_agreement", BUDGET, ORDER)); arms["uniform"].append(run("uniform", BUDGET, ORDER))
    if MODE == "disjoint12":
        arms["restart_offset"].append(run_offset(BUDGET, ORDER, 0, 5, 10)); arms["restart_plain"].append(run_online(BUDGET, ORDER, n0=3, R=1, seeded=False))
        outside = np.setdiff1d(np.arange(N), b); sc_s = rule(Xall[outside], yall[outside]).predict_proba(Xall[b])[:, 1]; arms["static_out"].append(int(y[np.argsort(-sc_s)[:BUDGET]].sum()))
if MODE != "iid_nr": arms["carry_plain"], diag["carry_plain"], Lplain = carry_plain(Xall, yall, batches, BUDGET, ORDER, za)
if MODE == "disjoint12": arms["carry_offset_lam1"], diag["carry_offset_lam1"] = carry_offset(Zoff_g, Zoth_g, yall, batches, BUDGET, ORDER, 1, za)
info = {}
for n in CAL_N:
    for kind in KINDS: arms[f"amort_{kind}_{n}"], info[f"amort_{kind}_{n}"] = amortized(kind, n, batches, BUDGET, ORDER)
    arms[f"freeze_plain_{n}"], info[f"freeze_plain_{n}"] = freeze_plain(arms["carry_plain"], Lplain, n, batches, BUDGET, ORDER)
chk = {"batch1_carry_plain_eq_restart_plain": bool(arms["carry_plain"][0] == arms["restart_plain"][0])} if MODE == "disjoint12" else {}
chk["all_arms_have_NB_batches"] = bool(all(len(v) == NB for v in arms.values()) and len(base_errors) == NB and len(Lplain) == NB * BUDGET)
for n in CAL_N:
    k = -(-n // BUDGET); chk[f"amort_seed_{n}_eq_thr_batches_1_to_{k}"] = bool(arms[f"amort_seed_{n}"][:k] == arms["thr_agreement"][:k])
    chk[f"freeze_plain_{n}_eq_carry_plain_batches_1_to_{k}"] = bool(arms[f"freeze_plain_{n}"][:k] == arms["carry_plain"][:k])
    chk[f"amort_{n}_fit_set_size_ok"] = bool(info[f"amort_seed_{n}"]["n_fit"] == k * BUDGET and all(info[f"amort_{q}_{n}"]["n_fit"] == n for q in KINDS if q != "seed"))
    chk[f"amort_strat_{n}_bins_balanced_within_1"] = bool(info[f"amort_strat_{n}"]["bin_count_spread_max"] <= 1)
    if "bnd" in KINDS: chk[f"amort_bnd_{n}_windows_centred_on_cutoff"] = bool(len(info[f"amort_bnd_{n}"]["cal_windows"]) == k and all(abs((lo + hi) / 2 - BUDGET) <= 1 and 0 <= lo and hi <= M for lo, hi in info[f"amort_bnd_{n}"]["cal_windows"]))
out = {"feats": path, "label": LABEL, "features": cheap, "budget": BUDGET, "m": M, "n_batches": NB, "order_seed": ORDER, "policy_seed": ORDER, "mode": MODE, "batch_base_errors": base_errors,
       "arms": {k: [int(v) for v in vs] for k, vs in arms.items()}, "diag": diag, "arm_info": info, "checks": chk, "offset_c": OFFSET_C, "cal_n": list(CAL_N), "nbin": NBIN, "kinds": list(KINDS)}
if MODE in ("iid", "iid_nr"):
    Ls = np.asarray(Lplain); _, first = np.unique(Ls, return_index=True); rep = np.ones(len(Ls), bool); rep[first] = False  # rep: this pick re-pays a prompt carry_plain already labelled earlier
    allb = np.concatenate(batches)
    out["stream"] = {"stream_rng_seed": 30000 + ORDER, "n_distinct_prompts_drawn": int(len(np.unique(allb))), "mean_distinct_share_per_batch": float(np.mean([len(np.unique(b)) / M for b in batches])),
                     "carry_plain_repeat_pick_share": float(rep.mean()), "carry_plain_hits_total": int(yall[Ls].sum()), "carry_plain_hits_on_repeat_picks": int(yall[Ls][rep].sum()), "n_batches_run": int(NB), "final_unpaid_prompts": int(N - len(first))}
    out["memorisation"] = memo(Lplain)
json.dump(out, open(out_path, "w"), indent=1)
tot = lambda k: int(np.sum(arms[k]))
print(f"{os.path.basename(path)} B {BUDGET} seed {ORDER} mode {MODE} NB {NB} checks_all_true={all(chk.values())}; total hits " + " ".join(f"{k}={tot(k)}" for k in arms), flush=True)
if MODE in ("iid", "iid_nr"): print("memorisation", json.dumps(out["memorisation"]), "stream", json.dumps(out["stream"]), flush=True)
