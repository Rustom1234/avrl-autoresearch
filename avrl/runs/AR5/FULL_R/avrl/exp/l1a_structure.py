"""L1a structure statistics (CPU). Usage: l1a_structure.py <runs/L1a> [feats.parquet] [tag ...]   (no tag: all six tables, one process each, up to 6 in parallel)

Per table (judge x accept rule x label population), all AUCs out-of-fold with 5 folds assigned by pid, 3 random pid-to-fold seeds, class_weight balanced logistic on standardized features:
  judge_logit alone, cheap_r, cheap_r + 16 PCs, cheap_r + honest two-stage OOF target encoding of cell30 / cell100 / cell300 (shrinkage 10 rows; shrinkage 80 rows ~ 10 prompts as a robustness column),
  HistGradientBoosting on cheap_r vs cheap_r + PCs (CV seed 0), and the labelled positive control cheap_r + agreement (agreement is read here, in memory, from the feats file; it is NEVER written to a table).
Deltas are relative to cheap_r. Pid-level bootstrap (200 resamples of pids, the same resamples for every delta, delta averaged over the 3 CV seeds exactly like the point estimate).
Pid-level permutation null (20 permutations, CV seed 0): the PC vector and the three cell ids of each pid are replaced by those of a random donor pid; delta AUC against the same cheap_r OOF predictions.
Two-stage target encoding: for each outer fold, training rows are encoded from the other inner folds (5 inner folds by pid) and test rows from all training rows; value = (sum y + s * prior) / (n + s)."""
import os, sys, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

CHEAP_R = ["judge_logit", "log_tokens", "truncated", "has_decimal", "is_negative", "log_abs_ans", "empty_answer", "log_qlen"]
PCS = [f"pc{i}" for i in range(16)]; CELLS = ["cell30", "cell100", "cell300"]
NF, CV_SEEDS, SHR, SHR2, NBOOT, NPERM = 5, (0, 1, 2), 10, 80, 200, 20
TAGS = ["3B_q_corrupted", "3B_q_judgewrong", "7B_q_corrupted", "7B_q_judgewrong", "7B_p05_corrupted", "7B_p05_judgewrong"]

def auc_w(pred, y, w=None, grp=None):
    """Weighted AUC with ties counted 1/2. grp = (unique-inverse ids, n groups) can be precomputed for repeated calls on the same predictions."""
    if grp is None: u, gid = np.unique(pred, return_inverse=True); grp = (gid, len(u))
    gid, ng = grp; w = np.ones(len(y)) if w is None else w
    pos = np.bincount(gid, weights=w * y, minlength=ng); neg = np.bincount(gid, weights=w * (1 - y), minlength=ng)
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))

def logit_model(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))

def fold_ids(pix, n_pid, seed):
    rng = np.random.default_rng(seed); perm = rng.permutation(n_pid); fpid = np.empty(n_pid, int); fpid[perm] = np.arange(n_pid) % NF
    return fpid[pix]

def te_two_stage(cc, y, pix, tr, te, shr, seed):
    ncell = int(cc.max()) + 1; tr_i, te_i = np.where(tr)[0], np.where(te)[0]
    def enc(src, dst):
        s = np.bincount(cc[src], weights=y[src], minlength=ncell); n = np.bincount(cc[src], minlength=ncell); pr = y[src].mean()
        return (s[cc[dst]] + shr * pr) / (n[cc[dst]] + shr)
    up = np.unique(pix[tr_i]); rng = np.random.default_rng(seed + 1000); perm = rng.permutation(up); inner = np.zeros(pix.max() + 1, int); inner[perm] = np.arange(len(perm)) % NF
    f = inner[pix[tr_i]]; out_tr = np.empty(len(tr_i))
    for g in range(NF):
        m = f == g; out_tr[m] = enc(tr_i[~m], tr_i[m])
    return out_tr[:, None], enc(tr_i, te_i)[:, None]

def oof(X, y, fold, extra=None, model=logit_model):
    """Pooled out-of-fold probability. extra(tr, te) -> (columns for train rows, columns for test rows)."""
    p = np.zeros(len(y))
    for k in range(NF):
        tr, te = fold != k, fold == k; Xtr, Xte = X[tr], X[te]
        if extra is not None: a, b = extra(tr, te); Xtr, Xte = np.hstack([Xtr, a]), np.hstack([Xte, b])
        m = model().fit(Xtr, y[tr]); p[te] = m.predict_proba(Xte)[:, 1]
    return p

def hgb(): return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, early_stopping=False, random_state=0)

def run_table(args):
    rd, feats_path, tag = args; t0 = time.time()
    t = pd.read_parquet(os.path.join(rd, f"tab_{tag}.parquet")); lab = "corrupted" if tag.endswith("corrupted") else "judge_wrong"
    f = pd.read_parquet(feats_path).sort_values("pid").reset_index(drop=True); assert (f.pid.values == np.arange(len(f))).all()
    n_pid = len(f); pix = t.pid.values.astype(int); y = t[lab].values.astype(int)
    Xc = t[CHEAP_R].values.astype(float); Xp = t[PCS].values.astype(float); cells = {c: t[c].values.astype(int) for c in CELLS}
    assert np.allclose(Xp, f.loc[pix, PCS].values.astype(float), atol=1e-6) and all((cells[c] == f.loc[pix, c].values).all() for c in CELLS)
    agr = f.loc[pix, "agreement"].values.astype(float)[:, None]  # positive control only, in memory
    P = f[PCS].values.astype(float); CE = {c: f[c].values.astype(int) for c in CELLS}
    folds = {s: fold_ids(pix, n_pid, s) for s in CV_SEEDS}
    preds = {}; auc = {}
    def te_fn(c, shr, s, cc=None):
        cc = cells[c] if cc is None else cc
        return lambda tr, te: te_two_stage(cc, y, pix, tr, te, shr, s)
    for s in CV_SEEDS:
        fo = folds[s]
        preds[("judge_only", s)] = oof(Xc[:, :1], y, fo); preds[("cheap", s)] = oof(Xc, y, fo)
        preds[("pc", s)] = oof(np.hstack([Xc, Xp]), y, fo); preds[("agreement", s)] = oof(np.hstack([Xc, agr]), y, fo)
        for c in CELLS:
            preds[(c, s)] = oof(Xc, y, fo, te_fn(c, SHR, s)); preds[(c + "_s80", s)] = oof(Xc, y, fo, te_fn(c, SHR2, s))
    preds[("hgb_cheap", 0)] = oof(Xc, y, folds[0], model=hgb); preds[("hgb_pc", 0)] = oof(np.hstack([Xc, Xp]), y, folds[0], model=hgb)
    grp = {}
    for k, v in preds.items(): u, gid = np.unique(v, return_inverse=True); grp[k] = (gid, len(u))
    models = sorted({k[0] for k in preds})
    for m in models: auc[m] = {"per_seed": [auc_w(preds[(m, s)], y, grp=grp[(m, s)]) for s in CV_SEEDS if (m, s) in preds]}; auc[m]["mean"] = float(np.mean(auc[m]["per_seed"]))
    for m in models: assert abs(auc[m]["per_seed"][0] - roc_auc_score(y, preds[(m, 0)])) < 1e-9, "weighted AUC disagrees with sklearn"
    auc["judge_logit_raw"] = float(max(roc_auc_score(y, -t.judge_logit.values), roc_auc_score(y, t.judge_logit.values)))
    # deltas vs cheap_r
    pairs = {"pc": ("cheap", "pc"), "agreement": ("cheap", "agreement"), "hgb_pc": ("hgb_cheap", "hgb_pc")}
    for c in CELLS: pairs[c] = ("cheap", c); pairs[c + "_s80"] = ("cheap", c + "_s80")
    rng = np.random.default_rng(12345); present = np.unique(pix); idx_of = np.full(n_pid, -1); idx_of[present] = np.arange(len(present))
    W = []
    for b in range(NBOOT):
        cnt = np.bincount(rng.integers(0, len(present), len(present)), minlength=len(present)); W.append(cnt[idx_of[pix]].astype(float))
    delta = {}
    for name, (base, test) in pairs.items():
        seeds = [s for s in CV_SEEDS if (test, s) in preds]
        pt = [auc[test]["per_seed"][i] - auc[base]["per_seed"][i] for i in range(len(seeds))]
        bs = np.array([np.mean([auc_w(preds[(test, s)], y, w, grp[(test, s)]) - auc_w(preds[(base, s)], y, w, grp[(base, s)]) for s in seeds]) for w in W])
        delta[name] = {"mean": float(np.mean(pt)), "per_seed": [float(v) for v in pt], "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)), "boot_sd": float(bs.std(ddof=1)), "n_seeds": len(seeds)}
    # permutation null, CV seed 0, same cheap_r OOF baseline
    rngp = np.random.default_rng(777); null = {"pc": [], **{c: [] for c in CELLS}}; base0 = auc["cheap"]["per_seed"][0]
    for _ in range(NPERM):
        donor = rngp.permutation(n_pid)[pix]
        null["pc"].append(auc_w(oof(np.hstack([Xc, P[donor]]), y, folds[0]), y) - base0)
        for c in CELLS: null[c].append(auc_w(oof(Xc, y, folds[0], te_fn(c, SHR, 0, CE[c][donor])), y) - base0)
    for name, v in null.items(): delta[name]["null_mean"] = float(np.mean(v)); delta[name]["null_sd"] = float(np.std(v, ddof=1)); delta[name]["null_values"] = [float(x) for x in v]
    gain = delta["agreement"]["mean"]
    rec = {k: (delta[k]["mean"] / gain if gain > 0 else None) for k in ["pc"] + CELLS}
    out = {"tag": tag, "label": lab, "rows": int(len(t)), "pids": int(len(present)), "positives": int(y.sum()), "rate": float(y.mean()), "auc": auc, "delta": delta,
           "agreement_gain": gain, "fraction_of_agreement_gain_recovered": rec, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(rd, f"structure_{tag}.json"), "w"), indent=1)
    d = delta
    print(f"{tag}: n {len(t)} pos {int(y.sum())} | judge_only {auc['judge_only']['mean']:.4f} cheap {auc['cheap']['mean']:.4f} +pc {d['pc']['mean']:+.4f} [{d['pc']['ci_lo']:+.4f},{d['pc']['ci_hi']:+.4f}] null {d['pc']['null_mean']:+.4f}±{d['pc']['null_sd']:.4f} | "
          + " ".join(f"{c} {d[c]['mean']:+.4f} [{d[c]['ci_lo']:+.4f},{d[c]['ci_hi']:+.4f}] null {d[c]['null_mean']:+.4f}±{d[c]['null_sd']:.4f}" for c in CELLS)
          + f" | hgb_pc {d['hgb_pc']['mean']:+.4f} | agreement {auc['agreement']['mean']:.4f} (+{gain:.4f}) recovered pc {rec['pc']:.3f} | {out['seconds']}s", flush=True)
    return tag

if __name__ == "__main__":
    rd = sys.argv[1]; fp = sys.argv[2] if len(sys.argv) > 2 else "runs/p0/feats.parquet"; tags = sys.argv[3:] or TAGS
    bj = json.load(open(os.path.join(rd, "build.json"))); assert bj["gate_passed"], "build gate failed (positives >= 200 and judge AUC reproduction); structure not run"
    if len(tags) == 1: run_table((rd, fp, tags[0]))
    else:
        with Pool(min(6, len(tags))) as pool: pool.map(run_table, [(rd, fp, t) for t in tags], chunksize=1)
    json.dump({t: json.load(open(os.path.join(rd, f"structure_{t}.json"))) for t in TAGS if os.path.exists(os.path.join(rd, f"structure_{t}.json"))}, open(os.path.join(rd, "structure.json"), "w"), indent=1)
