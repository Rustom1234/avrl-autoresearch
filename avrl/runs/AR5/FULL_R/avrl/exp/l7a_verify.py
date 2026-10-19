"""L7a independent audit (CPU). Usage: l7a_verify.py <runs/L7a>. (1) re-derives the verdict-fold headline (best persistent / freeze_plain_k6, batches 7-12) per cell from the raw per-seed drift JSONs with its own bootstrap
seed (424242) and compares it to result.json; (2) re-implements thr_agreement, carry_plain, carry_plain_window_250, carry_plain_decay, reset_at_switch and freeze_plain_k6 from the two parquet tables in a different code
structure (the stream is built batch by batch from the table that is live at that batch, the label store holds copies of the z rows and labels rather than indices, a masked argsort tie rule written out explicitly) for
3 (direction, B, order seed) cells and compares all 12 per-batch hits, plus batch_base_errors; (3) prints the premise numbers (label correlation, judge feature means, error rates, row alignment)."""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
rd = sys.argv[1]; R = json.load(open(os.path.join(rd, "result.json"))); ok = True
J3, J7 = "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct-AWQ"; DIRS = {"d1 3B->7B-AWQ": (1, J3, J7), "d2 7B-AWQ->3B": (2, J7, J3)}
PERSIST = ["carry_plain", "carry_plain_window_125", "carry_plain_window_250", "carry_plain_window_500", "carry_plain_decay", "carry_offset_lam1"]
CJ = "agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std".split(",")
if R["verdict"]["outcome"] == "VOID": print("result.json is VOID, nothing to audit"); sys.exit(2)
ld = lambda d, B, o, m="drift": json.load(open(os.path.join(rd, f"l7a_d{d}_B{B}_o{o}_{m}.json")))
rng = np.random.default_rng(424242); IDX = rng.integers(0, 8, (10000, 8))
print("== (1) headline from raw JSONs (verdict fold: best persistent chosen on order seeds 0-7, reported on 8-15)")
for dn, (d, ja, jb) in DIRS.items():
    for B in (62, 125):
        D = [ld(d, B, o)["arms"] for o in range(16)]; T = lambda k: np.array([sum(x[k][6:]) for x in D], float)
        best = max(PERSIST, key=lambda k: T(k)[:8].mean()); num, den = T(best)[8:], T("freeze_plain_k6")[8:]; rb = num[IDX].mean(1) / den[IDX].mean(1)
        c = R["cells"][f"{dn} B={B}"]["verdict_folds"]["A"]["best_over_freeze_plain_k6"]; bc = R["cells"][f"{dn} B={B}"]["verdict_folds"]["A"]["best_arm"]
        good = best == bc and abs(num.mean() / den.mean() - c["ratio"]) < 1e-9 and abs(np.percentile(rb, 2.5) - c["ci_lo"]) < 0.01; ok &= bool(good)
        print(f"{dn} B={B}: best {best} (summarizer {bc}), best/freeze_k6 {num.mean() / den.mean():.4f} CI lo {np.percentile(rb, 2.5):.4f} (summarizer {c['ratio']:.4f}, {c['ci_lo']:.4f}) {'OK' if good else 'MISMATCH'}")
print("== (2) independent re-implementation from the parquet tables")
fit = lambda Z, y, w=None: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")).fit(Z, y, **({} if w is None else {"logisticregression__sample_weight": w}))
tabs = {j: pd.read_parquet(f"/homes/rdubash/avrl_live/FULL_R/avrl/runs/E04/feats_{j}.parquet") for j in (J3, J7)}
def live(j, b):  # (per-batch z of the cheap features, labels, agreement) for batch rows b of table j, standardized on the batch's own rows
    X = tabs[j][CJ].fillna(0).values[b]; sd = X.std(0); return (X - X.mean(0)) / np.where(sd > 0, sd, 1.0), tabs[j]["judge_error"].values[b].astype(int), tabs[j]["agreement"].values[b]
def carry(stream, B, seed, window=None, half_life=None, reset_at=None):
    sz, sy, hits = [], [], []  # label store holds copies of the z rows and labels in collection order
    for bi, (Z, y, ag) in enumerate(stream):
        if bi == reset_at: sz, sy = [], []
        score = np.random.default_rng(seed).random(250) * 1e-3; avail = np.ones(250, bool); h = 0; last = None
        for _ in range(B):
            n = len(sy); lo = max(0, n - window) if window else 0; yy = np.array(sy[lo:])
            if n >= 3 and n != last and 0 < yy.sum() < len(yy):
                w = None if half_life is None else 0.5 ** ((len(yy) - 1 - np.arange(len(yy))) / half_life)
                score = fit(np.array(sz[lo:]), yy, w).predict_proba(Z)[:, 1]; last = n
            # tie rule: numpy default argsort over the unlabeled rows, as in the L6a code (an early fit on 3 labels can put 16 rows at probability exactly 1.0; a first-index argmax resolves those ties differently and was seen to differ at d1 B=125 order 9)
            u = np.flatnonzero(avail); i = int(u[np.argsort(-score[u])[0]]); avail[i] = False; h += int(y[i]); sz.append(Z[i]); sy.append(int(y[i]))
        hits.append(h)
    return hits, np.array(sz), np.array(sy)
for dn, B, o in (("d1 3B->7B-AWQ", 62, 3), ("d2 7B-AWQ->3B", 125, 11), ("d1 3B->7B-AWQ", 125, 9)):
    d, ja, jb = DIRS[dn]; x = ld(d, B, o); perm = np.random.default_rng(10000 + o).permutation(3000)
    stream = [live(ja if bi < 6 else jb, perm[bi * 250:(bi + 1) * 250]) for bi in range(12)]
    mine = {"batch_base_errors": [int(s[1].sum()) for s in stream], "thr_agreement": [int(y[np.lexsort((np.random.default_rng(o).random(250), ag))[:B]].sum()) for _, y, ag in stream]}
    mine["carry_plain"], Zs, ys = carry(stream, B, o); mine["carry_plain_window_250"] = carry(stream, B, o, window=250)[0]; mine["carry_plain_decay"] = carry(stream, B, o, half_life=250)[0]; mine["reset_at_switch"] = carry(stream, B, o, reset_at=6)[0]
    clf = fit(Zs[:6 * B], ys[:6 * B])  # rule refit on the 6*B labels in hand after batch 6 (all table A, store in collection order), frozen
    mine["freeze_plain_k6"] = mine["carry_plain"][:6] + [int(y[np.argsort(-clf.predict_proba(Z)[:, 1])[:B]].sum()) for Z, y, _ in stream[6:]]
    for k, v in mine.items():
        g = x["batch_base_errors"] if k == "batch_base_errors" else x["arms"][k]; same = (v == g); ok &= bool(same)
        print(f"{dn} B={B} order {o} {k}: {'equal' if same else 'MISMATCH'}" + ("" if same else f"\n  mine {v}\n  json {g}"))
print("== (3) premise: row alignment, label correlation, judge feature means")
a, b = tabs[J3], tabs[J7]; al = np.array_equal(a["pid"].values, b["pid"].values) and all(np.array_equal(a[c].fillna(0).values, b[c].fillna(0).values) for c in CJ[:6]); ok &= bool(al)
print(f"row aligned (pid + 6 judge-independent features identical): {al}; judge_error corr {np.corrcoef(a['judge_error'], b['judge_error'])[0, 1]:.3f}; error rate 3B {a['judge_error'].mean():.3f}, 7B-AWQ {b['judge_error'].mean():.3f}; "
      f"judge_top mean 3B {a['judge_top'].mean():.3f}, 7B-AWQ {b['judge_top'].mean():.3f}; judge_mean 3B {a['judge_mean'].mean():.3f}, 7B-AWQ {b['judge_mean'].mean():.3f}")
print("ALL CHECKS", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
