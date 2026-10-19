"""L8a independent audit (CPU). Usage: l8a_verify.py <runs/L8a> [<runs/L7a>]. (1) re-derives the verdict-fold headline per cell from the raw per-seed JSONs with its own bootstrap seed (424242): best forgetting arm chosen on order seeds
0-7 by batches 7-12 hits, best forgetting / carry_plain on 8-15 with CI, carry_plain / best forgetting (kill i), best persistent / thr in 7-9 and 10-12 (kill ii), the PASS / KILL / INCONCLUSIVE outcome, and compares it to result.json;
recomputes the gate (f = +1 runs vs runs/L7a drift, all shared arms and batch_base_errors, 16 seeds) directly from the JSONs; (2) re-implements thr_agreement, carry_plain, carry_plain_window_250, carry_plain_decay, carry_plain_decay_hl100,
reset_at_switch and freeze_plain_k6 from the two parquet tables in a different code structure (the stream is built batch by batch from the table that is live at that batch, with judge_top and judge_mean z negated in post-switch batches
when f = -1; the label store holds copies of the z rows and labels rather than indices, a masked argsort tie rule written out explicitly) for 3 flip cells and 1 gate cell and compares all 12 per-batch hits, plus batch_base_errors;
(3) prints premise numbers (row alignment, rule transfer AUC from table A to table B at f = +1 and f = -1)."""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
rd = sys.argv[1]; r7 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(rd)), "L7a"); R = json.load(open(os.path.join(rd, "result.json"))); ok = True
J3, J7 = "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct-AWQ"; DIRS = {"d1 3B->7B-AWQ": (1, J3, J7), "d2 7B-AWQ->3B": (2, J7, J3)}
FORGET = ["carry_plain_window_125", "carry_plain_window_250", "carry_plain_window_500", "carry_plain_decay", "carry_plain_decay_hl100"]; PERSIST = ["carry_plain"] + FORGET + ["carry_offset_lam1"]
CJ = "agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std".split(","); FL = [CJ.index("judge_top"), CJ.index("judge_mean")]
if R["verdict"]["outcome"] == "VOID": print("result.json is VOID, nothing to audit"); sys.exit(2)
ld = lambda d, B, o, m: json.load(open(os.path.join(rd, f"l8a_d{d}_B{B}_o{o}_{m}.json")))
rng = np.random.default_rng(424242); IDX = rng.integers(0, 8, (10000, 8))
print("== (1a) gate recomputed: f = +1 runs vs runs/L7a drift, every L7a arm + batch_base_errors, 16 order seeds x 2 directions x 2 B")
nd = nv = 0
for d in (1, 2):
    for B in (62, 125):
        for o in range(16):
            g, r = ld(d, B, o, "gate"), json.load(open(os.path.join(r7, f"l7a_d{d}_B{B}_o{o}_drift.json")))
            for k in list(r["arms"]) + ["batch_base_errors"]:
                a_, b_ = (g["batch_base_errors"], r["batch_base_errors"]) if k == "batch_base_errors" else (g["arms"][k], r["arms"][k]); nv += len(b_); nd += sum(x != y for x, y in zip(a_, b_)) + abs(len(a_) - len(b_))
good = nd == 0 and nd == R["regression_gate_vs_L7a"]["batch_value_differences"] and nv == R["regression_gate_vs_L7a"]["batch_values_compared"]; ok &= bool(good)
print(f"per-batch differences {nd} in {nv} values (summarizer: {R['regression_gate_vs_L7a']['batch_value_differences']} in {R['regression_gate_vs_L7a']['batch_values_compared']}) {'OK' if good else 'MISMATCH'}")
print("== (1b) headline from raw JSONs (verdict fold: best forgetting / persistent arm chosen on order seeds 0-7, reported on 8-15)")
nP = nKi = nKii = 0
for dn, (d, ja, jb) in DIRS.items():
    for B in (62, 125):
        D = [ld(d, B, o, "flip")["arms"] for o in range(16)]; T = lambda k, a=6, b=12: np.array([sum(x[k][a:b]) for x in D], float)
        bf = max(FORGET, key=lambda k: T(k)[:8].mean()); bp = max(PERSIST, key=lambda k: T(k)[:8].mean())
        num, den = T(bf)[8:], T("carry_plain")[8:]; rb = num[IDX].mean(1) / den[IDX].mean(1); r = num.mean() / den.mean(); lo = np.percentile(rb, 2.5)
        f1012 = T(bf, 9, 12)[8:].sum() / T("thr_agreement", 9, 12)[8:].sum(); p79 = T(bp, 6, 9)[8:].sum() / T("thr_agreement", 6, 9)[8:].sum(); p1012 = T(bp, 9, 12)[8:].sum() / T("thr_agreement", 9, 12)[8:].sum()
        pas = r >= 1.03 and lo > 1.0 and f1012 >= 1.0; ki = 1 / r >= 0.99; kii = p79 < 1.0 and p1012 < 1.0; nP += pas; nKi += ki; nKii += kii
        c = R["cells"][f"{dn} B={B}"]["verdict_folds"]["A"]; rc = c["best_forgetting_over_carry_plain_7_12"]
        good = (bf == c["best_forgetting_arm"] and bp == c["best_persistent_arm"] and abs(r - rc["ratio"]) < 1e-9 and abs(lo - rc["ci_lo"]) < 0.01 and abs(f1012 - c["best_forgetting_over_thr_10_12"]["ratio"]) < 1e-9
                and abs(p79 - c["best_persistent_over_thr_7_9"]["ratio"]) < 1e-9 and abs(p1012 - c["best_persistent_over_thr_10_12"]["ratio"]) < 1e-9 and pas == c["pass_cell"] and ki == c["kill_i_cell_carry_plain_ge_0.99_best_forgetting"]
                and kii == c["kill_ii_cell_best_persistent_below_thr_in_7_9_and_10_12"]); ok &= bool(good)
        print(f"{dn} B={B}: best forgetting {bf}, best/carry_plain {r:.4f} (CI lo {lo:.4f}), best forgetting/thr 10-12 {f1012:.4f}, best persistent {bp}, /thr 7-9 {p79:.4f}, /thr 10-12 {p1012:.4f}; pass {pas} kill_i {ki} kill_ii {kii}; "
              f"summarizer {rc['ratio']:.4f} ({rc['ci_lo']:.4f}) {'OK' if good else 'MISMATCH'}")
out_mine = "PASS" if nP >= 3 else "KILL" if (nKi >= 3 or nKii >= 3) else "INCONCLUSIVE"; good = out_mine == R["verdict"]["outcome"]; ok &= bool(good)
print(f"outcome recomputed {out_mine} (pass cells {nP}, kill_i cells {nKi}, kill_ii cells {nKii}); summarizer {R['verdict']['outcome']} {'OK' if good else 'MISMATCH'}")
print("== (2) independent re-implementation from the parquet tables")
fit = lambda Z, y, w=None: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")).fit(Z, y, **({} if w is None else {"logisticregression__sample_weight": w}))
tabs = {j: pd.read_parquet(f"/homes/rdubash/avrl_live/FULL_R/avrl/runs/E04/feats_{j}.parquet") for j in (J3, J7)}
def live(j, b, f=1.0):  # (per-batch z of the cheap features, labels, agreement) for batch rows b of table j, standardized on the batch's own rows; the judge_top / judge_mean z times f
    X = tabs[j][CJ].fillna(0).values[b]; sd = X.std(0); Z = (X - X.mean(0)) / np.where(sd > 0, sd, 1.0); Z[:, FL] *= f; return Z, tabs[j]["judge_error"].values[b].astype(int), tabs[j]["agreement"].values[b]
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
            # tie rule: numpy default argsort over the unlabeled rows, as in the L6a code (an early fit on 3 labels can put 16 rows at probability exactly 1.0; a first-index argmax resolves those ties differently)
            u = np.flatnonzero(avail); i = int(u[np.argsort(-score[u])[0]]); avail[i] = False; h += int(y[i]); sz.append(Z[i]); sy.append(int(y[i]))
        hits.append(h)
    return hits, np.array(sz), np.array(sy)
for dn, B, o, m in (("d1 3B->7B-AWQ", 62, 3, "flip"), ("d2 7B-AWQ->3B", 125, 11, "flip"), ("d1 3B->7B-AWQ", 125, 9, "flip"), ("d2 7B-AWQ->3B", 62, 5, "gate")):
    d, ja, jb = DIRS[dn]; x = ld(d, B, o, m); f = -1.0 if m == "flip" else 1.0; perm = np.random.default_rng(10000 + o).permutation(3000)
    stream = [live(ja, perm[bi * 250:(bi + 1) * 250]) if bi < 6 else live(jb, perm[bi * 250:(bi + 1) * 250], f) for bi in range(12)]
    mine = {"batch_base_errors": [int(s[1].sum()) for s in stream], "thr_agreement": [int(y[np.lexsort((np.random.default_rng(o).random(250), ag))[:B]].sum()) for _, y, ag in stream]}
    mine["carry_plain"], Zs, ys = carry(stream, B, o); mine["carry_plain_window_250"] = carry(stream, B, o, window=250)[0]; mine["carry_plain_decay"] = carry(stream, B, o, half_life=250)[0]
    mine["carry_plain_decay_hl100"] = carry(stream, B, o, half_life=100)[0]; mine["reset_at_switch"] = carry(stream, B, o, reset_at=6)[0]
    clf = fit(Zs[:6 * B], ys[:6 * B])  # rule refit on the 6*B labels in hand after batch 6 (all table A, store in collection order), frozen
    mine["freeze_plain_k6"] = mine["carry_plain"][:6] + [int(y[np.argsort(-clf.predict_proba(Z)[:, 1])[:B]].sum()) for Z, y, _ in stream[6:]]
    for k, v in mine.items():
        g = x["batch_base_errors"] if k == "batch_base_errors" else x["arms"][k]; same = (v == g); ok &= bool(same)
        print(f"{dn} B={B} order {o} {m} {k}: {'equal' if same else 'MISMATCH'}" + ("" if same else f"\n  mine {v}\n  json {g}"))
print("== (3) premise: row alignment and rule transfer AUC (rule = StandardScaler + balanced logistic fit on all table-A rows of the first 6 batches of order seed 0, per-batch z; scored on the 6 post-switch batches of table B)")
a, b = tabs[J3], tabs[J7]; al = np.array_equal(a["pid"].values, b["pid"].values) and all(np.array_equal(a[c].fillna(0).values, b[c].fillna(0).values) for c in CJ[:6]); ok &= bool(al)
print(f"row aligned (pid + 6 judge-independent features identical): {al}")
perm = np.random.default_rng(10000).permutation(3000); pre, post = [perm[i * 250:(i + 1) * 250] for i in range(6)], [perm[i * 250:(i + 1) * 250] for i in range(6, 12)]
for dn, (d, ja, jb) in DIRS.items():
    Zp = [live(ja, b_) for b_ in pre]; clf = fit(np.vstack([z[0] for z in Zp]), np.concatenate([z[1] for z in Zp])); res_auc = {}
    for f in (1.0, -1.0, 0.0):
        Zq = [live(jb, b_, f) for b_ in post]; res_auc[f] = roc_auc_score(np.concatenate([z[1] for z in Zq]), clf.predict_proba(np.vstack([z[0] for z in Zq]))[:, 1])
    clf_b = fit(np.vstack([live(jb, b_)[0] for b_ in post]), np.concatenate([live(jb, b_)[1] for b_ in post])); Zq = [live(jb, b_) for b_ in post]
    ins = roc_auc_score(np.concatenate([z[1] for z in Zq]), clf_b.predict_proba(np.vstack([z[0] for z in Zq]))[:, 1])
    print(f"{dn}: transfer AUC A->B at f=+1 {res_auc[1.0]:.3f}, f=-1 {res_auc[-1.0]:.3f}, f=0 (judge_top, judge_mean z zeroed) {res_auc[0.0]:.3f}; in-sample B {ins:.3f}")
print("ALL CHECKS", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
