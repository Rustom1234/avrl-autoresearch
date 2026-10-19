"""L9c critic check (CPU, .venv python, read-only on runs/, prints to stdout). Independent recomputation of the L9c ledger numbers from pool/*/feats_J.parquet, rollouts summary.json and the L8c-A boot JSONs.
Usage: /homes/rdubash/active-audit/.venv/bin/python -I exp/l9c_critic_check.py"""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
RUNS = "/homes/rdubash/avrl_live/FULL/avrl/runs/"
def curve(F, sc_f, sgn, frac):
    ac = F.accept.values.astype(bool); y = F.judge_error.values; ag = F.agreement.values; nd = F.n_distinct.values; blocks = {}
    for i in np.flatnonzero(ac): blocks.setdefault((round(ag[i], 6), -nd[i]), []).append(i)
    left = int(frac * 2000); exp = 0.0; tb = 0
    for key in sorted(blocks):
        m = np.array(blocks[key]); s = min(left, len(m)); exp += s * y[m].mean(); order = np.argsort(-(sgn * F[sc_f].values[m]), kind="stable"); tb += y[m][order[:s]].sum(); left -= s
        if left <= 0: break
    return exp, tb
L = {}
for tag, root in (("L9c", "L9c-weak-policy-tiebreak"), ("L8cA", "L8c-A-fresh-tiebreak")):
    for k, sc, sg in (("q3_4b", "s_diff", -1.0), ("rm", "s_oth", 1.0)):
        F = pd.read_parquet(f"{RUNS}{root}/pool/{k}/feats_J.parquet").sort_values("pid").reset_index(drop=True); ac = F.accept.values.astype(bool); y = F.judge_error.values
        print(f"== {tag} {k}: accepted {ac.sum()}, J {y.sum()} (recomputed {(ac & (F.majority_correct.values == 0)).sum()}), mv-wrong {(F.majority_correct == 0).sum()}, rejected-wrong share {(1 - y.sum() / (F.majority_correct == 0).sum()):.3f}")
        out = {fr: curve(F, sc, sg, fr) for fr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)}
        r7 = [t / e for e, t in out.values()]; r6 = [out[f][1] / out[f][0] for f in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)]
        print("   closed-form tie-break/agree_acc by budget", {f: round(t / e, 4) for f, (e, t) in out.items()}, f"| 7-budget mean {np.mean(r7):.4f}, 6-budget (10..40) mean log {np.mean(np.log(r6)):.4f}")
        if tag == "L9c":
            ag, nd = F.agreement.values, F.n_distinct.values; idx = np.flatnonzero(ac); o = idx[np.lexsort((-nd[idx], ag[idx]))]; last = o[499]
            cell = ac & np.isclose(ag, ag[last]) & (nd == nd[last]); before = ac & ((ag < ag[last] - 1e-9) | (np.isclose(ag, ag[last]) & (nd > nd[last]))); e, t = out[0.25]
            s_ = sg * F[sc].values[cell]; rng = np.random.default_rng(0); n = int(cell.sum()); bs = []
            for _ in range(2000):
                i = rng.integers(0, n, n)
                if 0 < y[cell][i].sum() < n: bs.append(roc_auc_score(y[cell][i], s_[i]))
            print(f"   operative cell ({ag[last]}, {nd[last]}): {n} groups, {y[cell].sum()} errors, {before.sum()} before ({y[before].sum()} errors), {500 - before.sum()} slots; AUC {roc_auc_score(y[cell], s_):.4f} CI {np.percentile(bs, [2.5, 97.5]).round(4)}; tie-break {t}, exact agree_acc {e:.3f}, ratio {t / e:.4f}, perfect-order ceiling {(y[before].sum() + min(y[cell].sum(), 500 - before.sum())) / e:.4f}")
            A = F[F.accept == 1]; g = A.groupby([A.agreement.round(6), "n_distinct"]).judge_error.agg(["size", "sum"]); gt = A.groupby(A.agreement.round(6)).judge_error.agg(["size", "sum"])
            print(f"   largest (agreement, n_distinct) block by judge errors: {g['sum'].idxmax()} with {g['sum'].max()} of {g.loc[g['sum'].idxmax(), 'size']} groups; largest whole agreement tier: {gt['sum'].idxmax()} with {gt['sum'].max()}")
    for k in ("q3_4b", "rm"):
        for f in ("0.25",):
            try: fp = json.load(open(f"{RUNS}{root}/boot/{k}/boot_tb_f{f}.json"))["full_pool"]
            except Exception as ex: print("   no boot", root, k, ex); continue
            r = lambda a: float(np.mean(fp[a])); nat = fp["best_fixed7_natural"]
            print(f"   {tag} {k} {f}: oracle {r('oracle')}, natural best fixed {nat} {r(nat):.2f}, agree_acc {r('agree_acc'):.2f}, H25 = (oracle - best fixed)/oracle {(r('oracle') - r(nat)) / r('oracle'):.4f}, (oracle - agree_acc)/oracle {(r('oracle') - r('agree_acc')) / r('oracle'):.4f}, blend7/agree_acc {r('greedy_blend7') / r('agree_acc'):.4f}")
