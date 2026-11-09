"""L13a critic recheck (CPU, run from avrl/). Independent of l13a_summarize.py and l13a_sim_alloc.py's warm code. Usage: l13a_critic.py > runs/L13a/critic.log
A  primary cell (25% budget, n_sim 2250, corrupted, n = 200) recomputed from the raw per_seed lists of the 16 sim files per rung with own bootstrap (seed 7, 20000 resamples), own per-split ratios, scipy ttest_rel over the
   4 split means; compared with result.json.
B  invariants over all 104 sim files: 8 finite per_seed values per arm, 0 <= hits <= min(B, errors in sim), warm_refit true, no one-class free draws, warm arms present at all budgets.
C  literal per-pick reimplementation of the warm learner as run_online(greedy=True) with a warm initial score (per pick: argsort over the unseen subset) for the first file of each rung (n2250 sp0 sub0), 3 budgets,
   n = 200, seeds 0..7, compared with the raw per_seed lists (the run uses a block order from one full-population argsort between refits, which can differ only in tie order).
D  side check, NOT pre-registered: label-count control. A fixed rule (recal_free logic, same fit-half draw) given m free labels, m in 400, 600, 200 + B, over the 16 clusters of the primary cell, against warm_online_200
   (which holds 200 + (labels seen so far) labels, 200 + B at the end) and against static. Separates 'more labels' from 'refitting on own stream' only as far as a fixed rule at the final label count can."""
import json, sys, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CR = "judge_logit,log_tokens,truncated,has_decimal,is_negative,log_abs_ans,empty_answer,log_qlen".split(","); LABEL, G, MIN = "corrupted", "cell30", 3
TAGS = {"3B": "3B_q_corrupted", "7B-AWQ": "7B_q_corrupted"}
res = json.load(open("runs/L13a/result.json")); rng_b = np.random.default_rng(7); NB = 20000
J = lambda tag, tgt, sp, sub: json.load(open(f"runs/L13a/sim_{tag}_n{tgt}_sp{sp}_sub{sub}.json"))


def boot(a, b, C):
    idx = rng_b.integers(0, C, (NB, C)); r = a[idx].mean((1, 2)) / b[idx].mean((1, 2)); return np.percentile(r, [2.5, 97.5])


def per_seed(ds, bud, pol): return np.array([d[f"budget_{bud}"][pol]["per_seed"] for d in ds], float)


print("== A: primary cell recomputed from raw JSON")
for rg, tag in TAGS.items():
    ds = [J(tag, 2250, sp, sub) for sp in range(4) for sub in range(4)]; w, f, f4, st, g = (per_seed(ds, "0.25", p) for p in ("warm_online_200", "recal_free_200", "recal_free_400", "static", "greedy_online"))
    lo, hi = boot(w, f, 16); spr = [w[4 * s:4 * s + 4].mean() / f[4 * s:4 * s + 4].mean() for s in range(4)]
    tt = stats.ttest_rel([w[4 * s:4 * s + 4].mean() for s in range(4)], [f[4 * s:4 * s + 4].mean() for s in range(4)])
    R = res["verdict"]["rungs"][rg]
    print(f"{rg}: warm200 {w.mean():.3f} free200 {f.mean():.3f} ratio {w.mean() / f.mean():.4f} [{lo:.4f}, {hi:.4f}] splits {[round(x, 4) for x in spr]} t(df3) {tt.statistic:.3f} p {tt.pvalue:.4f} | "
          f"warm/free400 {w.mean() / f4.mean():.4f} warm/cold {w.mean() / g.mean():.4f} warm/static {w.mean() / st.mean():.4f} gap closed {(w.mean() - f.mean()) / (st.mean() - f.mean()):.4f}")
    print(f"   result.json: ratio {R['warm200_over_free200']:.4f} [{R['ci'][0]:.4f}, {R['ci'][1]:.4f}] t {R['split_t_df3']:.3f} p {R['split_p_df3']:.4f} gap {R['share_gap_closed_200']:.4f} -> "
          f"max abs diff ratio {abs(w.mean() / f.mean() - R['warm200_over_free200']):.2e}, t {abs(tt.statistic - R['split_t_df3']):.2e}, splits {max(abs(a - b) for a, b in zip(spr, R['per_split_ratio'])):.2e}")
    d = (w - f).mean(1); print(f"   paired cluster differences (hits): mean {d.mean():.2f} min {d.min():.2f} max {d.max():.2f}, clusters with warm > free {int((d > 0).sum())} of 16; seeds (mean over clusters) with warm > free {int(((w - f).mean(0) > 0).sum())} of 8")
    print(f"   verdict flags in result.json: pass {R['pass']} kill {R['kill']}; outcome {res['verdict']['outcome']}")

print("\n== B: invariants over all 104 files")
nf = bad = 0
for rg, tag in TAGS.items():
    for tgt in (1125, 2250, 4500, 0):
        for sp in range(4):
            for sub in (range(4) if tgt else [0]):
                d = J(tag, tgt, sp, sub); nf += 1; L12 = json.load(open(f"runs/L12c/sim_{tag}_n{tgt}_sp{sp}_sub{sub}.json"))
                if d["warm_refit"] is not True or d["warm_oneclass_draws"] or d["warm_ns"] != [100, 200, 400]: bad += 1; print("BAD meta", tag, tgt, sp, sub)
                for bud in ("0.1", "0.25", "0.5"):
                    B = int(float(bud) * d["n_sim"]); row = d[f"budget_{bud}"]
                    for p, v in row.items():
                        x = np.array(v["per_seed"], float)
                        if len(x) != 8 or not np.isfinite(x).all() or x.min() < 0 or x.max() > min(B, d["hits"]): bad += 1; print("BAD range", tag, tgt, sp, sub, bud, p, x)
                    for n in (100, 200, 400):
                        if f"warm_online_{n}" not in row: bad += 1; print("BAD missing", tag, tgt, sp, sub, bud, n)
                    if any(row[p]["per_seed"] != L12[f"budget_{bud}"][p]["per_seed"] for p in L12[f"budget_{bud}"]): bad += 1; print("BAD L12c arm moved", tag, tgt, sp, sub, bud)
print(f"files {nf}, problems {bad}")


def build(tag, sp, tgt, sub):
    """Same population construction as l13a_sim_alloc.py (rollout mode)."""
    df = pd.read_parquet(f"runs/L1a/tab_{tag}.parquet"); cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
    r0 = np.random.default_rng(sp); groups = df[G].unique(); r0.shuffle(groups); fit_g = set(groups[: len(groups) // 4]); fit = df[df[G].isin(fit_g)]; sim = df[~df[G].isin(fit_g)].reset_index(drop=True)
    pids = sim["pid"].unique(); np.random.default_rng([sub, sp, 11]).shuffle(pids)
    cum = np.cumsum(sim.groupby("pid").size().reindex(pids).values); keep = pids[: int(np.argmin(np.abs(cum - tgt))) + 1]; sim = sim[sim["pid"].isin(set(keep))].reset_index(drop=True)
    Xf, Xs = fit[CR].fillna(0).values, sim[CR].fillna(0).values
    return Xf, fit[LABEL].values.astype(int), Xs, sim[LABEL].values.astype(int)


LR = lambda it: make_pipeline(StandardScaler(), LogisticRegression(max_iter=it, class_weight="balanced"))


def warm_literal(Xf, yf, Xs, y, budget, seed, n):
    """run_online(greedy=True) structure: per pick, argsort over the unseen subset; warm initial score; refit every R labels on the n free + seen sim labels."""
    R = max(budget // 8, 40); cal = np.random.default_rng([seed, 12]).permutation(len(yf))[:n]; Xc, yc = Xf[cal], yf[cal]
    score = LR(1000).fit(Xc, yc).predict_proba(Xs)[:, 1]; seen = np.zeros(len(y), bool); hits = spent = 0; next_fit = R
    while spent < budget:
        mu = np.where(~seen)[0]; c = mu[np.argsort(-score[mu])][:1]; seen[c] = True; hits += int(y[c].sum()); spent += 1
        if spent >= next_fit and spent < budget:
            yp = np.concatenate([yc, y[seen]])
            if 0 < yp.sum() < len(yp): score = LR(500).fit(np.vstack([Xc, Xs[seen]]), yp).predict_proba(Xs)[:, 1]; next_fit += R
    return hits


print("\n== C: literal per-pick reimplementation vs raw per_seed (n2250 sp0 sub0, n = 200)")
for rg, tag in TAGS.items():
    Xf, yf, Xs, y = build(tag, 0, 2250, 0); d = J(tag, 2250, 0, 0); assert len(y) == d["n_sim"]
    for bud in ("0.1", "0.25", "0.5"):
        B = int(float(bud) * len(y)); mine = [warm_literal(Xf, yf, Xs, y, B, s, 200) for s in range(8)]; raw = d[f"budget_{bud}"]["warm_online_200"]["per_seed"]
        print(f"{rg} {bud} B {B}: literal {mine} raw {raw} equal {mine == raw} max abs diff {max(abs(a - b) for a, b in zip(mine, raw))}")

print("\n== D: label-count control (side check, not pre-registered), primary cell, 16 clusters x 8 seeds")
for rg, tag in TAGS.items():
    ds = [J(tag, 2250, sp, sub) for sp in range(4) for sub in range(4)]; w = per_seed(ds, "0.25", "warm_online_200"); st = per_seed(ds, "0.25", "static"); f200 = per_seed(ds, "0.25", "recal_free_200"); f400r = per_seed(ds, "0.25", "recal_free_400")
    out = {m: [] for m in (400, 600, "200+B")}; n_fit = []
    for sp in range(4):
        for sub in range(4):
            Xf, yf, Xs, y = build(tag, sp, 2250, sub); B = int(0.25 * len(y)); n_fit.append(len(yf)); rows = {m: [] for m in out}
            for s in range(8):
                perm = np.random.default_rng([s, 12]).permutation(len(yf))
                for m in out:
                    k = B + 200 if m == "200+B" else m; cal = perm[:k]
                    order = np.argsort(-LR(1000).fit(Xf[cal], yf[cal]).predict_proba(Xs)[:, 1]); rows[m].append(int(y[order[:B]].sum()))
            for m in out: out[m].append(rows[m])
    out = {m: np.array(v, float) for m, v in out.items()}
    print(f"{rg}: reproduces recal_free_400 per seed: {bool((out[400] == f400r).all())}; n_fit min {min(n_fit)}")
    for m, a in out.items():
        lo, hi = boot(w, a, 16); print(f"   free_{m}: {a.mean():.1f} | warm200 {w.mean():.1f} -> warm200 / free_{m} {w.mean() / a.mean():.4f} [{lo:.4f}, {hi:.4f}], free_{m} / static {a.mean() / st.mean():.4f}")
    print(f"   free_200 {f200.mean():.1f}, static {st.mean():.1f}, warm200 {w.mean():.1f}")
