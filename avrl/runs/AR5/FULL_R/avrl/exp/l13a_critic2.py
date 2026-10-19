"""L13a critic 2 (CPU, run from avrl/). Usage: l13a_critic2.py > runs/L13a/critic2.log
Independent of l13a_summarize.py, l13a_critic.py and l13a_sim_alloc.py. Reads the raw sim_*.json per_seed lists, runs/L1a tables and runs/L12c.
A  every number in the ledger table recomputed from raw files (primary cell), compared with the ledger text values; split-level (honest unit) inference.
B  24-cell secondary claims (rung x scale x budget) recomputed.
C  structure checks: fit/sim disjoint by pid and by cell30, features list, n_fit / n_sim ranges, gate counts decomposed.
D  second literal per-pick reimplementation of the warm learner (own code, sklearn Pipeline built differently, no shared helper with the first critic) on cells NOT used by the first critic.
E  L12c cross-check of the Hypothesis paragraph numbers (greedy / free_400).
"""
import json, itertools, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

R = "runs/L13a"; TAGS = {"3B": "3B_q_corrupted", "7B": "7B_q_corrupted"}
CR = ["judge_logit", "log_tokens", "truncated", "has_decimal", "is_negative", "log_abs_ans", "empty_answer", "log_qlen"]
J = lambda d, tag, tgt, sp, sub: json.load(open(f"{d}/sim_{tag}_n{tgt}_sp{sp}_sub{sub}.json"))
bad = []


def chk(name, got, want, tol):
    ok = abs(got - want) <= tol
    if not ok: bad.append((name, got, want))
    print(f"   {'ok ' if ok else 'BAD'} {name}: computed {got:.4f} ledger {want} (tol {tol})")


def arr(ds, bud, pol): return np.array([d[f"budget_{bud}"][pol]["per_seed"] for d in ds], float)


def cluster_boot(a, b, seed=99, nb=20000):
    C = a.shape[0]; idx = np.random.default_rng(seed).integers(0, C, (nb, C)); r = a[idx].mean((1, 2)) / b[idx].mean((1, 2)); return np.percentile(r, [2.5, 97.5])


def split_stats(a, b):
    """honest unit: 4 splits. returns per-split ratio, exact 256-resample bootstrap CI of the pooled ratio over splits, t-interval of the ratio of the split-mean ratio, t, p."""
    C = a.shape[0]; sa = a.reshape(4, C // 4, -1).mean((1, 2)); sb = b.reshape(4, C // 4, -1).mean((1, 2)); spr = sa / sb
    rs = np.array([sa[list(c)].mean() / sb[list(c)].mean() for c in itertools.product(range(4), repeat=4)])
    d = sa - sb; t = d.mean() / (d.std(ddof=1) / 2); p = 2 * stats.t.sf(abs(t), 3)
    m, s = spr.mean(), spr.std(ddof=1) / 2; tc = stats.t.ppf(0.975, 3)
    loo = [(sa[[i for i in range(4) if i != j]].mean() / sb[[i for i in range(4) if i != j]].mean()) for j in range(4)]
    return spr, np.percentile(rs, [2.5, 97.5]), (m - tc * s, m + tc * s), t, p, loo


L12 = {}
print("== A: primary cell (25% budget, n_sim 2250, corrupted, 16 clusters x 8 seeds) recomputed from raw per_seed lists")
LED = {  # ledger values (second entry table), 3B / 7B
    "3B": dict(errs=637, uni=158.8, greedy=265.6, ts=267.9, f100=254.0, f200=265.1, f400=275.2, w100=275.3, w200=276.4, w400=280.6, static=280.1, oracle=562.0, ratio=1.042, lo=1.032, hi=1.055, t=4.68,
               w_f400=1.004, w_cold=1.040, w_static=0.987, gap=0.748, head_static=1.057, head_cold=1.002, ts_greedy=1.009, w100f100=1.084, w400f400=1.020, up=16, splits=[1.030, 1.069, 1.025, 1.047]),
    "7B": dict(errs=580, uni=144.4, greedy=247.9, ts=248.6, f100=234.7, f200=251.9, f400=256.1, w100=259.9, w200=262.2, w400=261.7, static=264.0, oracle=558.2, ratio=1.041, lo=1.030, hi=1.051, t=15.74,
               w_f400=1.024, w_cold=1.058, w_static=0.993, gap=0.851, head_static=1.048, head_cold=0.984, ts_greedy=1.003, w100f100=1.107, w400f400=1.022, up=15, splits=[1.038, 1.037, 1.042, 1.046]),
}
PRIM = {}
for rg, tag in TAGS.items():
    ds = [J(R, tag, 2250, sp, sub) for sp in range(4) for sub in range(4)]; L = LED[rg]; PRIM[rg] = ds
    g = lambda p: arr(ds, "0.25", p)
    w, f, f4, st, cold = g("warm_online_200"), g("recal_free_200"), g("recal_free_400"), g("static"), g("greedy_online")
    print(f"{rg}: n_sim per cluster {[d['n_sim'] for d in ds]}")
    chk(f"{rg} errors in sim (mean hits)", np.mean([d["hits"] for d in ds]), L["errs"], 0.6)
    chk(f"{rg} B", np.mean([int(0.25 * d["n_sim"]) for d in ds]), 562, 0.6)
    for k, p in (("uni", "uniform"), ("greedy", "greedy_online"), ("ts", "ts_online"), ("f100", "recal_free_100"), ("f200", "recal_free_200"), ("f400", "recal_free_400"), ("w100", "warm_online_100"), ("w200", "warm_online_200"),
                 ("w400", "warm_online_400"), ("static", "static"), ("oracle", "oracle")): chk(f"{rg} mean {p}", g(p).mean(), L[k], 0.05)
    lo, hi = cluster_boot(w, f); chk(f"{rg} ratio", w.mean() / f.mean(), L["ratio"], 0.0006); chk(f"{rg} CI lo (own seed)", lo, L["lo"], 0.0012); chk(f"{rg} CI hi (own seed)", hi, L["hi"], 0.0012)
    spr, sbci, stci, t, p, loo = split_stats(w, f)
    for i in range(4): chk(f"{rg} split {i} ratio", spr[i], L["splits"][i], 0.0006)
    chk(f"{rg} split t", t, L["t"], 0.006)
    print(f"   SPLIT-LEVEL (honest unit): pooled ratio {w.mean() / f.mean():.4f}; exact 256-resample split bootstrap 95% [{sbci[0]:.4f}, {sbci[1]:.4f}]; t-interval of mean split ratio [{stci[0]:.4f}, {stci[1]:.4f}]; "
          f"p {p:.4f}; leave-one-split-out pooled ratios {[round(x, 4) for x in loo]} (min {min(loo):.4f}); min split ratio {spr.min():.4f}")
    chk(f"{rg} warm/free400", w.mean() / f4.mean(), L["w_f400"], 0.0006); chk(f"{rg} warm/cold", w.mean() / cold.mean(), L["w_cold"], 0.0006); chk(f"{rg} warm/static", w.mean() / st.mean(), L["w_static"], 0.0006)
    chk(f"{rg} gap closed", (w.mean() - f.mean()) / (st.mean() - f.mean()), L["gap"], 0.0006); chk(f"{rg} static/free200", st.mean() / f.mean(), L["head_static"], 0.0006); chk(f"{rg} cold/free200", cold.mean() / f.mean(), L["head_cold"], 0.0006)
    chk(f"{rg} ts/greedy", g("ts_online").mean() / cold.mean(), L["ts_greedy"], 0.0006)
    chk(f"{rg} warm100/free100", g("warm_online_100").mean() / g("recal_free_100").mean(), L["w100f100"], 0.0006); chk(f"{rg} warm400/free400", g("warm_online_400").mean() / g("recal_free_400").mean(), L["w400f400"], 0.0006)
    d = (w - f).mean(1); chk(f"{rg} clusters warm>free", float((d > 0).sum()), L["up"], 0.1); print(f"   cluster diffs min {d.min():.2f} max {d.max():.2f}")
    dr = (w - f).ravel(); print(f"   128 paired runs: warm>free {np.mean(dr > 0):.3f} warm<free {np.mean(dr < 0):.3f} tie {np.mean(dr == 0):.3f} mean diff {dr.mean():.2f} sd {dr.std(ddof=1):.2f}")
    # seed-level: pooled over clusters, per seed ratio
    print(f"   per-seed pooled ratios {[round(float(w[:, s].mean() / f[:, s].mean()), 4) for s in range(8)]}")
    # ledger: threshold 1.03 equals this share of the f200-to-static gap
    print(f"   share of the gap that a ratio of exactly 1.03 closes: {0.03 * f.mean() / (st.mean() - f.mean()):.3f}")
    # sensitivity: does the verdict survive using the seed-clustered or split-clustered CI
    print(f"   verdict flags: ratio>=1.03 {w.mean() / f.mean() >= 1.03}; cluster-CI lo>1 {lo > 1}; splits>1 {int((spr > 1).sum())}/4; split-bootstrap lo>1 {sbci[0] > 1}")

print("\n== B: 24 cells (rung x scale x budget): warm200/free200")
rows = []
for rg, tag in TAGS.items():
    for tgt in (1125, 2250, 4500, 0):
        ds = [J(R, tag, tgt, sp, sub) for sp in range(4) for sub in (range(4) if tgt else [0])]
        for bud in ("0.1", "0.25", "0.5"):
            w, f, w1, f1, w4, f4, st, f400 = (arr(ds, bud, p) for p in ("warm_online_200", "recal_free_200", "warm_online_100", "recal_free_100", "warm_online_400", "recal_free_400", "static", "recal_free_400"))
            lo, hi = cluster_boot(w, f, nb=10000); spr, _, _, t, p, _ = split_stats(w, f); lo4 = cluster_boot(w4, f4, nb=10000)[0]
            rows.append(dict(rg=rg, tgt=tgt, bud=bud, ratio=w.mean() / f.mean(), lo=lo, hi=hi, minsp=spr.min(), p=p, r100=w1.mean() / f1.mean(), r400=w4.mean() / f4.mean(), lo400=lo4, ws=w.mean() / st.mean()))
T = pd.DataFrame(rows)
print(f"cells {len(T)}; ratio range {T.ratio.min():.3f} to {T.ratio.max():.3f}; CI lo > 1 in {(T.lo > 1).sum()} (lowest {T.lo.min():.3f}); all 4 splits > 1.00 in {(T.minsp > 1).sum()} (lowest split ratio {T.minsp.min():.4f}); "
      f"split p < .05 in {(T.p < .05).sum()}, < .10 in {(T.p < .10).sum()}")
print(f"n=100 ratio range {T.r100.min():.3f} to {T.r100.max():.3f}; n=400 range {T.r400.min():.3f} to {T.r400.max():.3f}, CI lo>1 in {(T.lo400 > 1).sum()} of 24")
print(f"warm200/static range {T.ws.min():.3f} to {T.ws.max():.3f}, > 1 in {(T.ws > 1).sum()} cells: {T[T.ws > 1][['rg', 'tgt', 'bud']].values.tolist()}")
for rg in TAGS:
    for bud in ("0.1", "0.25", "0.5"):
        r = T[(T.rg == rg) & (T.tgt == 2250) & (T.bud == bud)].iloc[0]; print(f"   {rg} n2250 {bud}: {r.ratio:.3f} [{r.lo:.3f}, {r.hi:.3f}]")
for rg, tag in TAGS.items():
    ds = [J(R, tag, 2250, sp, sub) for sp in range(4) for sub in range(4)]
    for bud in ("0.1", "0.5"):
        w, f4 = arr(ds, bud, "warm_online_200"), arr(ds, bud, "recal_free_400"); lo, hi = cluster_boot(w, f4, nb=10000); print(f"   {rg} {bud} warm200/free400 {w.mean() / f4.mean():.3f} [{lo:.3f}, {hi:.3f}]")

print("\n== C: structure")
for rg, tag in TAGS.items():
    t = pd.read_parquet(f"runs/L1a/tab_{tag}.parquet"); cnt = t.cell30.value_counts(); t = t[t.cell30.isin(cnt[cnt >= 3].index)].reset_index(drop=True)
    print(f"{rg}: table rows {len(t)} pids {t.pid.nunique()} base corrupted rate {t.corrupted.mean():.4f}; pid->cell30 unique max {t.groupby('pid').cell30.nunique().max()}")
    for sp in range(4):
        r0 = np.random.default_rng(sp); groups = t.cell30.unique(); r0.shuffle(groups); fg = set(groups[: len(groups) // 4]); fit = t[t.cell30.isin(fg)]; sim = t[~t.cell30.isin(fg)]
        d0 = J(R, tag, 0, sp, 0); print(f"   split {sp}: n_fit {len(fit)} (file {d0['n_fit']}), n_sim {len(sim)} (file {d0['n_sim']}), fit share {len(fit) / len(t):.3f}, pid overlap fit/sim {len(set(fit.pid) & set(sim.pid))}, fit pos rate {fit.corrupted.mean():.3f} sim {sim.corrupted.mean():.3f}")
    print(f"   features exclude gold-derived columns: {not ({'correct', 'accepted', 'corrupted', 'judge_wrong', 'gold'} & set(CR))}; features = {CR}")
nfit = [J(R, tag, 2250, sp, 0)["n_fit"] for tag in TAGS.values() for sp in range(4)]; nbef = [J(R, tag, 2250, sp, 0)["n_sim_before_subsample"] for tag in TAGS.values() for sp in range(4)]
print(f"n_fit range {min(nfit)}..{max(nfit)}; n_sim_before_subsample range {min(nbef)}..{max(nbef)}")
for k, d in (("gate_check", "gate/gate_check.json"), ("ident", "gate/gate_check_ident.json"), ("post", "gate/gate_check_post.json")): print(k, json.load(open(f"{R}/{d}")))
# decompose the "cells" in gate 1: 5 meta keys per file + policy x budget cells
n_meta = 5 * 4
g1 = sum(len(a) for tag in TAGS.values() for tgt in (2250, 0) for a in [list(J("runs/L12c", tag, tgt, 0, 0)["budget_0.1"]) + list(J("runs/L12c", tag, tgt, 0, 0)["budget_0.25"]) + list(J("runs/L12c", tag, tgt, 0, 0)["budget_0.5"])])
print(f"gate 1: 254 reported = {n_meta} metadata comparisons + {g1} policy-budget cells -> {n_meta + g1}")
g3 = sum(len(J("runs/L12c", tag, tgt, sp, sub)[f"budget_{b}"]) for tag in TAGS.values() for tgt in (1125, 2250, 4500, 0) for sp in range(4) for sub in (range(4) if tgt else [0]) for b in ("0.1", "0.25", "0.5"))
print(f"post gate: 6440 reported = {5 * 104} metadata comparisons + {g3} policy-budget cells -> {5 * 104 + g3}")
# budget accounting: every arm has per_seed <= B, oracle == min(B, errors), and B identical by construction (B = int(frac * n_sim)); hits for free/warm arms
mx = 0
for tag in TAGS.values():
    for tgt in (1125, 2250, 4500, 0):
        for sp in range(4):
            for sub in (range(4) if tgt else [0]):
                d = J(R, tag, tgt, sp, sub)
                for bud in ("0.1", "0.25", "0.5"):
                    B = int(float(bud) * d["n_sim"]); o = d[f"budget_{bud}"]["oracle"]["per_seed"]
                    assert all(v == min(B, d["hits"]) for v in o), (tag, tgt, sp, sub, bud, o)
                    for p, v in d[f"budget_{bud}"].items():
                        if p.startswith("warm_online") or p.startswith("recal_free"): mx = max(mx, max(v["per_seed"]) / B)
print(f"oracle == min(B, errors) in all 312 cells (so B picks are charged for the best arm); max warm/free hits / B over all files {mx:.3f}")

print("\n== D: second literal per-pick reimplementation (own code) on cells not used by the first critic")


def fit_lr(X, y, it):
    sc = StandardScaler().fit(X); m = LogisticRegression(max_iter=it, class_weight="balanced").fit(sc.transform(X), y); return lambda Z: m.predict_proba(sc.transform(Z))[:, 1]


def pop(tag, sp, tgt, sub):
    t = pd.read_parquet(f"runs/L1a/tab_{tag}.parquet"); cnt = t.cell30.value_counts(); t = t[t.cell30.isin(cnt[cnt >= 3].index)].reset_index(drop=True)
    g = t.cell30.unique(); np.random.default_rng(sp).shuffle(g); fg = set(g[: len(g) // 4]); fit = t[t.cell30.isin(fg)]; sim = t[~t.cell30.isin(fg)].reset_index(drop=True)
    pids = sim.pid.unique(); np.random.default_rng([sub, sp, 11]).shuffle(pids); cum = np.cumsum(sim.groupby("pid").size().reindex(pids).values); keep = set(pids[: int(np.argmin(np.abs(cum - tgt))) + 1])
    sim = sim[sim.pid.isin(keep)].reset_index(drop=True)
    return fit[CR].fillna(0).values, fit.corrupted.values.astype(int), sim[CR].fillna(0).values, sim.corrupted.values.astype(int)


def warm_literal(Xf, yf, Xs, y, B, seed, n):
    R_ = max(B // 8, 40); cal = np.random.default_rng([seed, 12]).permutation(len(yf))[:n]; Xc, yc = Xf[cal], yf[cal]
    s = fit_lr(Xc, yc, 1000)(Xs); picked = []; seenmask = np.zeros(len(y), bool); nxt = R_
    for step in range(1, B + 1):
        cand = np.flatnonzero(~seenmask); j = cand[np.argmax(s[cand])]
        seenmask[j] = True; picked.append(j)
        if step >= nxt and step < B:
            Xp, yp = np.vstack([Xc, Xs[seenmask]]), np.concatenate([yc, y[seenmask]])
            if 0 < yp.sum() < len(yp): s = fit_lr(Xp, yp, 500)(Xs); nxt += R_
    return int(y[picked].sum())


for tag_name, sp, sub in (("3B", 3, 3), ("7B", 1, 2), ("3B", 2, 1), ("7B", 3, 0)):
    tag = TAGS[tag_name]; Xf, yf, Xs, y = pop(tag, sp, 2250, sub); d = J(R, tag, 2250, sp, sub); assert len(y) == d["n_sim"] and len(yf) == d["n_fit"]
    for bud in ("0.1", "0.25", "0.5"):
        B = int(float(bud) * len(y)); mine = [warm_literal(Xf, yf, Xs, y, B, s, 200) for s in range(8)]; raw = d[f"budget_{bud}"]["warm_online_200"]["per_seed"]
        print(f"{tag_name} sp{sp} sub{sub} {bud} B {B}: literal {mine} raw {raw} equal {mine == raw} max abs diff {max(abs(a - b) for a, b in zip(mine, raw))}")
        if mine != raw: bad.append(("literal", tag_name, sp, sub, bud))

print("\n== E: Hypothesis paragraph numbers from runs/L12c")
for rg, tag in TAGS.items():
    ds = [J("runs/L12c", tag, 2250, sp, sub) for sp in range(4) for sub in range(4)]
    g, f4, f2, st = (arr(ds, "0.25", p) for p in ("greedy_online", "recal_free_400", "recal_free_200", "static"))
    print(f"{rg}: greedy/free_400 {g.mean() / f4.mean():.4f} (ledger hypothesis: 0.965 on 3B, 0.976 on 7B-AWQ); greedy/free_200 {g.mean() / f2.mean():.4f}; static/free_200 {st.mean() / f2.mean():.4f}")
print("\nSUMMARY: bad items", bad)

print("\n== F: independent re-run of the label-count control (fixed rule given m free labels at once; m = 400, 600, 200 + B), 16 clusters x 8 seeds, 25 percent budget")
for rg, tag in TAGS.items():
    ds = [J(R, tag, 2250, sp, sub) for sp in range(4) for sub in range(4)]; w = arr(ds, "0.25", "warm_online_200"); f4raw = arr(ds, "0.25", "recal_free_400"); out = {m: np.zeros((16, 8)) for m in (400, 600, "762")}
    for sp in range(4):
        for sub in range(4):
            Xf, yf, Xs, y = pop(tag, sp, 2250, sub); B = int(0.25 * len(y)); c = sp * 4 + sub
            for s in range(8):
                perm = np.random.default_rng([s, 12]).permutation(len(yf))
                for m in out:
                    k = B + 200 if m == "762" else m; sc = fit_lr(Xf[perm[:k]], yf[perm[:k]], 1000)(Xs); out[m][c, s] = y[np.argsort(-sc)[:B]].sum()
    print(f"{rg}: free_400 reproduces recal_free_400 per seed: {bool((out[400] == f4raw).all())}")
    for m, a in out.items():
        lo, hi = cluster_boot(w, a, nb=20000); spr, sbci, stci, t, p, _ = split_stats(w, a)
        print(f"   warm200 {w.mean():.1f} / free_{m} {a.mean():.1f} = {w.mean() / a.mean():.4f} [{lo:.4f}, {hi:.4f}]; split ratios {[round(float(x), 3) for x in spr]}; split-level t-interval [{stci[0]:.4f}, {stci[1]:.4f}]")
