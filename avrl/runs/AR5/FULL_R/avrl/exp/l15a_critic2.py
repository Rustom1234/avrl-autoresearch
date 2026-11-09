"""L15a critic 2 (CPU, run from avrl/). Independent of exp/l15a_critic.py and exp/l15a_summarize.py. Usage: l15a_critic2.py <part>
part A: recompute every primary / descriptive number of the ledger from the raw sim_*.json with own bootstrap (seed 11, 20000 resamples) and exact split bootstrap; parse summary.md main table and compare all 72 rows x 5 ratios.
part B: leakage + budget checks (pid overlap fit/sim, feature list, per-file n_sim and B, hits invariants, flip realised rate).
part C: literal per-pick reimplementation (own code, argsort recomputed on every pick) of warm_stale, w25, recal_free_stale and cold greedy for the flip25 / 25% cell, all 16 clusters x 8 seeds per rung, compared with raw per_seed lists.
part D: side claims (cross-judge label agreement, ts/greedy of L13a, strict-reading verdict under the honest unit)."""
import json, sys, re, itertools
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

part = sys.argv[1]; RD = "runs/L15a"; BUDS = ("0.1", "0.25", "0.5"); TAGS = {"3B": "3B_q_corrupted", "7B-AWQ": "7B_q_corrupted"}
CHEAP = "judge_logit,log_tokens,truncated,has_decimal,is_negative,log_abs_ans,empty_answer,log_qlen".split(",")
jl = lambda p: json.load(open(p))
SPL = np.array(list(itertools.product(range(4), repeat=4)))


def grid(tag, st, root=RD):
    ds = [jl(f"{root}/sim_{tag}_{st}_n2250_sp{sp}_sub{sub}.json") if root == RD else jl(f"{root}/sim_{tag}_n2250_sp{sp}_sub{sub}.json") for sp in range(4) for sub in range(4)]
    return ds, {b: {p: np.array([d[f"budget_{b}"][p]["per_seed"] for d in ds], float) for p in ds[0][f"budget_{b}"]} for b in BUDS}


def cmpr(a, b, rng, nb=20000):
    idx = rng.integers(0, 16, (nb, 16)); r = a[idx].mean((1, 2)) / b[idx].mean((1, 2)); sa, sb = a.reshape(4, 4, 8).mean((1, 2)), b.reshape(4, 4, 8).mean((1, 2)); spr = sa / sb
    t = stats.ttest_rel(sa, sb); eb = np.percentile(sa[SPL].mean(1) / sb[SPL].mean(1), [2.5, 97.5]); se = spr.std(ddof=1) / 2; T3 = stats.t.ppf(0.975, 3)
    return dict(ratio=a.mean() / b.mean(), lo=np.percentile(r, 2.5), hi=np.percentile(r, 97.5), spr=spr, t=t.statistic, p=t.pvalue, eb=eb, tci=(spr.mean() - T3 * se, spr.mean() + T3 * se),
                cl_up=int((a.mean(1) > b.mean(1)).sum()), seed_up=int((a.mean(0) > b.mean(0)).sum()), da=a.mean(), db=b.mean())


if part == "A":
    rng = np.random.default_rng(11); R = {}
    for rg, tag in TAGS.items():
        _, clean = grid(tag, None, "runs/L13a")
        for st in ("flip25", "flip50", "lowlen", "none"):
            ds, H = grid(tag, st)
            for b in BUDS:
                h = H[b]; w, f, w25, g = h["warm_online_200"], h["recal_free_200"], h["warm_online_200_w25"], h["greedy_online"]; c = clean[b]["warm_online_200"]
                R[(rg, st, b)] = dict(wc=cmpr(w, g, rng), wf=cmpr(w, f, rng), wcl=cmpr(w, c, rng), xc=cmpr(w25, g, rng), xw=cmpr(w25, w, rng), fc=f.mean() / g.mean(),
                                      means=dict(u=h["uniform"].mean(), s=h["static"].mean(), g=g.mean(), f=f.mean(), w=w.mean(), x=w25.mean(), c=c.mean()))
    for rg in TAGS:
        r = R[(rg, "flip25", "0.25")]; print(f"== {rg} primary flip25 25%: means", {k: round(v, 1) for k, v in r["means"].items()})
        for k in ("wc", "wf", "wcl", "xc", "xw"):
            d = r[k]; print(f"  {k}: {d['ratio']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}] per-split {np.round(d['spr'], 3).tolist()} split t {d['t']:.2f} p {d['p']:.3f} t-int [{d['tci'][0]:.3f}, {d['tci'][1]:.3f}] exact split boot [{d['eb'][0]:.3f}, {d['eb'][1]:.3f}] clusters up {d['cl_up']}/16 seeds up {d['seed_up']}/8")
        print(f"  free_stale/cold {r['fc']:.4f}; warm/static {r['means']['w'] / r['means']['s']:.4f}; static/warm {r['means']['s'] / r['means']['w']:.4f}; warm_stale/uniform {r['means']['w'] / r['means']['u']:.3f}")
        print(f"  share of clean edge kept: {(r['means']['w'] - r['means']['g']) / (r['means']['c'] - r['means']['g']):.3f}; stale share of last refit set = {200 / (200 + 562):.3f}")
    # table parse
    rows = [l for l in open(f"{RD}/summary.md").read().split("\n") if re.match(r"^\| (3B|7B-AWQ) \| (flip25|flip50|lowlen|none) \| (10|25|50)% \| \d+ \| [\d.]+ \|", l)]
    rows = [l for l in rows if len([x for x in l.strip("|").split("|")]) == 15]; print("\nmain-table rows parsed:", len(rows)); worst = 0; bad = 0
    for l in rows:
        c = [x.strip() for x in l.strip("|").split("|")]; rg, st, bp = c[0], c[1], c[2]; b = {"10%": "0.1", "25%": "0.25", "50%": "0.5"}[bp]; r = R[(rg, st, b)]
        num = lambda s: [float(v) for v in re.findall(r"-?\d+\.\d+", s)]
        for col, key in ((c[9], "wc"), (c[10], "wf"), (c[11], "wcl"), (c[12], "w25c" if False else "xc"), (c[13], "xw")):
            v = num(col); d = r[key]; dd = max(abs(v[0] - d["ratio"]), abs(v[1] - d["lo"]), abs(v[2] - d["hi"])); worst = max(worst, dd)
            if abs(v[0] - d["ratio"]) > 0.0006 or dd > 0.006: bad += 1; print("MISMATCH", rg, st, b, key, v, round(d["ratio"], 4), round(d["lo"], 4), round(d["hi"], 4))
        mm = r["means"]; ab = [float(c[i]) for i in (4, 5, 6, 7, 8)]
        for x, y_ in zip(ab, (mm["w"], mm["g"], mm["f"], mm["c"], mm["x"])):
            if abs(x - y_) > 0.051: bad += 1; print("ABS MISMATCH", rg, st, b, x, y_)
        if abs(float(c[14]) - r["fc"]) > 0.0006: bad += 1; print("FREE/COLD MISMATCH", rg, st, b)
    print(f"main-table: {bad} mismatches over {len(rows)} rows x 5 ratios + 5 means + free/cold, max |ratio/CI diff| {worst:.4f}")
    # result.json verdict block
    res = jl(f"{RD}/result.json")
    for rg in TAGS:
        v = res["verdict"]["rungs"][rg]; r = R[(rg, "flip25", "0.25")]
        print(f"result.json {rg}: warm/cold {v['warm_stale_over_cold']:.4f} mine {r['wc']['ratio']:.4f}; warm/free {v['warm_stale_over_free_stale']:.4f} mine {r['wf']['ratio']:.4f}; pass {v['pass']} kill {v['kill']} B {v['B']}")
    print("result.json outcome:", res["verdict"]["outcome"], "| gate_ok", res["verdict"]["gate_ok"])
    lit = all(R[(rg, "flip25", "0.25")]["wc"]["ratio"] >= 1.0 and R[(rg, "flip25", "0.25")]["wf"]["ratio"] >= 1.06 and R[(rg, "flip25", "0.25")]["wf"]["lo"] > 1 for rg in TAGS)
    lit_strict = all(lit and R[(rg, "flip25", "0.25")]["wc"]["lo"] > 1 for rg in TAGS)
    kill = any(R[(rg, "flip25", "0.25")]["wc"]["ratio"] <= 0.97 and R[(rg, "flip25", "0.25")]["wc"]["hi"] < 1 for rg in TAGS)
    print(f"literal PASS (spec text, CI lower > 1 on the free ratio): {lit}; PASS with CI lower > 1 also on the cold ratio: {lit_strict}; KILL: {kill}")
    for rg in TAGS:
        d = R[(rg, "flip25", "0.25")]["wc"]; print(f"  honest-unit view {rg}: warm/cold exact split boot lower {d['eb'][0]:.4f} (> 1: {d['eb'][0] > 1}), t-interval lower {d['tci'][0]:.4f}")
        d = R[(rg, "flip25", "0.25")]["wf"]; print(f"  honest-unit view {rg}: warm/free exact split boot lower {d['eb'][0]:.4f}, t-interval lower {d['tci'][0]:.4f}, (>= 1.06 on the t interval lower: {d['tci'][0] >= 1.06})")
    print("\ndescriptive lines quoted in the entries:")
    for st in ("flip50", "lowlen"):
        for rg in TAGS:
            r = R[(rg, st, "0.25")]; print(f"  {rg} {st} 25%: warm/cold {r['wc']['ratio']:.3f} [{r['wc']['lo']:.3f}, {r['wc']['hi']:.3f}] t-int [{r['wc']['tci'][0]:.3f}, {r['wc']['tci'][1]:.3f}], warm/free {r['wf']['ratio']:.3f}, free/cold {r['fc']:.3f}, w25/cold {r['xc']['ratio']:.3f} [{r['xc']['lo']:.3f}, {r['xc']['hi']:.3f}], w25/warm {r['xw']['ratio']:.3f}, warm/warmclean {r['wcl']['ratio']:.3f}")
    for rg in TAGS:
        for b in ("0.1", "0.5"):
            r = R[(rg, "flip25", b)]; print(f"  {rg} flip25 {b}: warm/cold {r['wc']['ratio']:.3f} [{r['wc']['lo']:.3f}, {r['wc']['hi']:.3f}] warm/free {r['wf']['ratio']:.3f}")
        r = R[(rg, "none", "0.25")]; print(f"  {rg} none 25%: warm/cold {r['wc']['ratio']:.3f} [{r['wc']['lo']:.3f}, {r['wc']['hi']:.3f}] warm/free {r['wf']['ratio']:.3f}; w25/warm {r['xw']['ratio']:.3f} [{r['xw']['lo']:.3f}, {r['xw']['hi']:.3f}]")

if part == "B":
    bad = 0
    for rg, tag in TAGS.items():
        df = pd.read_parquet(f"runs/L1a/tab_{tag}.parquet"); cnt = df["cell30"].value_counts(); df = df[df["cell30"].isin(cnt[cnt >= 3].index)].reset_index(drop=True)
        assert df.groupby("pid")["cell30"].nunique().max() == 1
        print(f"{rg}: rows {len(df)} pids {df.pid.nunique()} corrupted rate {df.corrupted.mean():.4f}; features used {CHEAP}; label col 'corrupted' not in features: {'corrupted' not in CHEAP}")
        for f in CHEAP:
            if df[f].nunique() > 1:
                from sklearn.metrics import roc_auc_score
                print(f"   feature {f:14s} univariate AUC vs label {roc_auc_score(df.corrupted, df[f].fillna(0)):.3f}")
        for sp in range(4):
            rng0 = np.random.default_rng(sp); groups = df["cell30"].unique(); rng0.shuffle(groups); fit_g = set(groups[: len(groups) // 4])
            fit = df[df["cell30"].isin(fit_g)]; sim = df[~df["cell30"].isin(fit_g)]
            ov = len(set(fit.pid) & set(sim.pid)); d0 = jl(f"{RD}/sim_{tag}_flip25_n2250_sp{sp}_sub0.json")
            print(f"   split {sp}: fit pids {fit.pid.nunique()} sim pids {sim.pid.nunique()} overlap {ov}; n_fit {len(fit)} (file {d0['n_fit']}) n_fit_pos {int(fit.corrupted.sum())} (file {d0['n_fit_pos']})")
            if ov or len(fit) != d0["n_fit"]: bad += 1
        for st in ("flip25", "flip50", "lowlen", "none"):
            for sp in range(4):
                for sub in range(4):
                    d = jl(f"{RD}/sim_{tag}_{st}_n2250_sp{sp}_sub{sub}.json")
                    for b in BUDS:
                        B = int(float(b) * d["n_sim"])
                        for p, v in d[f"budget_{b}"].items():
                            ps = np.array(v["per_seed"])
                            if len(ps) != 8 or ps.max() > min(B, d["hits"]) or ps.min() < 0 or abs(v["mean"] - ps.mean()) > 1e-9: bad += 1; print("INV", tag, st, sp, sub, b, p)
                    # warm arms cannot exceed B and cold arms equal budget by construction
    print("invariant problems:", bad)
    # realised flip rate (own implementation of the spec)
    for rg, tag in TAGS.items():
        df = pd.read_parquet(f"runs/L1a/tab_{tag}.parquet"); cnt = df["cell30"].value_counts(); df = df[df["cell30"].isin(cnt[cnt >= 3].index)].reset_index(drop=True)
        fr = []
        for sp in range(4):
            rng0 = np.random.default_rng(sp); groups = df["cell30"].unique(); rng0.shuffle(groups); fit_g = set(groups[: len(groups) // 4]); fit = df[df["cell30"].isin(fit_g)]
            for s in range(8):
                cal = np.random.default_rng([s, 12]).permutation(len(fit))[:200]; y = fit.corrupted.values[cal].astype(int); fl = np.random.default_rng([s, 15]).random(200) < 0.25; fr.append(fl.mean())
        print(f"{rg}: realised flip fraction over 4 splits x 8 seeds: mean {np.mean(fr):.4f} min {np.min(fr):.3f} max {np.max(fr):.3f}")

if part == "C":
    from multiprocessing import Pool
    rung = sys.argv[2]; tag = TAGS[rung]; df0 = pd.read_parquet(f"runs/L1a/tab_{tag}.parquet"); cnt = df0["cell30"].value_counts(); df0 = df0[df0["cell30"].isin(cnt[cnt >= 3].index)].reset_index(drop=True)
    FLIP = 0.25

    def fitlr(Xa, ya, sw=None, mi=500):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=mi, class_weight="balanced"))
        return m.fit(Xa, ya) if sw is None else m.fit(Xa, ya, logisticregression__sample_weight=sw)

    def build(sp, sub):
        rng0 = np.random.default_rng(sp); groups = df0["cell30"].unique(); rng0.shuffle(groups); fit_g = set(groups[: len(groups) // 4])
        fit = df0[df0["cell30"].isin(fit_g)]; sim = df0[~df0["cell30"].isin(fit_g)].reset_index(drop=True)
        pids = sim["pid"].unique(); np.random.default_rng([sub, sp, 11]).shuffle(pids); cum = np.cumsum(sim.groupby("pid").size().reindex(pids).values)
        keep = pids[: int(np.argmin(np.abs(cum - 2250))) + 1]; sim = sim[sim["pid"].isin(set(keep))].reset_index(drop=True)
        return fit, sim

    def literal(args):
        sp, sub = args; fit, sim = build(sp, sub); Xf = fit[CHEAP].fillna(0).values; yf = fit["corrupted"].values.astype(int); Xs = sim[CHEAP].fillna(0).values; y = sim["corrupted"].values.astype(int)
        n = len(y); B = int(0.25 * n); R = max(B // 8, 40); out = {"warm": [], "w25": [], "free": [], "cold": []}
        for seed in range(8):
            cal = np.random.default_rng([seed, 12]).permutation(len(fit))[:200]; yc = yf[cal].copy()
            yc = np.where(np.random.default_rng([seed, 15]).random(200) < FLIP, 1 - yc, yc); Xc = Xf[cal]
            s0 = fitlr(Xc, yc, mi=1000).predict_proba(Xs)[:, 1]
            out["free"].append(int(y[np.argsort(-s0)[:B]].sum()))
            for name, wst in (("warm", 1.0), ("w25", 0.25)):
                score = s0.copy(); seen = []; seen_mask = np.zeros(n, bool); hits = 0
                for spent in range(1, B + 1):  # one pick per iteration, ranking recomputed from scratch
                    order = np.argsort(-score); c = next(i for i in order if not seen_mask[i]); seen_mask[c] = True; seen.append(c); hits += y[c]
                    if spent % R == 0 and spent < B:
                        yp = np.concatenate([yc, y[seen]])
                        if 0 < yp.sum() < len(yp):
                            sw = np.concatenate([np.full(200, wst), np.ones(len(seen))]) if wst != 1.0 else None
                            score = fitlr(np.vstack([Xc, Xs[seen]]), yp, sw).predict_proba(Xs)[:, 1]
                out[name].append(int(hits))
            rg = np.random.default_rng(seed); score = rg.random(n) * 1e-3; seen_mask = np.zeros(n, bool); seen = []; hits = 0
            for spent in range(1, B + 1):
                mu = np.where(~seen_mask)[0]; c = mu[np.argsort(-score[mu])][0]; seen_mask[c] = True; seen.append(c); hits += y[c]
                if spent % R == 0 and 0 < y[seen_mask].sum() < seen_mask.sum():
                    score = fitlr(Xs[seen_mask], y[seen_mask]).predict_proba(Xs)[:, 1]
            out["cold"].append(int(hits))
        return (sp, sub, out)

    jobs = [(sp, sub) for sp in range(4) for sub in range(4)]
    with Pool(12) as P: res = P.map(literal, jobs)
    tot = bad = 0
    for sp, sub, out in res:
        d = jl(f"{RD}/sim_{tag}_flip25_n2250_sp{sp}_sub{sub}.json")["budget_0.25"]
        for k, arm in (("warm", "warm_online_200"), ("w25", "warm_online_200_w25"), ("free", "recal_free_200"), ("cold", "greedy_online")):
            tot += 1
            if out[k] != d[arm]["per_seed"]: bad += 1; print("DIFF", rung, sp, sub, arm, out[k], d[arm]["per_seed"])
    print(f"{rung}: literal per-pick reimplementation vs raw per_seed, flip25 25% budget: {tot} (cluster, arm) lists of 8 seeds compared, {bad} differ")

if part == "D":
    a = pd.read_parquet("runs/L1a/tab_3B_q_corrupted.parquet"); b = pd.read_parquet("runs/L1a/tab_7B_q_corrupted.parquet")
    m = a.merge(b, on=["pid", "k"], suffixes=("_3", "_7")); print("shared rollouts (pid, k):", len(m), "label disagreements:", int((m.corrupted_3 != m.corrupted_7).sum()), "| rows 3B", len(a), "7B", len(b))
    for tag in TAGS.values():
        ds, H = grid(tag, None, "runs/L13a"); print(tag, "L13a ts_online/greedy_online 25%:", round(np.mean([d["budget_0.25"]["ts_online"]["mean"] for d in ds]) / np.mean([d["budget_0.25"]["greedy_online"]["mean"] for d in ds]), 4))
    for f in ("gate/gate_check.json", "gate/gate_check_ident.json", "gate/gate_check_post.json"):
        print(f, jl(f"{RD}/{f}").get("cells_compared"), jl(f"{RD}/{f}").get("differences"), jl(f"{RD}/{f}").get("passed"))

if part == "E":  # side check, not pre-registered: stale label-count control. Fixed rule given m stale (flip25) or m clean labels, m in 200 / 400 / 600 / 762 (= 200 + B), 25% budget, flip25 populations.
    from multiprocessing import Pool
    rung = sys.argv[2]; tag = TAGS[rung]; df0 = pd.read_parquet(f"runs/L1a/tab_{tag}.parquet"); cnt = df0["cell30"].value_counts(); df0 = df0[df0["cell30"].isin(cnt[cnt >= 3].index)].reset_index(drop=True)
    MS = (200, 400, 600, 762)

    def work(args):
        sp, sub = args; rng0 = np.random.default_rng(sp); groups = df0["cell30"].unique(); rng0.shuffle(groups); fit_g = set(groups[: len(groups) // 4])
        fit = df0[df0["cell30"].isin(fit_g)]; sim = df0[~df0["cell30"].isin(fit_g)].reset_index(drop=True)
        pids = sim["pid"].unique(); np.random.default_rng([sub, sp, 11]).shuffle(pids); cum = np.cumsum(sim.groupby("pid").size().reindex(pids).values)
        keep = pids[: int(np.argmin(np.abs(cum - 2250))) + 1]; sim = sim[sim["pid"].isin(set(keep))].reset_index(drop=True)
        Xf = fit[CHEAP].fillna(0).values; yf = fit["corrupted"].values.astype(int); Xs = sim[CHEAP].fillna(0).values; y = sim["corrupted"].values.astype(int); B = int(0.25 * len(y)); out = {}
        for seed in range(8):
            perm = np.random.default_rng([seed, 12]).permutation(len(fit)); u = np.random.default_rng([seed, 15]).random(762)
            for m in MS:
                cal = perm[:m]; yc = yf[cal]
                for kind in ("stale", "clean"):
                    yy = np.where(u[:m] < 0.25, 1 - yc, yc) if kind == "stale" else yc
                    sc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xf[cal], yy).predict_proba(Xs)[:, 1]
                    out.setdefault((kind, m), []).append(int(y[np.argsort(-sc)[:B]].sum()))
        return (sp, sub, out)

    with Pool(12) as P: res = P.map(work, [(sp, sub) for sp in range(4) for sub in range(4)])
    ds, H = grid(tag, "flip25"); w = H["0.25"]["warm_online_200"]; g = H["0.25"]["greedy_online"]; f0 = H["0.25"]["recal_free_200"]
    A = {k: np.array([r[2][k] for r in res], float) for k in res[0][2]}
    print(f"{rung}: m=200 stale reproduces raw recal_free_200 per_seed exactly:", bool((A[("stale", 200)] == f0).all()))
    rng = np.random.default_rng(5)
    print(f"{rung}: warm_stale_200 {w.mean():.1f}, cold {g.mean():.1f}")
    for kind in ("stale", "clean"):
        for m in MS:
            a = A[(kind, m)]; c = cmpr(w, a, rng); c2 = cmpr(a, g, rng)
            print(f"  fixed rule with {m} {kind} labels: {a.mean():.1f}; warm_stale / it = {c['ratio']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}] split-t-int [{c['tci'][0]:.3f}, {c['tci'][1]:.3f}] per-split {np.round(c['spr'], 3).tolist()}; it / cold = {c2['ratio']:.4f}")
