"""L13a: offset-ridge refit anchored at the matched 160-label fold table, at equal labels. CPU only, no GPU job.

Copied from avrl/exp/l10b_matched_precision.py (L10b is unchanged). Only the refit scorer changes. L10b refits a StandardScaler
plus LogisticRegression on six cheap features plus the table posterior, which moves the ranking freely: table_shrunk minus
table_zero was -1.07 on L2b and -1.11 on L4b. L13a keeps the matched table ranking as a fixed offset and moves only on
penalised evidence:

    linear predictor = offset + x . beta + b
    offset = logit of the row's matched table posterior (p0 rows: out-of-fold; eval rows: matched 160-label fold table)
    x = StandardScaler'd six cheap features (agreement, n_distinct, mean_tokens, std_tokens, any_truncated, log_qlen)
    fit minimises  sum_i w_i * logloss_i + 0.5 * ||beta||^2  with C = 1 and balanced w_i = n / (2 * n_class), the objective
    that class_weight='balanced' gives at the default C = 1; intercept b unpenalised; offset coefficient fixed at 1;
    beta starts at 0; scipy.optimize.minimize (L-BFGS-B) with an explicit gradient. A fit that stops with max |grad| >= 1e-5 raises.

Arms per subset and pool, 10% budget, splits 0 to 7, 8 tie-break seeds each:
  frozen        the 200-label subset table, fixed ranking, never refit (as L10b).
  table_zero    greedy from the matched posterior with beta = 0, no refit (same ranking as L10b table_zero).
  table_shrunk  greedy from the matched posterior, offset-ridge refit every R = max(budget // 8, 40) verified labels.
  fixed         plain fixed threshold (as L10b).

Everything else is as in L10b: 20 subsets (numpy RandomState(100 + i)), five-fold out-of-fold p0 posterior (RandomState(300 + i)),
one eval fold draw per eval row (RandomState(400 + i), the assumption carried over from L10b), tie key (-score, agreement,
-n_distinct, seeded draw), budget int(0.10 x n_sim), full pools.

ASSUMPTIONS (not in the spec, chosen here): (a) the posterior is clipped to [1e-6, 1 - 1e-6] before the logit, because a stratum
with agreement 1 and no errors has posterior exactly 0; the clip keeps every ordering (gate 3 checks this). (b) The StandardScaler
is fitted on the training rows of each refit (p0 rows plus verified eval rows), as L10b's pipeline did.

Gates before any verdict (full pools): 1 plain fixed reproduces runs/L2b and runs/L4b config B per split to 0.01 pp; 2 the
3,000-prompt frozen table reproduces runs/L5a per split to 1e-6 pp, and the frozen and fixed mean rates reproduce L10b; 3 with
refits off the table_zero arm reproduces runs/L10b/result.json table_zero minus frozen per subset and split to 1e-6 pp (+0.36 L2b,
-0.08 L4b); 4 the per-subset frozen gaps and row positions reproduce runs/L6b to 1e-6 pp; 5 matched zero-refit: table_zero hits
equal frozen_pick on the matched score; 6 spend: every arm takes int(budget x n_sim) distinct docs; 7 each subset has 200 distinct
prompts, fixed picks are identical across tables, and no eval question text occurs in p0.

Pid cut (--eval_pid_min / --eval_pid_max, L4b only): the eval frame is cut before the cell100 split, as in runs/L6a and runs/L11a.
For pid <= 1318, gates 1 and 2 are replaced by the L6a reproduction (fixed and frozen mean rates, per-subset row positions and frozen
gaps to 1e-6). For pid > 1318 there is no registered reference, so those gates are recorded as not applicable. Cut verdicts are
reported beside the full-pool verdict and are not claim verdicts.

Usage, one eval pool per call, then the combine step (l13a_combine.py) and the independent replay (l13a_critic_indep.py):
  python -I l13a_shrunk_refit.py --fit_feats FEATS --eval_feats FEATS --pool NAME --budget 0.10 --k0 50 --subsets 20
         --subset_size 200 --splits 0-7 --tie_seeds 8 --out OUTDIR [--eval_pid_min N] [--eval_pid_max N]
Writes <out>/result.json. If a gate fails, result.json gets verdict GATE_FAILED and the exit code is 1.
"""
import os, sys, json, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import l1b_tuned_threshold as L1B

RUNS = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
LABEL, GROUP = "proxy_error", "cell100"
STRATUM = ["agreement", "n_distinct", "any_truncated"]
CHEAP = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]  # G01 --cheap list
PASS_PP = 1.0  # L10b threshold, the same +1.0 pp as L9a (L6b used +2.0, L7a used +1.0)
REPRO_TOL_PP = 0.01
L5A_TOL_PP = 1e-6
L6B_TOL_PP = 1e-6
SUBSET_SEED0 = 100
OOF_SEED0 = 300
DRAW_SEED0 = 400  # eval-row fold draw, L10b only (assumption, not in the spec)
OFFSET_EPS = 1e-6  # posterior clip before the logit, L13a only (assumption, not in the spec)
C_REFIT = 1.0  # offset-ridge strength C, L13a only (spec: one strength, C = 1)
PID_TEST_MAX = 1318  # registered L4b test rows (pids 0 to 1318), as runs/L6a and runs/L11a
L6A_RESULT = f"{RUNS}/L6a/result.json"
L10B_RESULT = f"{RUNS}/L10b/result.json"
L6A_TOL_PP = 1e-6
L10B_TOL_PP = 1e-6
# Pool name -> (result.json holding plain fixed rates at 10%, config key of config B in that file)
REPRO_REF = {"L2b": (f"{RUNS}/L2b/result.json", "B"), "L4b": (f"{RUNS}/L4b/result.json", "B40")}
L5A_REF = f"{RUNS}/L5a/result.json"
L6B_REF = f"{RUNS}/L6b/result.json"

def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def parse_splits(s):
    if "-" in s:
        lo, hi = s.split("-"); return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def pool_name(path):
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


def fit_table(fit_df, k0):
    """Frozen table: posterior error rate per stratum from every labelled prompt of the fit pool. Never refit."""
    g = fit_df.assign(_y=fit_df[LABEL].to_numpy(int)).groupby(STRATUM)["_y"].agg(["sum", "size"]).reset_index()
    g = g.rename(columns={"sum": "errors", "size": "count"})
    g["prior"] = 1 - g["agreement"]
    g["post"] = (g["errors"] + k0 * g["prior"]) / (g["count"] + k0)
    return g


def attach_score(df, table):
    """Frozen score per doc: the table posterior of its stratum. A stratum missing from the table scores its prior."""
    for c in STRATUM: assert not df[c].isna().any(), c
    m = df[STRATUM].merge(table[STRATUM + ["post"]], on=STRATUM, how="left", validate="many_to_one")
    assert len(m) == len(df)
    post = m["post"].to_numpy(float); prior = 1 - df["agreement"].to_numpy(float)
    return df.assign(frozen_score=np.where(np.isnan(post), prior, post), in_table=~np.isnan(post))


def p0_folds(n, i):
    """Five balanced fold labels for the n p0 rows of subset i, from RandomState(OOF_SEED0 + i) (unchanged from L9a)."""
    fold = np.empty(n, int)
    fold[np.random.RandomState(OOF_SEED0 + i).permutation(n)] = np.arange(n) % 5
    return fold


def fold_tables(sub, k0, i):
    """The five 160-label fold tables of subset i: entry f is fitted on the 200 p0 rows outside fold f."""
    fold = p0_folds(len(sub), i)
    return [fit_table(sub[fold != f], k0) for f in range(5)]


def oof_posterior(sub, k0, i):
    """Out-of-fold table posterior for the 200 p0 rows of subset i: five balanced folds inside the subset, each row scored by
    the table fitted on the other four folds. Same values as L9a."""
    fold = p0_folds(len(sub), i); tabs = fold_tables(sub, k0, i)
    out = np.empty(len(sub))
    for f in range(5):
        out[fold == f] = attach_score(sub[fold == f], tabs[f])["frozen_score"].to_numpy(float)
    return out


def matched_score(raw, tabs, draw):
    """Eval-row posterior at matched precision (L10b): row r is scored by the 160-label fold table tabs[draw[r]], where draw is
    one fold per eval row. No eval label is used. A missing stratum scores its prior, as in attach_score."""
    out = np.empty(len(raw))
    for f in range(5):
        m = draw == f
        out[m] = attach_score(raw[m], tabs[f])["frozen_score"].to_numpy(float)
    return out


def fixed_pick(S, budget, seed):
    rnd = np.random.default_rng(seed).random(S.n)
    return np.lexsort((rnd, -S.nd, S.ag))[:budget]


def frozen_pick(S, score, budget, seed):
    rnd = np.random.default_rng(seed).random(S.n)
    return np.lexsort((rnd, -S.nd, S.ag, -score))[:budget]


def job(args):
    feats, split, table, frac, seeds = args
    df = attach_score(pd.read_parquet(feats), table)
    sim, n_groups = L1B.build_sim(df, GROUP, split)
    S = L1B.Sim(sim, LABEL)
    score = sim["frozen_score"].to_numpy(float)
    budget = int(frac * S.n)
    fixed, frozen, spend_bad = [], [], []
    for s in seeds:
        sel_f, sel_z = fixed_pick(S, budget, s), frozen_pick(S, score, budget, s)
        hf = int(S.y[sel_f].sum())
        assert hf == S.fixed(budget, s), "fixed_pick disagrees with L1B.Sim.fixed"
        fixed.append(hf); frozen.append(int(S.y[sel_z].sum()))
        for name, sel in (("fixed", sel_f), ("frozen", sel_z)):
            d = int(len(np.unique(sel)))
            if len(sel) != budget or d != budget:
                spend_bad.append({"policy": name, "seed": int(s), "distinct": d, "expected": budget})
    return {"split": split, "n_sim": S.n, "n_buckets": int(sim[GROUP].nunique()), "n_groups": n_groups,
            "hits_total": int(S.y.sum()), "budget": budget, "fixed_hits": fixed, "frozen_hits": frozen,
            "spend_bad": spend_bad, "in_table_frac": float(sim["in_table"].mean())}


def run_pool(feats, table, frac, splits, seeds):
    with Pool(min(len(splits), os.cpu_count())) as p:
        res = p.map(job, [(feats, s, table, frac, seeds) for s in splits])
    return {r["split"]: r for r in res}


def logit(p):
    p = np.clip(np.asarray(p, float), OFFSET_EPS, 1.0 - OFFSET_EPS)
    return np.log(p) - np.log1p(-p)


def offset_ridge(Z, y, off, C=C_REFIT):
    """Beta (on the standardised features Z) and intercept b for the linear predictor off + Z beta + b. Minimises
    C * sum_i w_i logloss_i + 0.5 * ||beta||^2 with balanced w_i = n / (2 n_class) (class_weight='balanced'), intercept
    unpenalised, offset coefficient fixed at 1, from beta = 0 and b = 0. Raises if the solver stops with max |grad| >= 1e-5."""
    yy = 2.0 * y - 1.0
    w = (len(y) / (2.0 * np.bincount(y, minlength=2).astype(float)))[y]
    p = Z.shape[1]

    def f(th):
        beta, b = th[:p], th[p]
        m = yy * (off + Z @ beta + b)
        dz = -C * w * yy * expit(-m)
        return C * np.sum(w * np.logaddexp(0.0, -m)) + 0.5 * beta @ beta, np.append(Z.T @ dz + beta, dz.sum())

    res = minimize(f, np.zeros(p + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": 10000, "maxfun": 20000, "ftol": 1e-15, "gtol": 1e-10})
    gmax = float(np.max(np.abs(f(res.x)[1])))
    if not gmax < 1e-5:
        raise RuntimeError(f"offset ridge did not converge: max |grad| {gmax:.3e}, {res.message}")
    return res.x[:p], float(res.x[p])


def shrunk_score(Xtr, ytr, offtr, Xall, offall):
    """Offset-ridge refit (L13a): StandardScaler fitted on the training rows, then off + Z beta + b on every row."""
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd = np.where(sd > 0, sd, 1.0)
    beta, b = offset_ridge((Xtr - mu) / sd, ytr, offtr)
    return offall + ((Xall - mu) / sd) @ beta + b


def table_picks(y, ag, nd, X, off, Xp, offp, yp, budget, seed, refit_every):
    """Greedy online allocator (G01 greedy_online, K = 1) started from the matched score off = logit(matched posterior). Each step
    takes the top unseen doc in the key order (-score, agreement, -n_distinct, seeded draw). refit_every None means beta = 0, so the
    ranking stays the matched table. Otherwise the offset ridge is refitted after every refit_every verified labels, on the 200 p0
    rows (offp, Xp) plus every verified eval label so far (off, X). Returns row positions in order."""
    rnd = np.random.default_rng(seed).random(len(y))
    seen = np.zeros(len(y), bool); score = off.copy()
    picks, next_fit = [], refit_every
    while len(picks) < budget:
        mu = np.flatnonzero(~seen)
        c = mu[np.lexsort((rnd[mu], -nd[mu], ag[mu], -score[mu]))[0]]
        seen[c] = True; picks.append(int(c))
        if refit_every and len(picks) >= next_fit and len(picks) < budget:
            score = shrunk_score(np.vstack([Xp, X[seen]]), np.concatenate([yp, y[seen]]),
                                 np.concatenate([offp, off[seen]]), X, off)
            next_fit += refit_every
    return np.asarray(picks)


def job_table(args):
    feats, split, tabs, draw, Xp, offp, yp, frac, seeds = args
    assert 0 < yp.sum() < len(yp), "p0 subset must hold both classes"
    raw = pd.read_parquet(feats)
    sim, _ = L1B.build_sim(raw.assign(matched_score=matched_score(raw, tabs, draw)), GROUP, split)
    y = sim[LABEL].to_numpy(int); ag = sim["agreement"].values; nd = sim["n_distinct"].values
    init = sim["matched_score"].to_numpy(float); off = logit(init)
    X = sim[CHEAP].fillna(0).to_numpy(float)
    S = L1B.Sim(sim, LABEL)
    budget = int(frac * len(y)); R = max(budget // 8, 40)
    zero, shrunk, matched, spend_bad = [], [], [], []
    for s in seeds:
        sel_z = table_picks(y, ag, nd, X, off, Xp, offp, yp, budget, s, None)
        sel_s = table_picks(y, ag, nd, X, off, Xp, offp, yp, budget, s, R)
        sel_m = frozen_pick(S, init, budget, s)
        zero.append(int(y[sel_z].sum())); shrunk.append(int(y[sel_s].sum())); matched.append(int(y[sel_m].sum()))
        for name, sel in (("table_zero", sel_z), ("table_shrunk", sel_s)):
            d = int(len(np.unique(sel)))
            if len(sel) != budget or d != budget:
                spend_bad.append({"policy": name, "seed": int(s), "distinct": d, "expected": budget})
    return {"split": split, "budget": budget, "zero_hits": zero, "shrunk_hits": shrunk, "matched_pick_hits": matched,
            "spend_bad": spend_bad}


def run_pool_table(feats, tabs, draw, Xp, offp, yp, frac, splits, seeds):
    with Pool(min(len(splits), os.cpu_count())) as p:
        res = p.map(job_table, [(feats, s, tabs, draw, Xp, offp, yp, frac, seeds) for s in splits])
    return {r["split"]: r for r in res}


def pool_summary(R, splits):
    gaps, fixed_pct, frozen_pct = [], [], []
    for s in splits:
        r = R[s]; b = r["budget"]
        f = 100.0 * np.asarray(r["fixed_hits"], float) / b; z = 100.0 * np.asarray(r["frozen_hits"], float) / b
        gaps.append(float((z - f).mean())); fixed_pct.append(float(f.mean())); frozen_pct.append(float(z.mean()))
    st = L1B.stats(gaps)
    return {"n_sim_per_split": [R[s]["n_sim"] for s in splits], "budget_per_split": [R[s]["budget"] for s in splits],
            "gap_pp_per_split": gaps, "fixed_pct_per_split": fixed_pct, "frozen_pct_per_split": frozen_pct,
            "in_table_frac_per_split": [R[s]["in_table_frac"] for s in splits],
            "mean_pp": st["mean_pp"], "sd_pp": st["sd_pp"], "n_positive": st["n_positive"],
            "n_above_pass_pp": int((np.asarray(gaps) > PASS_PP).sum()),
            "verdict": "PASS" if st["mean_pp"] > PASS_PP else "KILL"}


def table_summary(R, T, splits):
    """Per split, pp, mean over tie-break seeds: table_shrunk minus frozen (primary), table_shrunk minus plain fixed (context),
    table_zero minus frozen (context: matched precision with refits off), and the spread of table_shrunk hits across tie seeds."""
    vs_frozen, vs_fixed, zero_vs_frozen, pct, spread = [], [], [], [], []
    for s in splits:
        b = R[s]["budget"]; assert T[s]["budget"] == b
        l = 100.0 * np.asarray(T[s]["shrunk_hits"], float) / b
        q = 100.0 * np.asarray(T[s]["zero_hits"], float) / b
        z = 100.0 * np.asarray(R[s]["frozen_hits"], float) / b
        f = 100.0 * np.asarray(R[s]["fixed_hits"], float) / b
        vs_frozen.append(float((l - z).mean())); vs_fixed.append(float((l - f).mean()))
        zero_vs_frozen.append(float((q - z).mean()))
        pct.append(float(l.mean())); spread.append(int(max(T[s]["shrunk_hits"]) - min(T[s]["shrunk_hits"])))
    return {"gap_vs_frozen_pp_per_split": vs_frozen, "gap_vs_fixed_pp_per_split": vs_fixed, "shrunk_pct_per_split": pct,
            "zero_minus_frozen_pp_per_split": zero_vs_frozen, "mean_zero_minus_frozen_pp": float(np.mean(zero_vs_frozen)),
            "tie_seed_spread_hits_per_split": spread, "mean_vs_frozen_pp": float(np.mean(vs_frozen)),
            "mean_vs_fixed_pp": float(np.mean(vs_fixed))}


def repro_gate(name, R, splits):
    path, key = REPRO_REF[name]
    ref = json.load(open(path))["configs"][key]
    ref_pct = [float(x) for x in ref["fixed_hit_pct_per_split"]]; ref_calls = [int(x) for x in ref["calls_per_split"]]
    mine = [float(100.0 * np.mean(R[s]["fixed_hits"]) / R[s]["budget"]) for s in splits]
    calls = [R[s]["budget"] for s in splits]
    d = max(abs(m - r) for m, r in zip(mine, ref_pct))
    return {"ref": path, "ref_config": key, "ref_fixed_pct_per_split": ref_pct, "mine_fixed_pct_per_split": mine,
            "max_abs_diff_pp": float(d), "tol_pp": REPRO_TOL_PP, "calls_match": calls == ref_calls,
            "pass": bool(d <= REPRO_TOL_PP and calls == ref_calls)}


def l5a_gate(name, summary):
    ref = json.load(open(L5A_REF))["pools"][name]
    mine = summary["gap_pp_per_split"]; theirs = [float(x) for x in ref["gap_pp_per_split"]]
    d = max(abs(m - t) for m, t in zip(mine, theirs))
    return {"ref": L5A_REF, "ref_mean_pp": float(ref["mean_pp"]), "mine_mean_pp": summary["mean_pp"],
            "ref_gap_pp_per_split": theirs, "mine_gap_pp_per_split": mine, "max_abs_diff_pp": float(d),
            "tol_pp": L5A_TOL_PP, "pass": bool(d <= L5A_TOL_PP and len(mine) == len(theirs))}


def l6b_gate(name, subs):
    """Per-subset frozen gaps (plain fixed as base) and row positions against runs/L6b/result.json subset_detail."""
    ref = json.load(open(L6B_REF))["subset_detail"]
    worst, pos_ok = 0.0, len(ref) == len(subs)
    for x in subs:
        r = ref[x["subset"]]
        pos_ok = pos_ok and r["row_positions"] == x["row_positions"] and r["distinct_pids"] == x["distinct_pids"]
        mine = x["summary"][name]["gap_pp_per_split"]; theirs = [float(v) for v in r[f"{name}_gap_pp_per_split"]]
        worst = max(worst, max(abs(m - t) for m, t in zip(mine, theirs)))
    return {"ref": L6B_REF, "subsets": len(subs), "row_positions_match": bool(pos_ok), "max_abs_diff_pp": float(worst),
            "tol_pp": L6B_TOL_PP, "pass": bool(pos_ok and worst <= L6B_TOL_PP)}


def matched_zero_gate(name, subs, splits):
    """Matched zero-refit gate (replaces L9a's zero-refit gate, which cannot hold with a fold-table init): the greedy ranking with
    refits off must equal the frozen_pick order on the matched fold-table score, so table_zero hits equal matched_pick_hits on
    every subset, split and tie-break seed."""
    same = all(x["tab"][name][s]["zero_hits"] == x["tab"][name][s]["matched_pick_hits"] for x in subs for s in splits)
    return {"subsets": len(subs), "splits": len(splits), "hits_equal_frozen_pick_on_matched_score": bool(same), "pass": bool(same)}


def spend_gate(pools):
    bad = [dict(pool=name, split=s, **b) for name, R in pools.items() for s in R for b in R[s]["spend_bad"]]
    checks = sum(2 * len(R[next(iter(R))]["fixed_hits"]) * len(R) for R in pools.values())
    return {"checks": checks, "failures": len(bad), "examples": bad[:5], "pass": not bad}


def table_spend(subs, names):
    bad = [dict(pool=n, subset=x["subset"], split=s, **b)
           for x in subs for n in names for s in x["tab"][n] for b in x["tab"][n][s]["spend_bad"]]
    checks = sum(2 * len(x["tab"][n][s]["shrunk_hits"]) for x in subs for n in names for s in x["tab"][n])
    return {"checks": checks, "failures": len(bad), "examples": bad[:5], "pass": not bad}


def draw_subset(fit_df, i, size):
    """Row positions of label subset i: size distinct p0 prompts, numpy RandomState(100 + i), without replacement."""
    return np.random.RandomState(SUBSET_SEED0 + i).choice(len(fit_df), size=size, replace=False)


def write_json(out_dir, out):
    with open(os.path.join(out_dir, "result.json"), "w") as fh: json.dump(out, fh, indent=1, default=_jsonable)


def l6a_fixed_gate(name, summary):
    """Pid <= 1318 cut: the plain fixed mean rate over the 8 split seeds reproduces runs/L6a (table_200 fixed_pct_mean)."""
    ref = json.load(open(L6A_RESULT))["table_200"]
    mine = float(np.mean(summary["fixed_pct_per_split"])); d = abs(mine - float(ref["fixed_pct_mean"]))
    return {"ref": L6A_RESULT, "ref_fixed_pct_mean": float(ref["fixed_pct_mean"]), "mine_fixed_pct_mean": mine,
            "abs_diff_pp": float(d), "tol_pp": L6A_TOL_PP, "pass": bool(d <= L6A_TOL_PP)}


def l6a_subset_gate(name, subs):
    """Pid <= 1318 cut: per-subset row positions and frozen gaps over 8 splits, and the frozen mean rate, reproduce runs/L6a."""
    ref = json.load(open(L6A_RESULT))
    detail = ref["subset_detail"]
    pos_ok = len(detail) == len(subs) and all(detail[x["subset"]]["row_positions"] == x["row_positions"] for x in subs)
    worst = max(abs(m - float(t)) for x in subs
                for m, t in zip(x["summary"][name]["gap_pp_per_split"], detail[x["subset"]]["gap_pp_per_split"]))
    frozen = float(np.mean([np.mean(x["summary"][name]["frozen_pct_per_split"]) for x in subs]))
    d_frozen = abs(frozen - float(ref["table_200"]["frozen_pct_mean"]))
    return {"ref": L6A_RESULT, "subsets": len(subs), "row_positions_match": bool(pos_ok), "max_abs_diff_pp": float(worst),
            "frozen_pct_abs_diff": float(d_frozen), "tol_pp": L6A_TOL_PP,
            "pass": bool(pos_ok and worst <= L6A_TOL_PP and d_frozen <= L6A_TOL_PP)}


def l10b_reference_gate(name, subs):
    """Full pools: with refits off, table_zero minus frozen per subset and split, its mean, and the 20-subset frozen and plain fixed
    mean rates reproduce runs/L10b/result.json (gate 3 of the spec and the frozen-rate check)."""
    ref = json.load(open(L10B_RESULT))["pools"][name]
    theirs = [[float(v) for v in row] for row in ref["zero_minus_frozen_pp_per_split"]]
    mine = [x["tab_summary"][name]["zero_minus_frozen_pp_per_split"] for x in subs]
    worst = max(abs(m - t) for mr, tr in zip(mine, theirs) for m, t in zip(mr, tr))
    z_mean = float(np.mean([x["tab_summary"][name]["mean_zero_minus_frozen_pp"] for x in subs]))
    fixed = float(np.mean([np.mean(x["summary"][name]["fixed_pct_per_split"]) for x in subs]))
    frozen = float(np.mean([np.mean(x["summary"][name]["frozen_pct_per_split"]) for x in subs]))
    d = max(worst, abs(z_mean - float(ref["zero_minus_frozen_mean_pp"])),
            abs(fixed - float(ref["fixed_pct_mean"])), abs(frozen - float(ref["frozen_pct_mean"])))
    return {"ref": L10B_RESULT, "subsets": len(mine), "table_zero_minus_frozen_mean_pp": z_mean,
            "ref_zero_minus_frozen_mean_pp": float(ref["zero_minus_frozen_mean_pp"]), "fixed_pct_mean": fixed,
            "ref_fixed_pct_mean": float(ref["fixed_pct_mean"]), "frozen_pct_mean": frozen,
            "ref_frozen_pct_mean": float(ref["frozen_pct_mean"]), "max_abs_diff_pp": float(d), "tol_pp": L10B_TOL_PP,
            "pass": bool(len(mine) == len(theirs) and d <= L10B_TOL_PP)}


def gate_line(label, n, v):
    extra = f" (not applicable: {v['not_applicable']})" if "not_applicable" in v else ""
    diff = f", max abs diff {v['max_abs_diff_pp']:.2e} pp" if "max_abs_diff_pp" in v else ""
    return f"{label} {n}: pass {v['pass']}{diff}{extra}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_feats", required=True)
    ap.add_argument("--eval_feats", nargs="+", required=True)
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--k0", type=float, default=50.0)
    ap.add_argument("--subsets", type=int, default=20)
    ap.add_argument("--subset_size", type=int, default=200)
    ap.add_argument("--splits", default="0-7")
    ap.add_argument("--tie_seeds", type=int, default=8)
    ap.add_argument("--pool", default=None, help="pool name; required with a pid cut")
    ap.add_argument("--eval_pid_min", type=int, default=None, help="inclusive lower pid bound, applied before the cell100 split")
    ap.add_argument("--eval_pid_max", type=int, default=None, help="inclusive upper pid bound, applied before the cell100 split")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    splits, seeds = parse_splits(a.splits), list(range(a.tie_seeds))
    cut = a.eval_pid_min is not None or a.eval_pid_max is not None
    if cut and (len(a.eval_feats) != 1 or a.pool is None):
        sys.exit("a pid cut takes one eval pool and --pool NAME")

    fit_df = pd.read_parquet(a.fit_feats)
    evals = {}
    for f in a.eval_feats:
        name = a.pool or pool_name(f)
        if not cut and name not in REPRO_REF: sys.exit(f"no reproduction reference for pool {name}")
        if cut:
            ev = pd.read_parquet(f); keep = np.ones(len(ev), bool)
            if a.eval_pid_min is not None: keep &= ev["pid"].to_numpy() >= a.eval_pid_min
            if a.eval_pid_max is not None: keep &= ev["pid"].to_numpy() <= a.eval_pid_max
            f = os.path.join(a.out, "eval_cut_feats.parquet"); ev[keep].reset_index(drop=True).to_parquet(f)
        evals[name] = f
    p0_questions = set(fit_df["question"])
    overlap = {n: int(pd.read_parquet(f)["question"].isin(p0_questions).sum()) for n, f in evals.items()}
    n_eval = {n: len(pd.read_parquet(f)) for n, f in evals.items()}

    full_table = fit_table(fit_df, a.k0)
    print(f"gate table: {len(full_table)} strata from {len(fit_df)} p0 prompts, k0 {a.k0:g}, no refit; eval rows {n_eval}", flush=True)
    full = {n: run_pool(f, full_table, a.budget, splits, seeds) for n, f in evals.items()}
    full_sum = {n: pool_summary(full[n], splits) for n in evals}

    subs = []
    for i in range(a.subsets):
        idx = draw_subset(fit_df, i, a.subset_size)
        sub = fit_df.iloc[idx]
        table = fit_table(sub, a.k0)
        tabs = fold_tables(sub, a.k0, i)  # the five 160-label fold tables, used for eval rows only (matched precision)
        draw = {n: np.random.RandomState(DRAW_SEED0 + i).randint(0, 5, size=n_eval[n]) for n in evals}
        Xp = sub[CHEAP].fillna(0).to_numpy(float)  # p0 refit features: six cheap features, no posterior column
        offp = logit(oof_posterior(sub, a.k0, i))  # p0 offset: logit of the out-of-fold posterior
        yp = sub[LABEL].to_numpy(int)
        runs = {n: run_pool(f, table, a.budget, splits, seeds) for n, f in evals.items()}
        tab = {n: run_pool_table(f, tabs, draw[n], Xp, offp, yp, a.budget, splits, seeds) for n, f in evals.items()}
        summ = {n: pool_summary(runs[n], splits) for n in evals}
        tsum = {n: table_summary(runs[n], tab[n], splits) for n in evals}
        subs.append({"subset": i, "row_positions": [int(x) for x in idx], "distinct_pids": int(sub["pid"].nunique()),
                     "strata_in_table": int(len(table)), "eval_fold_counts": {n: np.bincount(draw[n], minlength=5).tolist()
                                                                              for n in evals},
                     "runs": runs, "summary": summ, "tab": tab, "tab_summary": tsum})
        print(f"subset {i:2d}: {len(table)} strata, frozen minus fixed " + ", ".join(f"{n} {summ[n]['mean_pp']:+.2f}" for n in evals)
              + ", table_shrunk minus frozen " + ", ".join(f"{n} {tsum[n]['mean_vs_frozen_pp']:+.2f}" for n in evals)
              + ", table_zero minus frozen " + ", ".join(f"{n} {tsum[n]['mean_zero_minus_frozen_pp']:+.2f}" for n in evals), flush=True)

    fixed_same = all(R[s]["fixed_hits"] == full[n][s]["fixed_hits"]
                     for x in subs for n, R in x["runs"].items() for s in splits)
    sg = spend_gate({**{f"{n}/gate": full[n] for n in evals},
                     **{f"{n}/subset{x['subset']}": x["runs"][n] for x in subs for n in evals}})
    tg = table_spend(subs, list(evals))
    na = "no registered reference for this cut"
    if cut:
        ref_ok = a.eval_pid_max == PID_TEST_MAX and a.eval_pid_min is None
        repro = {n: l6a_fixed_gate(n, full_sum[n]) if ref_ok else {"pass": True, "not_applicable": na} for n in evals}
        l5a = {n: {"pass": True, "not_applicable": na} for n in evals}
        l6b = {n: l6a_subset_gate(n, subs) if ref_ok else {"pass": True, "not_applicable": na} for n in evals}
        l10 = {n: {"pass": True, "not_applicable": na} for n in evals}
    else:
        repro = {n: repro_gate(n, full[n], splits) for n in evals}
        l5a = {n: l5a_gate(n, full_sum[n]) for n in evals}
        l6b = {n: l6b_gate(n, subs) for n in evals}
        l10 = {n: l10b_reference_gate(n, subs) for n in evals}
    zg = {n: matched_zero_gate(n, subs, splits) for n in evals}
    subset_ok = all(x["distinct_pids"] == a.subset_size and len(set(x["row_positions"])) == a.subset_size for x in subs)
    overlap_ok = all(v == 0 for v in overlap.values())
    gates_ok = (sg["pass"] and tg["pass"] and subset_ok and overlap_ok and fixed_same
                and all(g["pass"] for g in repro.values()) and all(g["pass"] for g in l5a.values())
                and all(g["pass"] for g in l6b.values()) and all(g["pass"] for g in zg.values())
                and all(g["pass"] for g in l10.values()))
    scope = ("full pool, registered" if not cut else
             f"registered pid cut {a.eval_pid_min}..{a.eval_pid_max} on pool {a.pool}, reported beside the full pool, not a claim verdict")
    out = {"experiment": "L13a", "scope": scope, "label": LABEL, "group": GROUP, "stratum": STRATUM, "cheap": CHEAP,
           "refit_model": "offset ridge: linear predictor = logit(matched table posterior) + x.beta + b, x = StandardScaler'd cheap "
                          "features, fit minimises sum_i w_i logloss_i + 0.5 ||beta||^2 (C = 1, balanced w_i), b unpenalised, "
                          "offset coefficient 1, beta from 0",
           "offset_clip": OFFSET_EPS, "C": C_REFIT, "k0": a.k0, "budget_frac": a.budget, "fit_feats": a.fit_feats,
           "fit_prompts": int(len(fit_df)), "subsets": a.subsets, "subset_size": a.subset_size,
           "subset_draw": "numpy RandomState(100 + i).choice(n_p0_rows, subset_size, replace=False)",
           "oof_folds": "5 balanced folds inside each 200-label subset, numpy RandomState(300 + i).permutation (as L9a, L10b)",
           "eval_fold_draw": "ASSUMPTION carried from L10b (not in the spec): per eval pool, one fold f in 0..4 per eval row, numpy "
                             "RandomState(400 + i).randint(0, 5, size = eval rows), file row order",
           "pid_cut": None if not cut else {"min": a.eval_pid_min, "max": a.eval_pid_max, "eval_rows": n_eval},
           "seeds": seeds, "splits": splits, "sd_ddof": 1, "pass_threshold_pp": PASS_PP, "tie_seeds": a.tie_seeds,
           "eval_question_overlap_with_p0": overlap,
           "gates": {"fixed_reproduction": repro, "frozen_3000_reproduction_L5a": l5a, "frozen_subset_reproduction_L6b": l6b,
                     "matched_zero_gate": zg, "l10b_table_zero_reference": l10, "spend_gate": sg, "table_spend_gate": tg,
                     "fixed_identical_across_tables": bool(fixed_same), "subset_distinct_prompts": bool(subset_ok),
                     "eval_question_overlap_zero": bool(overlap_ok), "pass": bool(gates_ok)}}
    for label, G in (("fixed reproduction", repro), ("L5a 3000-table", l5a), ("subset reproduction", l6b),
                     ("matched zero", zg), ("L10b table_zero", l10)):
        for n, v in G.items(): print(gate_line(label, n, v), flush=True)
    print(f"spend gate: {sg['checks']} checks, failures {sg['failures']}; table spend: {tg['checks']} checks, failures "
          f"{tg['failures']}; subsets distinct {subset_ok}; fixed identical {fixed_same}; eval overlap with p0 {overlap}; "
          f"gates {'PASS' if gates_ok else 'FAIL'}", flush=True)
    if not gates_ok:
        out["verdict"] = "GATE_FAILED"; write_json(a.out, out)
        print("GATE FAILED: no L13a verdict written", flush=True); sys.exit(1)

    pools = {}
    for n in evals:
        ts = [x["tab_summary"][n] for x in subs]; sm = [x["summary"][n] for x in subs]
        m = np.asarray([t["mean_vs_frozen_pp"] for t in ts], float)
        zero = np.asarray([t["mean_zero_minus_frozen_pp"] for t in ts], float)
        pools[n] = {"feats": evals[n], "eval_rows": n_eval[n], "mean_pp": float(m.mean()), "sd_pp": float(m.std(ddof=1)),
                    "n_subsets": len(m), "n_subsets_above_pass": int((m > PASS_PP).sum()),
                    "n_subsets_at_or_below_pass": int((m <= PASS_PP).sum()),
                    "min_subset_mean_pp": float(m.min()), "max_subset_mean_pp": float(m.max()), "subset_mean_pp": [float(v) for v in m],
                    "gap_pp_per_subset": [t["gap_vs_frozen_pp_per_split"] for t in ts],
                    "n_split_gaps": int(sum(len(t["gap_vs_frozen_pp_per_split"]) for t in ts)),
                    "n_split_gaps_above_pass": int(sum((np.asarray(t["gap_vs_frozen_pp_per_split"]) > PASS_PP).sum() for t in ts)),
                    "max_tie_seed_spread_hits": int(max(max(t["tie_seed_spread_hits_per_split"]) for t in ts)),
                    "shrunk_pct_mean": float(np.mean([np.mean(t["shrunk_pct_per_split"]) for t in ts])),
                    "fixed_pct_mean": float(np.mean([np.mean(x["fixed_pct_per_split"]) for x in sm])),
                    "frozen_pct_mean": float(np.mean([np.mean(x["frozen_pct_per_split"]) for x in sm])),
                    "frozen_vs_fixed_mean_pp": float(np.mean([x["mean_pp"] for x in sm])),
                    "shrunk_vs_fixed_mean_pp": float(np.mean([t["mean_vs_fixed_pp"] for t in ts])),
                    "zero_minus_frozen_mean_pp": float(zero.mean()),
                    "zero_minus_frozen_pp_per_split": [t["zero_minus_frozen_pp_per_split"] for t in ts],
                    "shrunk_minus_zero_mean_pp": float((m - zero).mean()),
                    "full_table_3000_mean_pp": full_sum[n]["mean_pp"],
                    "verdict": "PASS" if m.mean() > PASS_PP else "KILL"}
    out["pools"] = pools
    out["verdicts"] = {n: pools[n]["verdict"] for n in evals}
    out["verdict"] = "PASS" if all(v == "PASS" for v in out["verdicts"].values()) else "KILL"
    out["subset_detail"] = [dict({"subset": x["subset"], "row_positions": x["row_positions"], "distinct_pids": x["distinct_pids"],
                                  "strata_in_table": x["strata_in_table"]},
                                 **{f"{n}_table_shrunk_minus_frozen_pp": x["tab_summary"][n]["mean_vs_frozen_pp"] for n in evals},
                                 **{f"{n}_table_shrunk_minus_frozen_pp_per_split": x["tab_summary"][n]["gap_vs_frozen_pp_per_split"]
                                    for n in evals},
                                 **{f"{n}_table_zero_minus_frozen_pp": x["tab_summary"][n]["mean_zero_minus_frozen_pp"] for n in evals},
                                 **{f"{n}_frozen_minus_fixed_pp": x["summary"][n]["mean_pp"] for n in evals},
                                 **{f"{n}_gap_pp_per_split": x["summary"][n]["gap_pp_per_split"] for n in evals},
                                 **{f"{n}_table_shrunk_minus_fixed_pp": x["tab_summary"][n]["mean_vs_fixed_pp"] for n in evals},
                                 **{f"{n}_eval_fold_counts": x["eval_fold_counts"][n] for n in evals})
                           for x in subs]
    for n in pools:
        p = pools[n]
        print(f"{n} ({scope}): table_shrunk minus frozen 200-label, mean over {p['n_subsets']} subsets {p['mean_pp']:+.2f}, sd over "
              f"subset means {p['sd_pp']:.2f}, subsets above +1.0 {p['n_subsets_above_pass']} of {p['n_subsets']}, min "
              f"{p['min_subset_mean_pp']:+.2f} max {p['max_subset_mean_pp']:+.2f}, table_shrunk minus table_zero "
              f"{p['shrunk_minus_zero_mean_pp']:+.2f}, table_zero minus frozen {p['zero_minus_frozen_mean_pp']:+.2f}, "
              f"frozen minus fixed {p['frozen_vs_fixed_mean_pp']:+.2f}, verdict {p['verdict']}", flush=True)
    print(f"L13a verdict: {out['verdict']} (per pool {out['verdicts']})", flush=True)
    write_json(a.out, out)
    print(f"wrote {os.path.join(a.out, 'result.json')}", flush=True)


if __name__ == "__main__":
    main()
