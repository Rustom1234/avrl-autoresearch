"""L19c critic check: independent recomputation of the 200-label test-row tables from the feature files alone.

Does not import l1b_tuned_threshold or any analysis code. Rebuilds, with numpy and pandas only:
  - the 20 label subsets (numpy RandomState(100 + i).choice(3000, 200, replace=False) over the p0 rows) and each frozen table:
    per stratum (agreement, n_distinct, any_truncated), (errors + k0 * (1 - agreement)) / (count + k0), k0 50, no refit;
  - the cell100 split for split seeds 0 to 7: cell100 groups with at least 3 docs, unique groups shuffled with
    default_rng(split), the first quarter of the groups dropped;
  - for every subset, split and tie-break seed 0 to 7: the hit counts of plain fixed and frozen picks at 10% (budget
    int(0.10 x n_sim)), with the order (fixed) lexsort on (random key, -n_distinct, agreement) and (frozen) lexsort on
    (random key, -n_distinct, agreement, -score);
  - the gap per split, the mean over splits per subset, the mean over subsets, the 3,000-label table on the test rows, the
    200-label gain on pids above 1318, the L16a percentiles and the verdict rule.
It compares every hit count and gap with analysis/result.json and checks the pool composition against runs/L4b.
Writes <out>/check.json and exits 1 on any mismatch.

Usage: python -I l19c_check.py --fit_feats P0 --eval_feats runs/L19c/feats.parquet --analysis runs/L19c/analysis/result.json --out DIR
"""
import os, sys, json, argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--fit_feats", required=True)
ap.add_argument("--eval_feats", required=True)
ap.add_argument("--analysis", required=True)
ap.add_argument("--l4b_feats", default="/homes/rdubash/avrl_live/ROUTED/avrl/runs/L4b/feats.parquet")
ap.add_argument("--l16a", default="/homes/rdubash/avrl_live/ROUTED/avrl/runs/L16a/result.json")
ap.add_argument("--out", required=True)
ap.add_argument("--tol", type=float, default=1e-9)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

K0, BUDGET, EVAL_MAX, VERDICT = 50.0, 0.10, 1318, 1.0
N_SUB, SUB_N, SUB_SEED0 = 20, 200, 100
SPLITS, SEEDS = list(range(8)), list(range(8))
ST = ["agreement", "n_distinct", "any_truncated"]
problems, notes = [], {}


def check(cond, msg):
    if not cond: problems.append(msg)


def posterior(df, k0):
    out = {}
    for (ag, nd, tr), g in df.groupby(ST):
        out[(float(ag), int(nd), int(tr))] = (int(g["proxy_error"].sum()) + k0 * (1.0 - float(ag))) / (len(g) + k0)
    return out


def score_rows(df, post):
    keys = zip(df["agreement"].astype(float), df["n_distinct"].astype(int), df["any_truncated"].astype(int))
    return np.array([post.get((float(ag), int(nd), int(tr)), 1.0 - float(ag)) for ag, nd, tr in keys])


def split_frame(df, split):
    cnt = df["cell100"].value_counts()
    kept = df[df["cell100"].isin(cnt[cnt >= 3].index.tolist())].reset_index(drop=True)
    groups = kept["cell100"].unique()
    np.random.default_rng(split).shuffle(groups)
    fit = groups[: len(groups) // 4].tolist()
    return kept[~kept["cell100"].isin(fit)].reset_index(drop=True)


def evaluate(frame, post):
    frame = frame.assign(score=score_rows(frame, post))
    out = {}
    for s in SPLITS:
        sim = split_frame(frame, s)
        n = len(sim); budget = int(BUDGET * n)
        y = sim["proxy_error"].to_numpy(int); ag = sim["agreement"].to_numpy(float)
        nd = sim["n_distinct"].to_numpy(float); sc = sim["score"].to_numpy(float)
        fx, fz, spend = [], [], []
        for seed in SEEDS:
            rnd = np.random.default_rng(seed).random(n)
            sel_f = np.lexsort((rnd, -nd, ag))[:budget]
            sel_z = np.lexsort((rnd, -nd, ag, -sc))[:budget]
            fx.append(int(y[sel_f].sum())); fz.append(int(y[sel_z].sum()))
            for sel in (sel_f, sel_z):
                if len(sel) != budget or len(np.unique(sel)) != budget: spend.append({"split": s, "seed": seed})
        gap = float(np.mean([100.0 * z / budget - 100.0 * f / budget for f, z in zip(fx, fz)]))
        out[s] = {"n_sim": n, "budget": budget, "fixed_hits": fx, "frozen_hits": fz, "gap_pp": gap, "spend_bad": spend}
    return out


with open(a.analysis) as fh: res = json.load(fh)
p0 = pd.read_parquet(a.fit_feats)
ev = pd.read_parquet(a.eval_feats).reset_index(drop=True)
l4b = pd.read_parquet(a.l4b_feats).reset_index(drop=True)

# Pool composition: the same questions in the same pid order as the L4b pool, and the same cell100 partition
check(len(ev) == 2792 and len(l4b) == 2792, f"pool sizes {len(ev)} and {len(l4b)}, expected 2792")
check(ev["question"].tolist() == l4b["question"].tolist(), "L19c questions differ from L4b in pid order")
check(ev["pid"].tolist() == list(range(2792)), "L19c pids are not 0 to 2791 in order")
notes["cell100_identical_to_L4b"] = bool((ev["cell100"].to_numpy() == l4b["cell100"].to_numpy()).all())
notes["n_test_rows"] = int((ev["pid"] <= EVAL_MAX).sum())
check(notes["n_test_rows"] == 1319, "test rows are not 1,319")
check(len(set(ev["question"]) & set(p0["question"])) == 0, "L19c questions overlap p0")

ev_t = ev[ev["pid"] <= EVAL_MAX].reset_index(drop=True)
ev_r = ev[ev["pid"] > EVAL_MAX].reset_index(drop=True)

# 3,000-label table on the test rows
crit_3000 = evaluate(ev_t, posterior(p0, K0))
raw3 = res["raw_3000_test"]
for s in SPLITS:
    r = raw3[str(s)]; c = crit_3000[s]
    check(r["n_sim"] == c["n_sim"] and r["budget"] == c["budget"], f"3000 table split {s}: n_sim or budget differs")
    check(r["fixed_hits"] == c["fixed_hits"] and r["frozen_hits"] == c["frozen_hits"], f"3000 table split {s}: hit counts differ")
    check(len(c["spend_bad"]) == 0, f"3000 table split {s}: spend failure")
crit_3000_mean = float(np.mean([crit_3000[s]["gap_pp"] for s in SPLITS]))
check(abs(crit_3000_mean - res["context_only"]["table_3000_test"]["mean_pp"]) <= a.tol,
      "3,000-label gap mean differs from result.json")
notes["table_3000_test_gap_pp_critic"] = crit_3000_mean

# 200-label tables: subset draws, tables, test and remaining rows
subs = []
for i in range(N_SUB):
    idx = np.random.RandomState(SUB_SEED0 + i).choice(len(p0), size=SUB_N, replace=False)
    check(len(set(idx.tolist())) == SUB_N, f"subset {i}: draw is not {SUB_N} distinct rows")
    post = posterior(p0.iloc[idx], K0)
    subs.append({"row_positions": [int(x) for x in idx], "test": evaluate(ev_t, post), "remain": evaluate(ev_r, post)})

det = res["subset_detail"]
check(len(det) == N_SUB, f"result.json has {len(det)} subsets, expected {N_SUB}")
n_hits_compared, max_gap_diff = 0, 0.0
for i, (d, c) in enumerate(zip(det, subs)):
    check(d["subset"] == i, f"subset index {i} mismatch")
    check(d["row_positions"] == c["row_positions"], f"subset {i}: row positions differ")
    gaps_c = [c["test"][s]["gap_pp"] for s in SPLITS]
    for s in SPLITS:
        r = d["raw"][str(s)]; cc = c["test"][s]
        check(r["n_sim"] == cc["n_sim"] and r["budget"] == cc["budget"], f"subset {i} split {s}: n_sim or budget differs")
        check(r["fixed_hits"] == cc["fixed_hits"], f"subset {i} split {s}: fixed hit counts differ")
        check(r["frozen_hits"] == cc["frozen_hits"], f"subset {i} split {s}: frozen hit counts differ")
        n_hits_compared += len(cc["fixed_hits"]) + len(cc["frozen_hits"])
        check(len(cc["spend_bad"]) == 0, f"subset {i} split {s}: spend failure")
        diff = abs(d["gap_pp_per_split"][s] - cc["gap_pp"]); max_gap_diff = max(max_gap_diff, diff)
        check(diff <= a.tol, f"subset {i} split {s}: gap {d['gap_pp_per_split'][s]} vs {cc['gap_pp']}")
    check(abs(d["mean_pp"] - float(np.mean(gaps_c))) <= a.tol, f"subset {i}: mean gap differs")
    rem_mean = float(np.mean([c["remain"][s]["gap_pp"] for s in SPLITS]))
    check(abs(d["remain_mean_pp"] - rem_mean) <= a.tol, f"subset {i}: pids above 1318 mean differs")
notes["subset_split_hit_counts_compared"] = n_hits_compared
notes["max_abs_gap_diff_pp"] = max_gap_diff

# Verdict from the critic's own subset means and the L16a cut values
sub_means = [float(np.mean([c["test"][s]["gap_pp"] for s in SPLITS])) for c in subs]
value = float(np.mean(sub_means)); sd = float(np.std(sub_means, ddof=1))
rem_sub = [float(np.mean([c["remain"][s]["gap_pp"] for s in SPLITS])) for c in subs]
with open(a.l16a) as fh: cuts = json.load(fh)["train_cuts"]["values_pp"]
check(len(cuts) == 48, f"L16a has {len(cuts)} cut means, expected 48")
p5, p95 = float(np.percentile(cuts, 5)), float(np.percentile(cuts, 95))
if value <= VERDICT or value < p5: v = "KILL"
elif value > p95: v = "INCONCLUSIVE"
else: v = "PASS"
check(abs(value - res["value_pp"]) <= a.tol, f"value {value} vs result {res['value_pp']}")
check(abs(sd - res["table_200"]["sd_pp"]) <= a.tol, "sd over subset means differs")
check(v == res["verdict"], f"verdict {v} (critic) vs {res['verdict']} (result)")
check(abs(p5 - res["reference"]["L16a_cut_p5_pp"]) <= a.tol and abs(p95 - res["reference"]["L16a_cut_p95_pp"]) <= a.tol,
      "L16a p5 or p95 differs from result.json")
check(res["gates"]["pass"] is True and res["gates"]["spend_gate"]["failures"] == 0, "result.json gates not all pass")

check_out = {"experiment": "L19c", "critic": True, "problems": problems, "pass": len(problems) == 0,
             "value_pp_critic": value, "value_pp_result": res["value_pp"], "sd_pp_critic": sd, "verdict_critic": v,
             "verdict_result": res["verdict"], "L16a_p5_pp": p5, "L16a_p95_pp": p95, "verdict_rule": "PASS above +1.0 and inside [p5, p95]; KILL at or below +1.0 or below p5; INCONCLUSIVE above p95",
             "subset_means_pp_critic": sub_means, "remaining_200_mean_pp_critic": float(np.mean(rem_sub)), **notes}
with open(os.path.join(a.out, "check.json"), "w") as fh: json.dump(check_out, fh, indent=1)
print(f"critic: 200-label value {value:+.4f} pp (result {res['value_pp']:+.4f}), sd {sd:.4f}, band [{p5:+.4f}, {p95:+.4f}], "
      f"verdict {v} (result {res['verdict']})", flush=True)
print(f"critic: 3,000-label test gap {crit_3000_mean:+.4f} pp; pids above 1318, 200-label mean {np.mean(rem_sub):+.4f} pp", flush=True)
print(f"critic: hit counts compared {n_hits_compared} subset-split values plus the 3,000-label table; max abs gap diff "
      f"{max_gap_diff:.2e} pp; notes {notes}", flush=True)
print(f"critic: {len(problems)} problems" + ("" if not problems else ": " + "; ".join(problems[:10])), flush=True)
sys.exit(0 if not problems else 1)
