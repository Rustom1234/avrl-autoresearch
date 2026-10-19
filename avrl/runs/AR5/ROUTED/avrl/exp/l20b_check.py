"""L20b critic check: independent recomputation of the verdict from the feature and rollout files alone.

Does not import l1b_tuned_threshold or any analysis code. Rebuilds, with numpy and pandas only:
  - the 3,000-label frozen table on the test rows: per stratum (agreement, n_distinct, any_truncated),
    (errors + k0 * (1 - agreement)) / (count + k0), k0 50, fitted once on the fit pool and never refit;
  - the cell100 split for split seeds 0 to 7: cell100 groups with at least 3 docs, unique groups shuffled with
    default_rng(split), the first quarter of the groups dropped;
  - for split seeds 0 to 7 and tie seeds 0 to 7: the hit counts of plain fixed (lexsort on random key, -n_distinct, agreement)
    and frozen (lexsort on random key, -n_distinct, agreement, -score) picks at budget int(0.10 x n_sim);
  - the 3,000-label gap (mean over splits), the verdict PASS above +2.0 pp and KILL otherwise;
  - the 20 subset tables (numpy RandomState(100 + i).choice(3000, 200, replace=False)), their hit counts and means (context).
It compares every hit count, gap, verdict and pool composition with analysis/result.json, and reports a stream-overlap
diagnostic (context only, not a gate): how often aligned (pid, k) rollouts of the new pools match the p0 and L19c rollouts
exactly, against the same shares for shifted k, which are independent samples of the same prompts.
Writes <out>/check.json and exits 1 on any mismatch.

Usage: python -I l20b_check.py --fit_feats runs/L20b/fit/feats.parquet --eval_feats runs/L20b/test/feats.parquet
           --analysis runs/L20b/analysis/result.json --out DIR
"""
import os, sys, json, argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--fit_feats", required=True)
ap.add_argument("--eval_feats", required=True)
ap.add_argument("--analysis", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--p0_feats", default="/homes/rdubash/avrl_live/ROUTED/avrl/runs/p0/feats.parquet")
ap.add_argument("--l19c_feats", default="/homes/rdubash/avrl_live/ROUTED/avrl/runs/L19c/feats.parquet")
ap.add_argument("--p0_rollouts", default="/homes/rdubash/avrl_live/ROUTED/avrl/runs/p0/rollouts.parquet")
ap.add_argument("--l19c_rollouts", default="/homes/rdubash/avrl_live/ROUTED/avrl/runs/L19c/rollouts.parquet")
ap.add_argument("--fit_rollouts", required=True)
ap.add_argument("--test_rollouts", required=True)
ap.add_argument("--tol", type=float, default=1e-9)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

K0, BUDGET, EVAL_MAX, VERDICT = 50.0, 0.10, 1318, 2.0
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


def stream_overlap(new_path, old_path):
    """Share of aligned (pid, k) pairs with the same token count and the same answer, for shift 0 and the other shifts k + s."""
    new = pd.read_parquet(new_path).sort_values(["pid", "k"]); old = pd.read_parquet(old_path)
    old = old[old.pid.isin(new.pid.unique())].sort_values(["pid", "k"])
    n = int(new.pid.nunique()); g = int(new.k.nunique())
    assert len(old) == len(new) and int(old.pid.nunique()) == n, "stream overlap: pools differ in size"
    nt, ot = new.tokens.to_numpy().reshape(n, g), old.tokens.to_numpy().reshape(n, g)
    na, oa = new.answer.to_numpy().reshape(n, g), old.answer.to_numpy().reshape(n, g)
    tok, ans = [], []
    for sft in range(g):
        ot_s, oa_s = np.roll(ot, -sft, axis=1), np.roll(oa, -sft, axis=1)
        tok.append(float((nt == ot_s).mean())); ans.append(float((na == oa_s).mean()))
    return {"prompts": n, "samples_per_prompt": g, "identical_tokens_shift0": tok[0], "identical_answer_shift0": ans[0],
            "identical_tokens_mean_shifts_1_to_7": float(np.mean(tok[1:])), "identical_answer_mean_shifts_1_to_7": float(np.mean(ans[1:]))}


with open(a.analysis) as fh: res = json.load(fh)
p0 = pd.read_parquet(a.p0_feats)
fit = pd.read_parquet(a.fit_feats).reset_index(drop=True)
ev = pd.read_parquet(a.eval_feats).reset_index(drop=True)
l19 = pd.read_parquet(a.l19c_feats).reset_index(drop=True)

# Pool composition: fit pool = the p0 questions in p0 order with the p0 cell partition; test pool = L19c pids 0 to 1318
check(len(fit) == 3000, f"fit pool has {len(fit)} rows, expected 3000")
check(fit["question"].tolist() == p0["question"].tolist(), "fit questions differ from p0 in order")
check(fit["pid"].tolist() == list(range(3000)), "fit pids are not 0 to 2999 in order")
check(all(np.array_equal(fit[c].to_numpy(), p0[c].to_numpy()) for c in ["cell30", "cell100", "cell300"]),
      "fit cell columns differ from runs/p0/feats.parquet")
check(len(ev) == 1319 and ev["pid"].tolist() == list(range(1319)), f"test pool has {len(ev)} rows, expected pids 0 to 1318")
l19_t = l19[l19["pid"] <= EVAL_MAX].reset_index(drop=True)
check(ev["question"].tolist() == l19_t["question"].tolist(), "test questions differ from L19c pids 0 to 1318 in order")
check(all(np.array_equal(ev[c].to_numpy(), l19_t[c].to_numpy()) for c in ["cell30", "cell100", "cell300"]),
      "test cell columns differ from runs/L19c/feats.parquet")
overlap = len(set(ev["question"]) & set(fit["question"]))
check(overlap == 0, f"{overlap} test questions occur in the fit pool")
notes["test_questions_in_fit_pool"] = overlap
notes["fit_proxy_error_rate"] = float(fit["proxy_error"].mean())
notes["test_proxy_error_rate"] = float(ev["proxy_error"].mean())

# 3,000-label table on the test rows
crit_3000 = evaluate(ev, posterior(fit, K0))
raw3 = res["raw_3000_test"]
for s in SPLITS:
    r = raw3[str(s)]; c = crit_3000[s]
    check(r["n_sim"] == c["n_sim"] and r["budget"] == c["budget"], f"3000 table split {s}: n_sim or budget differs")
    check(r["fixed_hits"] == c["fixed_hits"], f"3000 table split {s}: fixed hit counts differ")
    check(r["frozen_hits"] == c["frozen_hits"], f"3000 table split {s}: frozen hit counts differ")
    check(len(c["spend_bad"]) == 0, f"3000 table split {s}: spend failure")
crit_gaps = [crit_3000[s]["gap_pp"] for s in SPLITS]
value = float(np.mean(crit_gaps)); sd = float(np.std(crit_gaps, ddof=1))
v = "PASS" if value > VERDICT else "KILL"
check(abs(value - res["value_pp"]) <= a.tol, f"value {value} vs result {res['value_pp']}")
check(abs(sd - res["table_3000"]["sd_pp"]) <= a.tol, "sd over split gaps differs")
check(v == res["verdict"], f"verdict {v} (critic) vs {res['verdict']} (result)")
check(abs(value - np.mean(res["table_3000"]["gap_pp_per_split"])) <= a.tol, "table_3000 gap list does not average to value")
notes["table_3000_gap_pp_critic"] = value
notes["table_3000_sd_pp_critic"] = sd

# 200-label subset tables on the test rows (context)
subs = []
for i in range(N_SUB):
    idx = np.random.RandomState(SUB_SEED0 + i).choice(len(fit), size=SUB_N, replace=False)
    check(len(set(idx.tolist())) == SUB_N, f"subset {i}: draw is not {SUB_N} distinct rows")
    post = posterior(fit.iloc[idx], K0)
    subs.append({"row_positions": [int(x) for x in idx], "test": evaluate(ev, post)})

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
sub_means = [float(np.mean([c["test"][s]["gap_pp"] for s in SPLITS])) for c in subs]
check(abs(float(np.mean(sub_means)) - res["table_200"]["mean_pp"]) <= a.tol, "200-label mean of subset means differs")
check(abs(float(np.std(sub_means, ddof=1)) - res["table_200"]["sd_pp"]) <= a.tol, "200-label sd over subset means differs")
notes["subset_split_hit_counts_compared"] = n_hits_compared
notes["max_abs_subset_gap_diff_pp"] = max_gap_diff
notes["subset_means_pp_critic"] = sub_means

check(res["gates"]["pass"] is True and res["gates"]["spend_gate"]["failures"] == 0, "result.json gates not all pass")

# Stream-overlap diagnostic (context only, not a gate)
notes["stream_overlap_fit_vs_p0"] = stream_overlap(a.fit_rollouts, a.p0_rollouts)
notes["stream_overlap_test_vs_L19c"] = stream_overlap(a.test_rollouts, a.l19c_rollouts)

check_out = {"experiment": "L20b", "critic": True, "problems": problems, "pass": len(problems) == 0,
             "value_pp_critic": value, "value_pp_result": res["value_pp"], "sd_pp_critic": sd, "verdict_critic": v,
             "verdict_result": res["verdict"], "verdict_rule": f"PASS above +{VERDICT:.1f} pp; KILL at or below +{VERDICT:.1f} pp",
             **notes}
with open(os.path.join(a.out, "check.json"), "w") as fh: json.dump(check_out, fh, indent=1, default=float)
print(f"critic: 3,000-label gap {value:+.4f} pp (result {res['value_pp']:+.4f}), sd {sd:.4f}, verdict {v} (result {res['verdict']})", flush=True)
print(f"critic: 200-label subset mean {np.mean(sub_means):+.4f} pp (result {res['table_200']['mean_pp']:+.4f}), sd {np.std(sub_means, ddof=1):.4f}", flush=True)
print(f"critic: hit counts compared {n_hits_compared} subset-split values plus the 3,000-label table; max abs subset gap diff "
      f"{max_gap_diff:.2e} pp", flush=True)
print(f"critic: stream overlap fit vs p0 {notes['stream_overlap_fit_vs_p0']}", flush=True)
print(f"critic: stream overlap test vs L19c {notes['stream_overlap_test_vs_L19c']}", flush=True)
print(f"critic: {len(problems)} problems" + ("" if not problems else ": " + "; ".join(problems[:10])), flush=True)
sys.exit(0 if not problems else 1)
