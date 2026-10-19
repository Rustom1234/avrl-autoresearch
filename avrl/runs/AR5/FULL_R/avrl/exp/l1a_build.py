"""L1a build (CPU): rollout-level, agreement-free tables from the p0 rollouts, the E04 judge scores and the p0 prompt features.
Usage: l1a_build.py --rollouts runs/p0/rollouts.parquet --feats runs/p0/feats.parquet --judges runs/E04 --out runs/L1a

Every one of the 24,000 rollouts is one decision. Columns written to the tab_*.parquet tables are per-rollout (or per-prompt constants such as the cell ids, the
16 PCs and log_qlen) and NEVER any statistic over the 8 rollouts of a prompt (FORBIDDEN below is asserted absent). judge_p is the answer-only judge score of THIS rollout's answer.
Accept rules: q = accepted iff judge_p >= the (1 - 0.6421) quantile of judge_p over all 24,000 rollouts (0.6421 = overall rollout accuracy; the only gold-derived scalar, it fixes the
accept rate); p05 = judge_p > 0.5 (7B-AWQ only, secondary). Labels: corrupted = accepted and wrong by gold, population = accepted rollouts only; judge_wrong = (accepted != correct), population = all rollouts.
Tables: tab_<3B|7B>_<q|p05>_<corrupted|judgewrong>.parquet. Gate (written to build.json, exit 3 if it fails): >= 200 positives per table and judge_p rollout AUC 0.654 (3B) / 0.702 (7B-AWQ) to 3 decimals."""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

FORBIDDEN = {"agreement", "n_distinct", "pass_rate", "majority", "majority_correct", "proxy", "proxy_error", "mean_tokens", "std_tokens", "any_truncated",
             "judge_top", "judge_margin", "judge_mean", "judge_std", "judge_sel", "judge_error"}
CHEAP_R = ["judge_logit", "log_tokens", "truncated", "has_decimal", "is_negative", "log_abs_ans", "empty_answer", "log_qlen"]
PCS = [f"pc{i}" for i in range(16)]
JUDGES = {"3B": "Qwen2.5-3B-Instruct", "7B": "Qwen2.5-7B-Instruct-AWQ"}
GATE_AUC = {"3B": 0.654, "7B": 0.702}
ACC_RATE = 0.6421

ap = argparse.ArgumentParser()
ap.add_argument("--rollouts", required=True); ap.add_argument("--feats", required=True); ap.add_argument("--judges", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

r = pd.read_parquet(a.rollouts)[["pid", "k", "answer", "correct", "tokens", "truncated"]]
f = pd.read_parquet(a.feats)[["pid", "log_qlen", "cell30", "cell100", "cell300"] + PCS]
assert len(r) == 24000 and r.groupby("pid").size().eq(8).all() and f.pid.is_unique
r = r.merge(f, on="pid", how="left", validate="many_to_one"); assert r[PCS + ["cell100"]].notna().all().all()
ans = r.answer.fillna("").astype(str)
r["has_decimal"] = ans.str.contains(r"\.", regex=True).astype(int)
r["is_negative"] = ans.str.startswith("-").astype(int)
r["empty_answer"] = (ans == "").astype(int)
num = pd.to_numeric(ans, errors="coerce"); assert num[ans != ""].notna().all(), "unparseable non-empty answer"
r["log_abs_ans"] = np.log10(num.abs().fillna(0) + 1)
r["log_tokens"] = np.log(r.tokens + 1)

meta = {"accept_rate_scalar": ACC_RATE, "rollout_accuracy": float(r.correct.mean()), "tables": {}, "gate": {}}
ok = True
for jk, name in JUDGES.items():
    j = pd.read_parquet(os.path.join(a.judges, f"judge_{name}.parquet"))
    assert not j.duplicated(["pid", "answer"]).any()
    m = r.merge(j, on=["pid", "answer"], how="left", validate="many_to_one"); assert m.judge_p.notna().all(), "unmatched (pid, answer)"
    m["judge_logit"] = np.log(m.judge_p.clip(1e-6, 1 - 1e-6) / (1 - m.judge_p.clip(1e-6, 1 - 1e-6)))
    auc = float(roc_auc_score(1 - m.correct, 1 - m.judge_p))  # rollout-level AUC of (1 - judge_p) for wrong, all 24,000 rollouts (E04 validity convention)
    gate_auc_ok = round(auc, 3) == GATE_AUC[jk]; meta["gate"][jk] = {"rollout_auc_judge_p_vs_wrong": auc, "expected": GATE_AUC[jk], "reproduces_3dp": bool(gate_auc_ok)}
    ok &= gate_auc_ok
    rules = {"q": float(np.quantile(m.judge_p, 1 - ACC_RATE))}
    if jk == "7B": rules["p05"] = 0.5
    for rule, thr in rules.items():
        acc = (m.judge_p >= thr) if rule == "q" else (m.judge_p > thr)
        thr_logit = float(np.log(np.clip(thr, 1e-6, 1 - 1e-6) / (1 - np.clip(thr, 1e-6, 1 - 1e-6))))
        d = m.copy(); d["accepted"] = acc.astype(int)
        d["boundary"] = (d.judge_logit - thr_logit).abs()  # distance to the accept boundary in logit space, label-free (global scalar threshold, per-rollout value)
        d["corrupted"] = (acc & (d.correct == 0)).astype(int); d["judge_wrong"] = (acc.astype(int) != d.correct).astype(int)
        for lab, pop in (("corrupted", d[d.accepted == 1]), ("judgewrong", d)):
            col = "corrupted" if lab == "corrupted" else "judge_wrong"
            keep = ["pid", "k", col, "judge_p", "tokens", "boundary"] + CHEAP_R + ["cell30", "cell100", "cell300"] + PCS
            t = pop[keep].reset_index(drop=True); assert not (set(t.columns) & FORBIDDEN), set(t.columns) & FORBIDDEN
            tag = f"{jk}_{rule}_{lab}"; t.to_parquet(os.path.join(a.out, f"tab_{tag}.parquet"), index=False)
            npos = int(t[col].sum()); gate_n = npos >= 200; ok &= gate_n
            meta["tables"][tag] = {"label_col": col, "rows": len(t), "positives": npos, "rate": float(t[col].mean()), "gate_ge_200": bool(gate_n), "threshold_judge_p": float(thr), "threshold_logit": thr_logit,
                                   "accepted": int(acc.sum()), "pids": int(t.pid.nunique())}
            print(f"{tag}: rows {len(t)} positives {npos} rate {t[col].mean():.4f} thr {thr:.6g} accepted {int(acc.sum())}", flush=True)
meta["gate_passed"] = bool(ok); json.dump(meta, open(os.path.join(a.out, "build.json"), "w"), indent=1)
print("GATE", "PASS" if ok else "FAIL", {k: (round(v["rollout_auc_judge_p_vs_wrong"], 4), v["reproduces_3dp"]) for k, v in meta["gate"].items()})
sys.exit(0 if ok else 3)
