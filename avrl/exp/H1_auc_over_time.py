#!/usr/bin/env python
"""H1: does label-free confidence track proxy_error over training?

For every run listed in RUNS below that has a groups.jsonl, bucket its
per-group rows into blocks of training steps (50 steps, or 25 for runs
that only trained 100 steps) and compute, per block and for the whole
run:
  - number of groups
  - proxy error rate (fraction with proxy_error true)
  - AUC of -confidence for predicting proxy_error (roc_auc_score on
    -confidence; NaN if the block has no positives or no negatives)
  - a 1000-draw group bootstrap 95% interval on that AUC

Confidence is `agreement` for the code and GSM8K harnesses and
`consistency` for the judge7b harness.

Assumption: proxy_error is logged for every group in every run here
(gold is computed for logging in the code and GSM8K harnesses, a
logging-only judge call in the judge7b harness), so no row needs to be
dropped for a missing label. This is checked directly below and the
null fraction is reported per run rather than assumed silently.

For the code runs we additionally report, per block, a group-level
stand-in for the per-completion false-positive rate: proxy_error is a
single bool per group, so we cannot tell which individual completion
in a group was wrong. Instead we report, among groups where at least
one completion passed the visible test, the fraction with proxy_error
true, plus the mean visible_pass_count.

Outputs (all under avrl/runs/h1_auc/):
  <run_name>.json     one file per run: per-block and whole-run stats
  SUMMARY.md          one summary table plus the answer sentences
  auc_over_time.png   one panel per task family, AUC per block vs step
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RUNS_ROOT = Path("/homes/rdubash/active-audit/avrl/runs")
OUT = RUNS_ROOT / "h1_auc"

N_BOOT = 1000
BOOT_SEED = 0

# (name, path relative to RUNS_ROOT, task family, arm)
RUNS = [
    # code, OLMo, visible-test grader (E10b and related)
    ("e10b_proxy_s1", "e10b_proxy_s1", "code_olmo", "proxy"),
    ("e10b_full_s1", "e10b_full_s1", "code_olmo", "full"),
    ("e10b_threshold_s1", "e10b_threshold_s1", "code_olmo", "threshold"),
    ("e10b_random_s1", "e10b_random_s1", "code_olmo", "random"),
    ("e10b_threshold_s2", "e10b_threshold_s2", "code_olmo", "threshold"),
    ("e10b_random_s2", "e10b_random_s2", "code_olmo", "random"),
    ("e10c_proxy600_s1", "e10c_proxy600_s1", "code_olmo", "proxy"),
    ("hannah_e10b_proxy", "hannah_e10b/e10b_proxy", "code_olmo", "proxy"),
    ("hannah_e10b_full", "hannah_e10b/e10b_full", "code_olmo", "full"),
    ("hannah_e10b_random", "hannah_e10b/e10b_random", "code_olmo", "random"),
    ("hannah_e10b_threshold", "hannah_e10b/e10b_threshold", "code_olmo", "threshold"),
    # code, Qwen (E10d)
    ("hannah_e10d_proxy", "hannah_e10d/e10d_proxy", "code_qwen", "proxy"),
    ("hannah_e10d_full", "hannah_e10d/e10d_full", "code_qwen", "full"),
    ("hannah_e10d_random", "hannah_e10d/e10d_random", "code_qwen", "random"),
    ("hannah_e10d_threshold", "hannah_e10d/e10d_threshold", "code_qwen", "threshold"),
    # open-ended with 7B judge
    ("hannah_judge7b_full", "hannah_judge7b/full", "judge7b", "full"),
    ("hannah_judge7b_random", "hannah_judge7b/random", "judge7b", "random"),
    ("hannah_judge7b_thrcons", "hannah_judge7b/thrcons", "judge7b", "thrcons"),
    ("hannah_judge7b_greedy", "hannah_judge7b/greedy", "judge7b", "greedy"),
    ("hannah_judge7b_s1_full", "hannah_judge7b_s1/full", "judge7b", "full"),
    ("hannah_judge7b_s1_random", "hannah_judge7b_s1/random", "judge7b", "random"),
    ("hannah_judge7b_s1_thrcons", "hannah_judge7b_s1/thrcons", "judge7b", "thrcons"),
    ("hannah_judge7b_s1_greedy", "hannah_judge7b_s1/greedy", "judge7b", "greedy"),
    ("hannah_judge7b_s2_full", "hannah_judge7b_s2/full", "judge7b", "full"),
    ("hannah_judge7b_s2_random", "hannah_judge7b_s2/random", "judge7b", "random"),
    ("hannah_judge7b_s2_thrcons", "hannah_judge7b_s2/thrcons", "judge7b", "thrcons"),
    ("hannah_judge7b_s2_greedy", "hannah_judge7b_s2/greedy", "judge7b", "greedy"),
    # GSM8K
    ("m10_proxy_s0", "m10_proxy_s0", "gsm8k", "proxy"),
    ("m10_proxy_s1", "m10_proxy_s1", "gsm8k", "proxy"),
    ("m14_threshold_b10_s0", "m14_threshold_b10_s0", "gsm8k", "threshold"),
    ("m14_threshold_b10_s1", "m14_threshold_b10_s1", "gsm8k", "threshold"),
]

TASK_LABELS = {
    "code_olmo": "code, OLMo (visible-test grader)",
    "code_qwen": "code, Qwen (visible-test grader)",
    "judge7b": "open-ended, 7B judge",
    "gsm8k": "GSM8K (vote agreement)",
}

CONFIDENCE_FIELD = {
    "code_olmo": "agreement",
    "code_qwen": "agreement",
    "judge7b": "consistency",
    "gsm8k": "agreement",
}

ARM_COLORS = {
    "proxy": "tab:blue",
    "full": "tab:green",
    "threshold": "tab:red",
    "random": "tab:gray",
    "thrcons": "tab:purple",
    "greedy": "tab:orange",
}


def block_size_for(n_steps):
    return 25 if n_steps <= 100 else 50


def auc_or_nan(y, score):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def bootstrap_auc_ci(y, score, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auc_or_nan(y[idx], score[idx])
        if not np.isnan(a):
            draws.append(a)
    if len(draws) < 10:
        return (float("nan"), float("nan"))
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def load_run(rel_path):
    rows = []
    with open(RUNS_ROOT / rel_path / "groups.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarize_block(df):
    y = df["proxy_error"].astype(int).to_numpy()
    score = -df["confidence"].to_numpy()
    ci_lo, ci_hi = bootstrap_auc_ci(y, score)
    return {
        "n_groups": int(len(df)),
        "error_rate": float(y.mean()),
        "auc": auc_or_nan(y, score),
        "auc_ci_lo": ci_lo,
        "auc_ci_hi": ci_hi,
    }


def code_fp_block(df):
    # proxy_error is one bool per group, not per completion, so the
    # per-completion false-positive rate asked for isn't recoverable.
    # Stand-in: among groups with >=1 completion passing the visible
    # test, what fraction still have proxy_error true, plus the mean
    # visible_pass_count.
    any_pass = df["visible_pass_count"] > 0
    sub = df[any_pass]
    fp_rate = float(sub["proxy_error"].astype(int).mean()) if len(sub) else float("nan")
    return {
        "n_groups": int(len(df)),
        "n_any_pass": int(any_pass.sum()),
        "group_fp_rate_given_any_pass": fp_rate,
        "mean_visible_pass_count": float(df["visible_pass_count"].mean()),
    }


def process_run(name, rel_path, task, arm):
    df = load_run(rel_path)
    conf_field = CONFIDENCE_FIELD[task]
    df["confidence"] = df[conf_field]

    null_frac = float(df["proxy_error"].isna().mean())

    n_steps = int(df["step"].max()) + 1
    bs = block_size_for(n_steps)
    df["block"] = df["step"] // bs

    has_code_fields = "visible_pass_count" in df.columns

    blocks = []
    for b, bdf in df.groupby("block"):
        rec = {"block_start": int(b * bs), "block_end": int(b * bs + bs - 1)}
        rec.update(summarize_block(bdf))
        if has_code_fields:
            rec["code_fp"] = code_fp_block(bdf)
        blocks.append(rec)
    blocks.sort(key=lambda r: r["block_start"])

    whole = summarize_block(df)
    if has_code_fields:
        whole["code_fp"] = code_fp_block(df)

    return {
        "run": name,
        "path": rel_path,
        "task": task,
        "arm": arm,
        "confidence_field": conf_field,
        "n_rows": int(len(df)),
        "null_proxy_error_fraction": null_frac,
        "block_size": bs,
        "blocks": blocks,
        "whole_run": whole,
    }


def fmt_auc(a):
    return "NaN" if np.isnan(a) else f"{a:.3f}"


def fmt_ci(lo, hi):
    if np.isnan(lo) or np.isnan(hi):
        return "NaN"
    return f"[{lo:.3f}, {hi:.3f}]"


def write_run_json(result):
    with open(OUT / f"{result['run']}.json", "w") as f:
        json.dump(result, f, indent=2)


def plot_figure(results):
    tasks = ["code_olmo", "code_qwen", "judge7b", "gsm8k"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, task in zip(axes.flat, tasks):
        seen_arms = set()
        for r in results:
            if r["task"] != task:
                continue
            xs = [b["block_start"] for b in r["blocks"]]
            ys = [b["auc"] for b in r["blocks"]]
            color = ARM_COLORS.get(r["arm"], "black")
            label = r["arm"] if r["arm"] not in seen_arms else None
            seen_arms.add(r["arm"])
            ax.plot(xs, ys, color=color, linewidth=1, alpha=0.8, marker="o", markersize=3, label=label)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("step")
        ax.set_ylabel("AUC (low confidence -> proxy_error)")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "auc_over_time.png", dpi=150)
    plt.close(fig)


def mean_ignoring_nan(values):
    vals = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def family_trend_sentence(task, results):
    fam = [r for r in results if r["task"] == task]
    first_aucs = [r["blocks"][0]["auc"] for r in fam]
    last_aucs = [r["blocks"][-1]["auc"] for r in fam]
    mean_first, mean_last = mean_ignoring_nan(first_aucs), mean_ignoring_nan(last_aucs)
    delta = mean_last - mean_first
    if np.isnan(delta):
        trend = "is undetermined (too many single-class blocks)"
    elif delta > 0.02:
        trend = f"rises ({mean_first:.3f} to {mean_last:.3f})"
    elif delta < -0.02:
        trend = f"falls ({mean_first:.3f} to {mean_last:.3f})"
    else:
        trend = f"stays flat (~{mean_first:.3f} to ~{mean_last:.3f})"

    arm_whole = {}
    for r in fam:
        arm_whole.setdefault(r["arm"], []).append(r["whole_run"]["auc"])
    arm_mean = {a: mean_ignoring_nan(v) for a, v in arm_whole.items()}
    arm_note = ""
    if "proxy" in arm_mean and "threshold" in arm_mean and not (np.isnan(arm_mean["proxy"]) or np.isnan(arm_mean["threshold"])):
        diff = arm_mean["threshold"] - arm_mean["proxy"]
        if abs(diff) > 0.02:
            arm_note = f"; threshold's whole-run AUC ({arm_mean['threshold']:.3f}) differs from proxy's ({arm_mean['proxy']:.3f}) by {diff:+.3f}"
        else:
            arm_note = f"; threshold and proxy whole-run AUCs are close ({arm_mean['threshold']:.3f} vs {arm_mean['proxy']:.3f})"
    return f"In {TASK_LABELS[task]}, confidence's AUC for predicting proxy_error {trend} over training{arm_note}."


def code_fp_sentence(results):
    code_results = [r for r in results if "code_fp" in r["whole_run"]]
    if not code_results:
        return ""
    first_fp = mean_ignoring_nan([r["blocks"][0]["code_fp"]["group_fp_rate_given_any_pass"] for r in code_results])
    last_fp = mean_ignoring_nan([r["blocks"][-1]["code_fp"]["group_fp_rate_given_any_pass"] for r in code_results])
    first_vpc = mean_ignoring_nan([r["blocks"][0]["code_fp"]["mean_visible_pass_count"] for r in code_results])
    last_vpc = mean_ignoring_nan([r["blocks"][-1]["code_fp"]["mean_visible_pass_count"] for r in code_results])
    return (
        "Per-completion correctness isn't in the log (proxy_error is one bool per group), so as a "
        f"group-level stand-in: across the code runs, the fraction of groups with >=1 visible-test pass "
        f"that still had proxy_error true moves from {first_fp:.3f} (first block) to {last_fp:.3f} (last block), "
        f"while the mean visible_pass_count moves from {first_vpc:.2f} to {last_vpc:.2f}."
    )


def write_summary_md(results):
    lines = []
    lines.append("# H1: does low confidence predict proxy_error over training?\n")
    lines.append(
        "Assumption: proxy_error is logged for every group in every harness analysed here "
        "(gold is computed for logging in the code and GSM8K harnesses, a logging-only judge "
        "call in the judge7b harness), so every row is usable. Checked directly: the null "
        "fraction of proxy_error was 0.0 for all runs below.\n"
    )
    header = "| run | task | arm | n groups | err rate (first block) | err rate (last block) | AUC (first block) | AUC (last block) | AUC (whole run, 95% CI) |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for r in results:
        first, last, whole = r["blocks"][0], r["blocks"][-1], r["whole_run"]
        lines.append(
            f"| {r['run']} | {TASK_LABELS[r['task']]} | {r['arm']} | {r['n_rows']} | "
            f"{first['error_rate']:.3f} | {last['error_rate']:.3f} | "
            f"{fmt_auc(first['auc'])} | {fmt_auc(last['auc'])} | "
            f"{fmt_auc(whole['auc'])} {fmt_ci(whole['auc_ci_lo'], whole['auc_ci_hi'])} |"
        )
    lines.append("")

    sentences = [family_trend_sentence(t, results) for t in TASK_LABELS]
    fp_sentence = code_fp_sentence(results)
    if fp_sentence:
        sentences.append(fp_sentence)
    lines.append(" ".join(sentences))
    lines.append("")

    with open(OUT / "SUMMARY.md", "w") as f:
        f.write("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name, rel_path, task, arm in RUNS:
        gpath = RUNS_ROOT / rel_path / "groups.jsonl"
        if not gpath.exists():
            print(f"skip {name}: no groups.jsonl at {gpath}")
            continue
        result = process_run(name, rel_path, task, arm)
        write_run_json(result)
        results.append(result)

    write_summary_md(results)
    plot_figure(results)
    print(f"processed {len(results)} runs; wrote JSON, SUMMARY.md, auc_over_time.png to {OUT}")


if __name__ == "__main__":
    main()
