"""S1 (scale-up) analysis: applies the pre-registered tests in EXPERIMENT_BACKLOG.md (entry S1, revised
2026-10-05) to the arm logs. Usage:
  .venv/bin/python avrl/exp/S1_analyze.py --runs_dir avrl/runs/hannah_s1 --out avrl/runs/s1_analysis
Each arm directory holds log.jsonl with train_step rows (corrupted_reward_groups, expensive_calls, n_groups,
rm_score_mean) and eval rows (eval_judge_score_mean) every 200 steps. Missing arms, incomplete runs and a
missing rm_score_mean field are reported, not fatal."""
import argparse, json, os
import numpy as np

ARMS = ["s1_proxy_s0", "s1_thrcons_s0", "s1_full_s0", "s1_proxy_lr3_s0", "s1_thrcons_s1", "s1_full_s1"]
PROXY_ARMS = ["s1_proxy_s0", "s1_proxy_lr3_s0"]
WINDOW = 200
EXPECTED_STEPS = 4000
MIN_STEPS = 600
MIN_FIRST_CORRUPTED = 10
DRIFT_RATIO = 1.5
OVEROPT_DROP = 0.15
CLAIM1_GAP = 0.15


def load(run_dir):
    path = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ts = [r for r in rows if r.get("type") == "train_step"]
    ev = [r for r in rows if r.get("type") == "eval"]
    return dict(ts=ts, ev=ev)


def window_stats(ts):
    c = sum(r["corrupted_reward_groups"] for r in ts)
    u = sum(r["n_groups"] - r["expensive_calls"] for r in ts)
    rm = [r["rm_score_mean"] for r in ts if r.get("rm_score_mean") is not None]
    return dict(corrupted=c, unverified=u, rate=(c / u if u else None), rm=(float(np.mean(rm)) if rm else None))


def blocks(ts):
    return [window_stats(ts[i:i + WINDOW]) for i in range(0, len(ts), WINDOW)]


def drift_test(ts, rng):
    n = len(ts)
    if n < MIN_STEPS:
        return dict(verdict="VOID", reason=f"only {n} steps completed (needs {MIN_STEPS})")
    first, last = ts[:WINDOW], ts[-WINDOW:]
    f, l = window_stats(first), window_stats(last)
    if f["corrupted"] < MIN_FIRST_CORRUPTED:
        return dict(verdict="VOID", reason=f"first window has {f['corrupted']} corrupted groups (needs {MIN_FIRST_CORRUPTED})",
                    first_rate=f["rate"], last_rate=l["rate"])
    ratio = l["rate"] / f["rate"]
    boots = []
    for _ in range(1000):
        fb = window_stats([first[i] for i in rng.integers(0, len(first), len(first))])
        lb = window_stats([last[i] for i in rng.integers(0, len(last), len(last))])
        if fb["rate"]:
            boots.append(lb["rate"] / fb["rate"])
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (None, None)
    return dict(verdict="PRESENT" if ratio >= DRIFT_RATIO else "ABSENT", first_rate=f["rate"], last_rate=l["rate"],
                ratio=ratio, ratio_ci=[lo, hi], first_corrupted=f["corrupted"], last_corrupted=l["corrupted"])


def overopt_test(ts, ev):
    if len(ev) < 3:
        return dict(verdict="VOID", reason=f"only {len(ev)} eval rows (needs 3)")
    scores = [e["eval_judge_score_mean"] for e in ev]
    peak_i = int(np.argmax(scores))
    last_two = float(np.mean(scores[-2:]))
    drop = scores[peak_i] - last_two
    f, l = window_stats(ts[:WINDOW]), window_stats(ts[-WINDOW:])
    if f["rm"] is None or l["rm"] is None:
        return dict(verdict="VOID", reason="rm_score_mean unavailable in this log", peak_step=ev[peak_i]["step"],
                    peak=scores[peak_i], last_two_mean=last_two, drop=drop)
    rising = l["rm"] > f["rm"]
    present = drop >= OVEROPT_DROP and rising
    return dict(verdict="PRESENT" if present else "ABSENT", peak_step=ev[peak_i]["step"], peak=scores[peak_i],
                last_two_mean=last_two, drop=drop, rm_first=f["rm"], rm_last=l["rm"], rm_rising=bool(rising))


def claim1_test(data):
    gaps = {}
    for seed in ("s0", "s1"):
        t, f = data.get(f"s1_thrcons_{seed}"), data.get(f"s1_full_{seed}")
        if t and f and t["ev"] and f["ev"] and len(t["ts"]) >= EXPECTED_STEPS and len(f["ts"]) >= EXPECTED_STEPS:
            gaps[seed] = f["ev"][-1]["eval_judge_score_mean"] - t["ev"][-1]["eval_judge_score_mean"]
    if not gaps:
        return dict(verdict="VOID", reason="no seed has both the threshold and full arms complete at EXPECTED_STEPS")
    mean_gap = float(np.mean(list(gaps.values())))
    return dict(verdict="PASS" if mean_gap <= CLAIM1_GAP else "FAIL", gaps_full_minus_threshold=gaps,
                mean_gap=mean_gap, n_seeds=len(gaps))


def summary_rows(data):
    rows = []
    for arm in ARMS:
        d = data.get(arm)
        if not d:
            rows.append(dict(arm=arm, status="missing")); continue
        ts, ev = d["ts"], d["ev"]
        all_s = window_stats(ts) if ts else None
        f = window_stats(ts[:WINDOW]) if ts else None
        l = window_stats(ts[-WINDOW:]) if len(ts) >= WINDOW else None
        rows.append(dict(arm=arm, status="complete" if len(ts) >= EXPECTED_STEPS else f"incomplete ({len(ts)} steps)", steps=len(ts),
                         expensive_calls=sum(r["expensive_calls"] for r in ts), corrupted=all_s["corrupted"] if all_s else None,
                         unverified=all_s["unverified"] if all_s else None, rate=all_s["rate"] if all_s else None,
                         rate_first=f["rate"] if f else None, rate_last=l["rate"] if l else None,
                         rm_first=f["rm"] if f else None, rm_last=l["rm"] if l else None,
                         evals=[(e["step"], round(e["eval_judge_score_mean"], 3)) for e in ev],
                         final_eval=(ev[-1]["eval_judge_score_mean"] if ev else None)))
    return rows


def fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="/homes/rdubash/active-audit/avrl/runs/hannah_s1")
    ap.add_argument("--out", default="/homes/rdubash/active-audit/avrl/runs/s1_analysis")
    ap.add_argument("--base_result", default="/homes/rdubash/active-audit/avrl/runs/s1_base_eval/result.json", help="untrained policy score on the same evaluation (reference row)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    data = {}
    for arm in ARMS:
        d = load(os.path.join(a.runs_dir, arm))
        if d is None:
            print(f"SKIP {arm}: no log.jsonl")
        else:
            data[arm] = d
    result = dict(drift={}, overoptimization={}, claim1=claim1_test(data), claim2="not run (no greedy arm in S1)")
    for arm in PROXY_ARMS:
        if arm in data:
            result["drift"][arm] = drift_test(data[arm]["ts"], rng)
            result["overoptimization"][arm] = overopt_test(data[arm]["ts"], data[arm]["ev"])
    result["blocks"] = {arm: blocks(d["ts"]) for arm, d in data.items()}
    result["summary"] = summary_rows(data)
    with open(os.path.join(a.out, "s1_result.json"), "w") as f:
        json.dump(result, f, indent=1, default=float)

    lines = ["# S1 result", ""]
    base = None
    if os.path.exists(a.base_result):
        base = json.load(open(a.base_result))
        lines.append(f"- Untrained policy on the same evaluation: judge score {fmt(base.get('judge_score_mean'))}, positive rate {fmt(base.get('gold_positive_rate'))}")
        result["base"] = base
    for arm in PROXY_ARMS:
        dr = result["drift"].get(arm); ov = result["overoptimization"].get(arm)
        if dr is None:
            lines.append(f"- DRIFT, {arm}: arm missing"); continue
        lines.append(f"- DRIFT, {arm}: {dr['verdict']}" + (f" (ratio {fmt(dr.get('ratio'))}, interval {dr.get('ratio_ci')}, first {fmt(dr.get('first_rate'))}, last {fmt(dr.get('last_rate'))})" if dr["verdict"] != "VOID" else f" ({dr['reason']})"))
        lines.append(f"- OVEROPTIMIZATION, {arm}: {ov['verdict']}" + (f" (peak {fmt(ov.get('peak'))} at step {ov.get('peak_step')}, last two {fmt(ov.get('last_two_mean'))}, drop {fmt(ov.get('drop'))}, rm first {fmt(ov.get('rm_first'))} last {fmt(ov.get('rm_last'))})" if ov["verdict"] != "VOID" else f" ({ov['reason']})"))
    c1 = result["claim1"]
    lines.append(f"- CLAIM 1 AT SCALE: {c1['verdict']}" + (f" (mean gap full minus threshold {fmt(c1.get('mean_gap'))} over {c1.get('n_seeds')} seed(s): {c1.get('gaps_full_minus_threshold')})" if c1["verdict"] != "VOID" else f" ({c1['reason']})"))
    lines.append(f"- CLAIM 2 AT SCALE: {result['claim2']}")
    lines += ["", "| arm | status | steps | calls | corrupted / unverified | rate | rate first 200 | rate last 200 | rm first | rm last | evals (step, score) | final |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["summary"]:
        if r["status"] == "missing":
            lines.append(f"| {r['arm']} | missing | | | | | | | | | | |"); continue
        lines.append(f"| {r['arm']} | {r['status']} | {r['steps']} | {r['expensive_calls']} | {fmt(r['corrupted'])} / {fmt(r['unverified'])} | {fmt(r['rate'])} | {fmt(r['rate_first'])} | {fmt(r['rate_last'])} | {fmt(r['rm_first'])} | {fmt(r['rm_last'])} | {r['evals']} | {fmt(r['final_eval'])} |")
    with open(os.path.join(a.out, "S1_RESULT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150)
    for arm, d in data.items():
        b = result["blocks"][arm]
        x = [(i + 1) * WINDOW for i in range(len(b))]
        rates = [s["rate"] if s["rate"] is not None else np.nan for s in b]
        rms = [s["rm"] if s["rm"] is not None else np.nan for s in b]
        axes[0].plot(x, rates, marker="o", label=arm)
        axes[1].plot(x, rms, marker="o", label=arm)
        if d["ev"]:
            axes[2].plot([e["step"] for e in d["ev"]], [e["eval_judge_score_mean"] for e in d["ev"]], marker="o", label=arm)
    axes[0].set_title("corrupted-group rate per 200-step block"); axes[1].set_title("mean reward-model score per block")
    axes[2].set_title("held-out judge score"); [ax.set_xlabel("training step") for ax in axes]; axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "s1_signals.png"))
    print(f"wrote {a.out}/S1_RESULT.md, s1_result.json, s1_signals.png")


if __name__ == "__main__":
    main()
