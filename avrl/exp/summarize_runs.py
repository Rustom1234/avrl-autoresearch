"""Summarize in-training runs: steps, corrupted-reward groups, expensive calls,
proxy error drift (first vs last quarter of steps), and the eval trajectory.
Usage: .venv/bin/python avrl/exp/summarize_runs.py avrl/runs/m10_full_s0 [more run dirs]
Works on any harness log.jsonl (p1_grpo, p1_judge_grpo, p1_code_grpo)."""
import json, os, sys

EVAL_KEYS = ("eval_accuracy", "eval_judge_score_mean", "eval_mbpp_pass1", "eval_humaneval_pass1")


def summarize(run_dir):
    path = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(path):
        return f"{run_dir}: no log.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ts = [r for r in rows if r.get("type") == "train_step"]
    ev = [r for r in rows if r.get("type") in ("eval", "eval_only")]
    if not ts:
        if ev:
            e = ev[-1]
            keys = [k for k in e if k.startswith("eval_") and isinstance(e[k], (int, float))]
            return f"{run_dir}: reference eval, " + ", ".join(f"{k} {e[k]:.3f}" for k in keys)
        return f"{run_dir}: no train_step rows"
    n = len(ts)
    corrupted = sum(r.get("corrupted_reward_groups", 0) for r in ts)
    calls = sum(r.get("expensive_calls", 0) for r in ts)
    unverified = sum(r.get("n_groups", 0) - r.get("expensive_calls", 0) for r in ts)
    errs = [r.get("proxy_error_rate_among_unverified") for r in ts]
    errs = [e for e in errs if e is not None]
    q = max(1, len(errs) // 4)
    first = sum(errs[:q]) / q if errs else float("nan")
    last = sum(errs[-q:]) / q if errs else float("nan")
    keys = [k for k in EVAL_KEYS if any(k in e for e in ev)]
    key = ",".join(keys) if keys else None
    traj = [(e.get("step"),) + tuple(round(e[k], 3) if k in e else None for k in keys) for e in ev] if keys else []
    return (f"{run_dir}: {n} steps, corrupted {corrupted}/{unverified} unverified, calls {calls}, "
            f"proxy err first/last quarter {first:.3f}/{last:.3f}, {key or 'eval'}: {traj}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(summarize(d.rstrip("/")))
