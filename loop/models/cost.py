"""Summarize a calls.jsonl audit log by served model, requested model and tag prefix, and count fallbacks.

calls.jsonl is the log call.py appends to: one JSON line per try, with the model asked for (requested_model), the
model that answered (served_model; the two differ after a fallback hop), token usage, two dollar figures and latency.
cost_usd_plan is the cost the claude CLI reports (0.0 for vllm models, which run on lab GPUs); cost_usd_nominal
prices vllm tokens at the registry's nominal rate and equals the plan cost for Claude.

Entry point: `summarize(log_path)` returns the grouped totals as a dict; run as a script it prints three tables.
Rows count tries, not logical calls: a call that needed three tries contributes three to `attempts`.
Usage: python cost.py [calls.jsonl]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def _new():
    """Empty accumulator for one group (a model, a tag prefix, or the grand total)."""
    return {"attempts": 0, "ok": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
            "cost_usd_plan": 0.0, "cost_usd_nominal": 0.0, "latency_sum": 0.0}


def _add(s, r):
    """Add one log record r to accumulator s. Missing or null fields count as zero."""
    s["attempts"] += 1
    s["ok"] += bool(r.get("ok"))
    for k, v in (r.get("usage") or {}).items():
        if k in s:  # only the four token kinds the accumulator tracks
            s[k] += v or 0
    s["cost_usd_plan"] += r.get("cost_usd_plan") or 0.0
    s["cost_usd_nominal"] += r.get("cost_usd_nominal") or 0.0
    s["latency_sum"] += r.get("latency_s") or 0.0


def _finish(s):
    """Replace the running latency sum with mean_latency_s (seconds per try). Returns s."""
    s["mean_latency_s"] = s.pop("latency_sum") / s["attempts"] if s["attempts"] else 0.0
    return s


def summarize(log_path) -> dict:
    """One log line is one attempt. Returns {"total", "by_served_model", "by_requested_model", "by_tag", "fallbacks",
    "bad_lines"} (tag grouped by text before the first ':'; "by_model" is the same dict as "by_served_model").
    Lines from before fallbacks existed have only model_name, which stands in for both. fallbacks counts attempts
    served by a model other than the requested one: {"attempts": n, "by_route": {"requested->served": n}}."""
    total, by_served, by_req, by_tag = _new(), defaultdict(_new), defaultdict(_new), defaultdict(_new)
    routes = defaultdict(int)
    bad = 0
    with open(log_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            # Older log lines carry only model_name; it stands in for both the served and the requested model.
            served = r.get("served_model") or r.get("model_name", "?")
            req = r.get("requested_model") or r.get("model_name", "?")
            if served != req:
                routes[f"{req}->{served}"] += 1
            # Each try is counted once in every grouping. Tags look like "judge:r1"; the part before ":" is the group.
            for s in (total, by_served[served], by_req[req], by_tag[(r.get("tag") or "").split(":")[0] or "(none)"]):
                _add(s, r)
    served_out = {k: _finish(v) for k, v in by_served.items()}  # one dict, returned under two keys (see docstring)
    return {"total": _finish(total), "by_model": served_out, "by_served_model": served_out,
            "by_requested_model": {k: _finish(v) for k, v in by_req.items()},
            "by_tag": {k: _finish(v) for k, v in by_tag.items()},
            "fallbacks": {"attempts": sum(routes.values()), "by_route": dict(routes)}, "bad_lines": bad}


def _table(title, groups):
    """Fixed-width text table: one row per group with tries, ok count, tokens, both costs and mean latency."""
    rows = [f"{title:<18}{'att':>5}{'ok':>5}{'in':>9}{'out':>8}{'plan$':>10}{'nominal$':>10}{'lat_s':>8}"]
    for name, s in groups.items():
        rows.append(f"{name:<18}{s['attempts']:>5}{s['ok']:>5}{s['input']:>9}{s['output']:>8}"
                    f"{s['cost_usd_plan']:>10.4f}{s['cost_usd_nominal']:>10.4f}{s['mean_latency_s']:>8.2f}")
    return "\n".join(rows)


if __name__ == "__main__":
    r = summarize(sys.argv[1] if len(sys.argv) > 1 else "calls.jsonl")
    print(_table("served model", {**r["by_served_model"], "TOTAL": r["total"]}))
    print()
    print(_table("requested model", r["by_requested_model"]))
    print()
    print(_table("tag", r["by_tag"]))
    print(f"\nfallback attempts: {r['fallbacks']['attempts']} {r['fallbacks']['by_route']}")
    if r["bad_lines"]:
        print(f"\n{r['bad_lines']} unparseable log lines skipped")
