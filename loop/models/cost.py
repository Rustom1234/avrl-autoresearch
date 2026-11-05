"""Summarize a calls.jsonl audit log by served model, requested model and tag prefix, and count fallbacks.
Usage: python cost.py [calls.jsonl]"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def _new():
    return {"attempts": 0, "ok": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
            "cost_usd_plan": 0.0, "cost_usd_nominal": 0.0, "latency_sum": 0.0}


def _add(s, r):
    s["attempts"] += 1
    s["ok"] += bool(r.get("ok"))
    for k, v in (r.get("usage") or {}).items():
        if k in s:
            s[k] += v or 0
    s["cost_usd_plan"] += r.get("cost_usd_plan") or 0.0
    s["cost_usd_nominal"] += r.get("cost_usd_nominal") or 0.0
    s["latency_sum"] += r.get("latency_s") or 0.0


def _finish(s):
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
            served = r.get("served_model") or r.get("model_name", "?")
            req = r.get("requested_model") or r.get("model_name", "?")
            if served != req:
                routes[f"{req}->{served}"] += 1
            for s in (total, by_served[served], by_req[req], by_tag[(r.get("tag") or "").split(":")[0] or "(none)"]):
                _add(s, r)
    served_out = {k: _finish(v) for k, v in by_served.items()}
    return {"total": _finish(total), "by_model": served_out, "by_served_model": served_out,
            "by_requested_model": {k: _finish(v) for k, v in by_req.items()},
            "by_tag": {k: _finish(v) for k, v in by_tag.items()},
            "fallbacks": {"attempts": sum(routes.values()), "by_route": dict(routes)}, "bad_lines": bad}


def _table(title, groups):
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
