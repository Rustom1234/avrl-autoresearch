"""Model cost per loop version, counting open-model calls as well as Claude calls.

Claude calls are summed at the list price the CLI reports (cost_usd_plan in calls.jsonl). Open-model calls ran
on owned lab GPUs at no per-call cost, so they are priced two ways for comparison: per token at hosted prices
(loop/models/open_prices.json) and as GPU rental for the time the model was busy. A third figure gives the cost of
renting the open-model GPUs for the whole run window, idle time included.

Usage:
    python -m loop.core.cost_report --run loop/runs/R1 [--calibration loop/runs/calibration]
                                    [--prices loop/models/open_prices.json] [--out <run>/analysis]
Writes cost_report.json and cost_report.md.
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRICES = ROOT / "loop" / "models" / "open_prices.json"


def _ts(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _rows(path):
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def price_calls(rows, prices):
    """Totals for one calls.jsonl: Claude list cost, open-model per-token cost, tokens, busy time, time span."""
    per_m = prices["per_million_tokens"]
    t = {"claude_usd": 0.0, "open_usd": 0.0, "claude_tokens": 0, "open_tokens": 0, "calls": 0, "open_calls": 0,
         "open_busy_s": 0.0, "open_tokens_by_model": defaultdict(int), "t_min": None, "t_max": None}
    for r in rows:
        model = r.get("served_model") or r.get("model_name")
        u = r.get("usage") or {}
        tin = (u.get("input") or 0) + (u.get("cache_read") or 0) + (u.get("cache_write") or 0)
        tout = u.get("output") or 0
        t["calls"] += 1
        if model in per_m:
            p = per_m[model]
            t["open_calls"] += 1
            t["open_tokens"] += tin + tout
            t["open_tokens_by_model"][model] += tin + tout
            t["open_usd"] += tin / 1e6 * p["input"] + tout / 1e6 * p["output"]
            t["open_busy_s"] += r.get("latency_s") or 0.0
        else:
            t["claude_usd"] += r.get("cost_usd_plan") or 0.0
            t["claude_tokens"] += tin + tout
        ts = _ts(r.get("ts"))
        if ts is not None:
            t["t_min"] = ts if t["t_min"] is None else min(t["t_min"], ts)
            t["t_max"] = ts if t["t_max"] is None else max(t["t_max"], ts)
    t["open_tokens_by_model"] = dict(t["open_tokens_by_model"])
    return t


def report(run, prices, calibration=None):
    run = Path(run)
    rate = prices["gpu_rental_per_hour"]
    arms = defaultdict(list)
    t_min, t_max = None, None
    for d in sorted(p for p in run.iterdir() if (p / "state.json").exists()):
        state = json.loads((d / "state.json").read_text())
        c = price_calls(_rows(d / "calls.jsonl"), prices)
        accepts = sum(1 for r in _rows(d / "ledger.jsonl") if str(r.get("decision") or "").startswith("accept"))
        c.update(loop_id=state.get("loop_id", d.name), accepts=accepts)
        arms[state["arm"]].append(c)
        for v in (c["t_min"], c["t_max"]):
            if v is not None:
                t_min = v if t_min is None else min(t_min, v)
                t_max = v if t_max is None else max(t_max, v)

    per_arm = {}
    for arm, rows in sorted(arms.items()):
        n = len(rows)
        claude = sum(r["claude_usd"] for r in rows)
        open_ = sum(r["open_usd"] for r in rows)
        accepts = sum(r["accepts"] for r in rows)
        per_arm[arm] = {
            "runs": n,
            "claude_usd_per_run": claude / n,
            "open_usd_per_run": open_ / n,
            "total_usd_per_run": (claude + open_) / n,
            "calls_per_run": sum(r["calls"] for r in rows) / n,
            "open_calls_per_run": sum(r["open_calls"] for r in rows) / n,
            "claude_tokens_per_run": sum(r["claude_tokens"] for r in rows) / n,
            "open_tokens_per_run": sum(r["open_tokens"] for r in rows) / n,
            "accepts": accepts,
            "total_usd_per_accept": (claude + open_) / accepts if accepts else None,
        }

    allrows = [r for rows in arms.values() for r in rows]
    busy_h = sum(r["open_busy_s"] for r in allrows) / 3600
    window_h = (t_max - t_min) / 3600 if t_min is not None else 0.0
    by_model = defaultdict(int)
    for r in allrows:
        for m, v in r["open_tokens_by_model"].items():
            by_model[m] += v
    whole = {
        "claude_usd": sum(r["claude_usd"] for r in allrows),
        "claude_tokens": sum(r["claude_tokens"] for r in allrows),
        "open_usd_per_token_pricing": sum(r["open_usd"] for r in allrows),
        "open_tokens": sum(r["open_tokens"] for r in allrows),
        "open_tokens_by_model": dict(by_model),
        "open_busy_gpu_hours": busy_h,
        "open_busy_rental_usd": [busy_h * rate["low"], busy_h * rate["high"]],
        "run_window_hours": window_h,
        "dedicated_rental_usd": [window_h * prices["n_open_model_servers"] * rate["low"],
                                 window_h * prices["n_open_model_servers"] * rate["high"]],
    }
    out = {"run": str(run), "prices_source": prices["source"], "per_arm": per_arm, "whole_run": whole}
    if calibration:
        rows = _rows(Path(calibration) / "calls.jsonl")
        by = defaultdict(list)
        for r in rows:
            by[r.get("served_model") or r.get("model_name")].append(r)
        cal = {m: price_calls(rs, prices) for m, rs in sorted(by.items())}
        busy = sum(c["open_busy_s"] for c in cal.values()) / 3600
        out["calibration"] = {
            "claude_usd": sum(c["claude_usd"] for c in cal.values()),
            "open_usd_per_token_pricing": sum(c["open_usd"] for c in cal.values()),
            "open_busy_rental_usd": [busy * rate["low"], busy * rate["high"]],
            "by_model": {m: {"claude_usd": c["claude_usd"], "open_usd": c["open_usd"], "tokens": c["claude_tokens"] + c["open_tokens"]}
                         for m, c in cal.items()},
        }
    return out


def to_markdown(r):
    lines = ["# Model cost per loop version", "",
             "Claude at list price from the CLI; open-model calls estimated per token at hosted prices (%s)." % r["prices_source"], "",
             "| arm | runs | Claude USD/run | open-model USD/run (est.) | total USD/run | calls/run | open calls/run | accepts | total USD per accepted change |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm, a in r["per_arm"].items():
        per_acc = "n/a" if a["total_usd_per_accept"] is None else "%.3f" % a["total_usd_per_accept"]
        lines.append("| %s | %d | %.3f | %.3f | %.3f | %.1f | %.1f | %d | %s |" % (
            arm, a["runs"], a["claude_usd_per_run"], a["open_usd_per_run"], a["total_usd_per_run"],
            a["calls_per_run"], a["open_calls_per_run"], a["accepts"], per_acc))
    w = r["whole_run"]
    lines += ["", "Whole run: Claude %.2f USD for %s tokens; open models %s tokens, %.2f USD at hosted per-token prices." % (
                  w["claude_usd"], format(w["claude_tokens"], ","), format(w["open_tokens"], ","), w["open_usd_per_token_pricing"]),
              "Open-model GPUs busy %.2f hours: %.2f to %.2f USD if rented for that time." % (
                  w["open_busy_gpu_hours"], *w["open_busy_rental_usd"]),
              "Renting the open-model GPUs for the whole %.1f-hour run window, idle time included: %.2f to %.2f USD." % (
                  w["run_window_hours"], *w["dedicated_rental_usd"])]
    if "calibration" in r:
        c = r["calibration"]
        lines += ["", "Calibration tryouts: Claude %.2f USD; open models %.2f USD at hosted per-token prices (%.2f to %.2f USD as GPU rental for busy time)." % (
            c["claude_usd"], c["open_usd_per_token_pricing"], *c["open_busy_rental_usd"])]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--prices", default=str(DEFAULT_PRICES))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    prices = json.loads(Path(a.prices).read_text())
    r = report(a.run, prices, a.calibration)
    out = Path(a.out or Path(a.run) / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    (out / "cost_report.json").write_text(json.dumps(r, indent=1))
    (out / "cost_report.md").write_text(to_markdown(r))
    print(to_markdown(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
