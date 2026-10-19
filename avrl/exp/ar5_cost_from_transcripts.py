"""AR5: per-run, per-role, per-model token and cost accounting from workflow agent transcripts (v2).

Each workflow agent leaves agent-<id>.jsonl (assistant messages carry message.id,
message.model and message.usage; streamed chunks repeat the same id, so usage is
deduplicated by message id, keeping the last record) and agent-<id>.meta.json
(description = the agent label "<RUN_ID> <role>:<iter>", roles gate, plan, run,
analyze, critic, screen0, screen1; meta "model" = the requested alias).

Usage:
  python ar5_cost_from_transcripts.py <transcript_dir> [--out result.json]
         [--models models.json|'{"FULL":{"plan":"sonnet",...}}']
         [--outcomes outcomes.json] [--day-one]

--models   assigned alias per run and role ({RUN: {role: alias}}, or a flat {role: alias} for all
           runs; screen0/screen1 use the "screen" key). Without it, the meta.json "model" alias is
           the assignment. The resolved model id must contain the assigned family (sonnet/haiku/opus);
           mismatches are recorded, printed, and make the exit code 3 (the JSON is still written).
--outcomes {RUN: {"settled_correct": n, "audited_correct": m}} for the headline cost ratios.
--day-one  labels look like "plan:1" with no run id; all agents are one run, DAYONE.

Prices are first-party API list prices (USD per million tokens). Tokens are primary, USD is the
list-price equivalent. Sensitivity re-prices Haiku at Sonnet/ratio for ratios 5x, 10x, 20x.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# input, output, cache read, cache write
PRICE = {
    "sonnet": (2.00, 10.00, 0.20, 2.50),
    "haiku": (0.10, 0.50, 0.01, 0.125),
    "opus": (4.00, 20.00, 0.20, 5.00),
}
RATIOS = (5, 10, 20)
FIELDS = ("input", "output", "cache_read", "cache_write")
LABEL = re.compile(r"^([A-Z][A-Z0-9_]*) (gate|plan|run|analyze|critic|screen[01]):(\S+)$")
DAYONE_LABEL = re.compile(r"^(gate|plan|run|analyze|critic|screen[01]?):(\S+)$")


def family(model):
    for f in PRICE:
        if f in (model or ""):
            return f
    return None


def price_cost(fam_tokens, ratio=None):
    """fam_tokens {family: {input,...}}; ratio None = list prices, else Haiku = Sonnet/ratio."""
    total = 0.0
    for fam, u in fam_tokens.items():
        p = PRICE.get(fam)
        if p is None:
            continue
        if ratio is not None and fam == "haiku":
            p = tuple(x / ratio for x in PRICE["sonnet"])
        total += sum(u[k] * p[i] for i, k in enumerate(FIELDS)) / 1e6
    return total


def msg_cost(model, u):
    fam = family(model)
    return price_cost({fam: u}) if fam else 0.0


def parse_label(label, day_one):
    """-> (run, role, iter). Role screen0/screen1 collapse to 'screen'."""
    if day_one:
        m = DAYONE_LABEL.match(label)
        if m:
            return "DAYONE", m.group(1).rstrip("01") if m.group(1).startswith("screen") else m.group(1), m.group(2)
        return "DAYONE", "unknown", label
    m = LABEL.match(label)
    if not m:
        return "UNLABELED", "unknown", label
    role = "screen" if m.group(2).startswith("screen") else m.group(2)
    return m.group(1), role, m.group(3)


def load_models(arg):
    if not arg:
        return None
    m = json.loads(arg if arg.lstrip()[:1] in "{[" else Path(arg).read_text())
    # also accept the autoresearch.live.js args shape: [{run_id, models}] or {"runs": [{run_id, models}]}
    lst = m.get("runs") if isinstance(m, dict) and isinstance(m.get("runs"), list) else m
    if isinstance(lst, list):
        m = {x["run_id"]: x["models"] for x in lst if isinstance(x, dict) and "run_id" in x and isinstance(x.get("models"), dict)}
    return m


def assigned_alias(models, run, role, meta_model):
    if models:
        per = models.get(run, models) if isinstance(models.get(run, None), dict) else models
        v = per.get(role)
        if isinstance(v, str):
            return v, "models"
    return meta_model, "meta"


def ts_epoch(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


def zero():
    return {k: 0 for k in FIELDS} | {"messages": 0, "agents": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--outcomes", type=Path, default=None)
    ap.add_argument("--day-one", action="store_true")
    a = ap.parse_args()
    models = load_models(a.models)

    cell = defaultdict(lambda: defaultdict(lambda: defaultdict(zero)))  # run -> role -> model -> usage
    fam_tok = defaultdict(lambda: defaultdict(lambda: {k: 0 for k in FIELDS}))  # run -> family -> tokens
    rows = []  # per-message
    iters = defaultdict(lambda: defaultdict(lambda: {"cost": 0.0, "first": None}))  # run -> iter
    plan_iters = defaultdict(set)
    critic_iters = defaultdict(set)
    screen_iters = defaultdict(set)
    sonnet_critic_calls = defaultdict(int)
    mism, n_checked, assign_log = [], 0, {}

    for meta_p in sorted(a.transcript_dir.rglob("agent-*.meta.json")):
        try:
            meta = json.loads(meta_p.read_text())
            jsonl_text = meta_p.with_suffix("").with_suffix(".jsonl").read_text()
        except (OSError, ValueError) as e:
            print(f"WARNING: skipping {meta_p.name}: {e}", file=sys.stderr)
            continue
        label = meta.get("description", "")
        run, role, it = parse_label(label, a.day_one)
        alias, src = assigned_alias(models, run, role, meta.get("model"))
        # streamed chunks repeat a message id with a growing output count: keep the last record per id
        last = {}
        for line in jsonl_text.splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = d.get("message")
            if isinstance(msg, dict) and msg.get("usage") and msg.get("id") and msg.get("model") != "<synthetic>":
                last[msg["id"]] = (msg, d.get("timestamp"))
        if role == "plan":
            plan_iters[run].add(it)
        if role == "critic":
            critic_iters[run].add(it)
        if role == "screen":
            screen_iters[run].add(it)
        counted_models = set()
        for msg, ts in last.values():
            u = msg["usage"]
            model = msg.get("model", "unknown")
            tok = {
                "input": u.get("input_tokens", 0),
                "output": u.get("output_tokens", 0),
                "cache_read": u.get("cache_read_input_tokens", 0),
                "cache_write": u.get("cache_creation_input_tokens", 0),
            }
            row = cell[run][role][model]
            for k in FIELDS:
                row[k] += tok[k]
            row["messages"] += 1
            if model not in counted_models:
                row["agents"] += 1
                counted_models.add(model)
                if role == "critic" and family(model) == "sonnet":
                    sonnet_critic_calls[run] += 1
                if alias:
                    n_checked += 1
                    ok = alias.lower() in model.lower()
                    assign_log[(run, role, alias, model)] = ok
                    if not ok:
                        mism.append({"run": run, "role": role, "iter": it, "label": label, "assigned": alias, "resolved": model, "source": src})
            fam = family(model)
            c = msg_cost(model, tok)
            if fam:
                for k in FIELDS:
                    fam_tok[run][fam][k] += tok[k]
            e = ts_epoch(ts)
            ic = iters[run][it]
            ic["cost"] += c
            if e is not None and (ic["first"] is None or e < ic["first"]):
                ic["first"] = e
            rows.append({"ts": e, "run": run, "role": role, "iter": it, "model": model, "cost_usd": c, **tok})

    # cumulative cost per run in time order
    rows.sort(key=lambda r: (r["run"], r["ts"] if r["ts"] is not None else 0.0))
    cum = defaultdict(float)
    for r in rows:
        cum[r["run"]] += r["cost_usd"]
        r["cum_cost_usd"] = round(cum[r["run"]], 6)
        r["cost_usd"] = round(r["cost_usd"], 6)

    outcomes = json.loads(a.outcomes.read_text()) if a.outcomes else {}
    out_runs = {}
    for run in sorted(cell):
        by_role = {}
        tot = {k: 0 for k in FIELDS}
        for role in sorted(cell[run]):
            by_role[role] = {}
            for model in sorted(cell[run][role]):
                u = cell[run][role][model]
                by_role[role][model] = {**u, "cost_usd": round(msg_cost(model, u), 4)}
                for k in FIELDS:
                    tot[k] += u[k]
        total = price_cost(fam_tok[run])
        order = sorted(iters[run].items(), key=lambda kv: (kv[1]["first"] is None, kv[1]["first"] or 0.0))
        c_run, cc = 0.0, []
        for it, v in order:
            c_run += v["cost"]
            cc.append({"iter": it, "cost_usd": round(v["cost"], 4), "cum_cost_usd": round(c_run, 4)})
        n_it = len(plan_iters[run]) or len(iters[run])
        rr = {
            "total_tokens": tot,
            "total_cost_usd": round(total, 3),
            "sensitivity_usd": {f"{r}x": round(price_cost(fam_tok[run], r), 3) for r in RATIOS},
            "by_role": by_role,
            "iterations": n_it,
            "sonnet_critic_calls": sonnet_critic_calls[run],
            "sonnet_critic_calls_per_iter": round(sonnet_critic_calls[run] / n_it, 3) if n_it else None,
            "screen_iters": len(screen_iters[run]),
            "escalated_iters": len(critic_iters[run] & screen_iters[run]) if screen_iters[run] else None,
            "escalation_rate": round(len(critic_iters[run] & screen_iters[run]) / len(screen_iters[run]), 3) if screen_iters[run] else None,
            "cum_by_iter": cc,
        }
        o = outcomes.get(run)
        if o:
            rr["cost_per_correct_settled_verdict"] = round(total / o["settled_correct"], 3) if o.get("settled_correct") else None
            rr["cost_per_audited_correct_entry"] = round(total / o["audited_correct"], 3) if o.get("audited_correct") else None
        out_runs[run] = rr

    result = {
        "mode": "day-one" if a.day_one else "ar5",
        "prices_usd_per_mtok": {k: dict(zip(FIELDS, v)) for k, v in PRICE.items()},
        "assertion": {"checked_agent_model_pairs": n_checked, "mismatches": mism},
        "runs": out_runs,
        "messages": rows,
    }

    print("| run | role | model | agents | messages | input | output | cache read | cache write | cost USD |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for run in sorted(cell):
        for role in sorted(cell[run]):
            for model in sorted(cell[run][role]):
                u = cell[run][role][model]
                print(f"| {run} | {role} | {model} | {u['agents']} | {u['messages']} | {u['input']} | {u['output']} | {u['cache_read']} | {u['cache_write']} | {round(msg_cost(model, u), 3)} |")
    print()
    print("| run | total tokens (in+out+cr+cw) | USD list | 5x | 10x | 20x | iters | sonnet critic calls | per iter | escalation rate |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for run, r in out_runs.items():
        t = sum(r["total_tokens"].values())
        s = r["sensitivity_usd"]
        print(f"| {run} | {t} | {r['total_cost_usd']} | {s['5x']} | {s['10x']} | {s['20x']} | {r['iterations']} | {r['sonnet_critic_calls']} | {r['sonnet_critic_calls_per_iter']} | {r['escalation_rate']} |")
    if outcomes:
        print()
        print("| run | USD list | settled correct | USD per correct settled verdict | audited correct | USD per audited-correct entry |")
        print("|---|---|---|---|---|---|")
        for run, r in out_runs.items():
            o = outcomes.get(run) or {}
            print(f"| {run} | {r['total_cost_usd']} | {o.get('settled_correct')} | {r.get('cost_per_correct_settled_verdict')} | {o.get('audited_correct')} | {r.get('cost_per_audited_correct_entry')} |")
    print(f"\nmodel assertion: {n_checked} agent-model pairs checked, {len(mism)} mismatches")
    for m in mism:
        print("  MISMATCH", m)
    if a.out:
        a.out.write_text(json.dumps(result, indent=1))
    sys.exit(3 if mism else 0)


if __name__ == "__main__":
    main()
