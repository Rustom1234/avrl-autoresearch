"""AR6: planner routing policies in the autoresearch loop, replayed offline on the ledger oracle.

Same substrate and analyst as AR1 (52 critic-checked experiments; the runner is a lookup; a fixed
Sonnet analyst tracks the three claims from the compressed ledger plus the new result; planner sees
full logs, the best context per AR1). Only the PLANNER changes between arms:

  HAIKU     one Haiku planner call per iteration
  SONNET    one Sonnet planner call per iteration (AR1 arm A)
  CONSIST3  three Haiku planner samples; if at least two pick the same id, take it; else one Sonnet call
  VERBAL    one Haiku call that also reports confidence 0 to 100; below --verbal-threshold, one Sonnet call
  STALL     Haiku by default; after --stall-k consecutive iterations with no change in the claim state,
            the next pick is made by Sonnet, then back to Haiku (progress-triggered escalation)
  STRONG_EARLY  Sonnet while any claim is open, Haiku once every claim has left "open"
  RANDOM25  Haiku, but each pick is made by Sonnet with probability 0.25 (budget-matched control)

Run stops when the claim state matches the campaign's final verdicts on two consecutive iterations
(claim 1 pass, claim 2 kill or tie, claim 3 kill), cap --max-iters. Writes
runs/AR6/<policy>_s<seed>/{trace.jsonl,result.json}.
"""
import argparse
import json
import random
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "runs" / "AR1" / "oracle.json"

CLAIMS = (
    "Thesis: reward computation in RL post-training is a budgeted active-measurement problem. "
    "A learned allocator decides per rollout group whether to pay the expensive verifier or accept the cheap proxy. "
    "Pre-registered claims. Claim 1: equal policy quality at a fraction of expensive verifier calls (a fixed rule may satisfy it). "
    "Claim 2: a LEARNED allocator admits fewer corrupted rewards into training than ANY fixed rule at the same budget. "
    "Claim 3: the learned allocator's edge comes from structure in prompt space (proxy error clusters by prompt embedding). "
    "Kill rules: claim 3 dies if embeddings add under 0.05 AUC over cheap features and the neighbor ratio is under 1.5; "
    "claim 2 dies where a budget-matched fixed threshold ties or beats the learned allocator with intervals; "
    "claim 1 passes where the quarter-budget rule is within the pre-registered margin of full verification across seeds."
)
TARGET = {"c1": {"pass"}, "c2": {"kill", "tie"}, "c3": {"kill"}}

PLAN_SCHEMA = {"type": "object", "properties": {"pick": {"type": "string"}, "why": {"type": "string"}, "confidence": {"type": "integer", "minimum": 0, "maximum": 100}}, "required": ["pick", "why", "confidence"]}
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "kill", "inconclusive", "tie"]},
        "one_line": {"type": "string"},
        "claims": {"type": "object", "properties": {k: {"type": "string", "enum": ["open", "pass", "kill", "tie"]} for k in ("c1", "c2", "c3")}, "required": ["c1", "c2", "c3"]},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "one_line", "claims", "reasoning"],
}


def call(system, prompt, schema, model, retries=3):
    cmd = ["claude", "-p", "--model", model, "--no-session-persistence", "--tools", "", "--strict-mcp-config", "--setting-sources", "",
           "--output-format", "json", "--json-schema", json.dumps(schema), "--system-prompt", system]
    d = {}
    for attempt in range(retries):
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600)
            d = json.loads(proc.stdout)
        except (json.JSONDecodeError, subprocess.TimeoutExpired):
            d = {"is_error": True}
        if not d.get("is_error") and isinstance(d.get("structured_output"), dict):
            u = d.get("usage", {})
            return d["structured_output"], {"cost_usd": d.get("total_cost_usd", 0.0), "model": model, "seconds": round(time.time() - t0, 1),
                                            "input_tokens": u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)}
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"claude call failed: {str(d)[:300]}")


def one_liners(history):
    return "\n".join(f"- {h['id']}: {h['verdict']}. {h['one_line']}" for h in history) or "(nothing run yet)"


def full_logs(history, oracle):
    if not history:
        return "(nothing run yet)"
    return "\n\n".join(f"### {h['id']} (verdict: {h['verdict']})\n{oracle[h['id']]['result_body']}\nAnalyst: {h['one_line']}" for h in history)


def claims_match(c):
    return all(c.get(k) in v for k, v in TARGET.items())


def plan_prompt(claims, history, oracle, remaining):
    menu = "\n".join(f"- {rid}: {oracle[rid]['hypothesis']}" for rid in remaining)
    return (f"{CLAIMS}\n\nCurrent claim state: {json.dumps(claims)}\n\nWhat has been run:\n{full_logs(history, oracle)}\n\n"
            f"Menu of experiments available to run next (id: hypothesis or setting):\n{menu}\n\n"
            "Pick the single experiment whose result would most change the state of the three claims (highest information value, prefer experiments that can kill a claim, "
            "avoid near-duplicates of what has been run unless a replication is what is missing). Return its exact id in 'pick', and in 'confidence' your 0 to 100 confidence that this pick is the most informative one available.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["HAIKU", "SONNET", "CONSIST3", "VERBAL", "STALL", "STRONG_EARLY", "RANDOM25"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--analyst-model", default="sonnet")
    ap.add_argument("--verbal-threshold", type=int, default=70)
    ap.add_argument("--stall-k", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or ROOT / "runs" / "AR6" / f"{a.policy}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    trace_f = (out / "trace.jsonl").open("w")
    oracle = {r["id"]: r for r in json.loads(ORACLE.read_text())}
    rng = random.Random(a.seed)
    remaining = list(oracle)
    rng.shuffle(remaining)
    history, claims = [], {"c1": "open", "c2": "open", "c3": "open"}
    first_match = settled_at = None
    flips = 0
    stall = 0
    tot = {"cost_usd": 0.0, "plan_cost_usd": 0.0, "plan_calls_haiku": 0, "plan_calls_sonnet": 0, "strong_picks": 0}
    PLAN_SYS = "You are the planner of an autonomous research loop. Answer only via the schema."

    for it in range(1, a.max_iters + 1):
        pp = plan_prompt(claims, history, oracle, remaining)
        route = None
        plans = []
        if a.policy == "SONNET":
            route = "sonnet"
        elif a.policy == "HAIKU":
            route = "haiku"
        elif a.policy == "RANDOM25":
            route = "sonnet" if rng.random() < 0.25 else "haiku"
        elif a.policy == "STRONG_EARLY":
            route = "sonnet" if any(v == "open" for v in claims.values()) else "haiku"
        elif a.policy == "STALL":
            route = "sonnet" if stall >= a.stall_k else "haiku"
        elif a.policy == "VERBAL":
            p, m = call(PLAN_SYS, pp, PLAN_SCHEMA, "haiku")
            plans.append((p, m)); tot["plan_calls_haiku"] += 1
            route = "haiku" if p.get("confidence", 0) >= a.verbal_threshold else "sonnet"
            if route == "haiku":
                plan, pm = p, m
        elif a.policy == "CONSIST3":
            for _ in range(3):
                p, m = call(PLAN_SYS, pp, PLAN_SCHEMA, "haiku")
                plans.append((p, m)); tot["plan_calls_haiku"] += 1
            picks = [p["pick"].strip() for p, _ in plans]
            top = max(set(picks), key=picks.count)
            if picks.count(top) >= 2:
                route = "haiku"
                plan, pm = next((p, m) for p, m in plans if p["pick"].strip() == top)
            else:
                route = "sonnet"
        if route == "sonnet":
            plan, pm = call(PLAN_SYS, pp, PLAN_SCHEMA, "sonnet"); tot["plan_calls_sonnet"] += 1; tot["strong_picks"] += 1
        elif a.policy in ("HAIKU", "RANDOM25", "STRONG_EARLY", "STALL"):
            plan, pm = call(PLAN_SYS, pp, PLAN_SCHEMA, "haiku"); tot["plan_calls_haiku"] += 1
        if a.policy == "STALL" and route == "sonnet":
            stall = 0
        plan_cost = pm["cost_usd"] + sum(m["cost_usd"] for p, m in plans if (p, m) is not (plan, pm))
        tot["plan_cost_usd"] += plan_cost

        pick = plan["pick"].strip()
        if pick not in remaining:
            cands = [r for r in remaining if r.lower() == pick.lower() or pick.lower().startswith(r.lower())]
            pick = cands[0] if cands else remaining[0]
        remaining.remove(pick)

        an_prompt = (f"{CLAIMS}\n\nClaim state before this experiment: {json.dumps(claims)}\n\nCompressed ledger so far:\n{one_liners(history)}\n\n"
                     f"New experiment {pick}. Hypothesis or setting: {oracle[pick]['hypothesis']}\n\nRaw result (verdict removed, numbers as recorded):\n{oracle[pick]['result_body']}\n\n"
                     "Give the verdict for this experiment against its own hypothesis (pass, kill, inconclusive, or tie), a one-line summary under 35 words with the key numbers, "
                     "and the updated state of each claim. Rules: move a claim from open only when the evidence is sufficient (a pre-registered kill rule triggered, or consistent results "
                     "across at least two experiments or seeds); you may revert a claim to open or change it if new evidence contradicts the earlier state; leave untouched claims as they were.")
        an, am = call("You are the analyst of an autonomous research loop. Answer only via the schema.", an_prompt, ANALYSIS_SCHEMA, a.analyst_model)
        new_claims = {k: an["claims"].get(k, claims[k]) for k in claims}
        changed = new_claims != claims
        flips += sum(1 for k in claims if claims[k] != "open" and new_claims[k] != claims[k])
        stall = 0 if changed else stall + 1
        claims = new_claims
        history.append({"id": pick, "verdict": an["verdict"], "one_line": an["one_line"]})
        tot["cost_usd"] += plan_cost + am["cost_usd"]
        trace_f.write(json.dumps({"iter": it, "route": route, "pick": pick, "confidence": plan.get("confidence"), "n_haiku_samples": len(plans),
                                  "haiku_picks": [p["pick"] for p, _ in plans], "verdict": an["verdict"], "claims": claims, "changed": changed,
                                  "plan_cost_usd": round(plan_cost, 4), "analyst_cost_usd": am["cost_usd"]}) + "\n")
        trace_f.flush()
        print(f"[{a.policy} s{a.seed}] it{it} {route:6s} {pick}: {an['verdict']} claims={claims} ${tot['cost_usd']:.2f}", flush=True)
        if claims_match(claims):
            if first_match is None:
                first_match = it
            elif settled_at is None:
                settled_at = first_match
                break
        else:
            first_match = None

    res = {"policy": a.policy, "seed": a.seed, "analyst_model": a.analyst_model, "experiments_to_verdict": settled_at, "censored": settled_at is None,
           "iterations": len(history), "final_claims": claims, "flips": flips, **{k: round(v, 4) if isinstance(v, float) else v for k, v in tot.items()},
           "strong_pick_fraction": round(tot["strong_picks"] / max(1, len(history)), 3), "order": [h["id"] for h in history]}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    (out / "done.flag").touch()
    print(json.dumps(res))


if __name__ == "__main__":
    main()
