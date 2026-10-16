"""AR1: planner-context ablation for the avrl autoresearch loop, replayed offline.

The loop's real trace (52 experiments, 2026-09-30 to 2026-10-03, with critic-checked
verdicts) is the oracle. Each iteration a planner picks the next experiment from
the menu of not-yet-run experiments; the "runner" returns the archived result body
(numbers only, verdict stripped); an analyst writes the verdict, a one-line summary,
and the updated state of the three pre-registered claims. The run stops when the
claim state matches the campaign's final verdicts (claim 1 pass, claim 2 kill or
tie, claim 3 kill) on two consecutive iterations, or after --max-iters.

Arms differ only in what the planner sees about past experiments:
  A  full result bodies plus verdicts (full logs)
  B  one line per experiment (compressed ledger)
  C  B plus two sampled full entries of killed or inconclusive experiments (inspirations)

Model calls go through `claude -p` (Sonnet, no tools, JSON schema output).
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

PLAN_SCHEMA = {
    "type": "object",
    "properties": {"pick": {"type": "string"}, "why": {"type": "string"}},
    "required": ["pick", "why"],
}
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "kill", "inconclusive", "tie"]},
        "one_line": {"type": "string"},
        "claims": {
            "type": "object",
            "properties": {
                "c1": {"type": "string", "enum": ["open", "pass", "kill", "tie"]},
                "c2": {"type": "string", "enum": ["open", "pass", "kill", "tie"]},
                "c3": {"type": "string", "enum": ["open", "pass", "kill", "tie"]},
            },
            "required": ["c1", "c2", "c3"],
        },
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "one_line", "claims", "reasoning"],
}


def call(system: str, prompt: str, schema: dict, model: str, retries: int = 3):
    cmd = [
        "claude", "-p", "--model", model, "--no-session-persistence", "--tools", "", "--strict-mcp-config", "--setting-sources", "",
        "--output-format", "json", "--json-schema", json.dumps(schema), "--system-prompt", system, prompt,
    ]
    for attempt in range(retries):
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        try:
            d = json.loads(proc.stdout)
        except json.JSONDecodeError:
            d = {"is_error": True, "result": proc.stdout[-500:] + proc.stderr[-500:]}
        if not d.get("is_error"):
            out = d.get("structured_output")
            if out is None:
                txt = d.get("result", "")
                try:
                    out = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
                except (ValueError, json.JSONDecodeError):
                    out = None
            if out is not None:
                u = d.get("usage", {})
                meta = {
                    "cost_usd": d.get("total_cost_usd", 0.0),
                    "input_tokens": u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0),
                    "output_tokens": u.get("output_tokens", 0),
                    "seconds": round(time.time() - t0, 1),
                }
                return out, meta
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"claude call failed after {retries} attempts: {str(d)[:400]}")


def one_liners(history):
    return "\n".join(f"- {h['id']}: {h['verdict']}. {h['one_line']}" for h in history) or "(nothing run yet)"


def planner_context(arm, history, oracle, rng):
    if arm == "A":
        if not history:
            return "(nothing run yet)"
        return "\n\n".join(
            f"### {h['id']} (verdict: {h['verdict']})\n{oracle[h['id']]['result_body']}\nAnalyst: {h['one_line']}" for h in history
        )
    block = "Compressed ledger (one line per experiment run so far):\n" + one_liners(history)
    if arm == "C":
        pool = [h for h in history if h["verdict"] in ("kill", "inconclusive", "tie")]
        picks = rng.sample(pool, min(2, len(pool)))
        if picks:
            block += "\n\nInspirations (full entries of experiments that did not pan out, re-read for near-misses and dead ends):\n" + "\n\n".join(
                f"### {h['id']} (verdict: {h['verdict']})\n{oracle[h['id']]['result_body']}" for h in picks
            )
    return block


def claims_match(c):
    return all(c.get(k) in v for k, v in TARGET.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or ROOT / "runs" / "AR1" / f"{a.arm}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    trace_f = (out / "trace.jsonl").open("w")

    oracle = {r["id"]: r for r in json.loads(ORACLE.read_text())}
    rng = random.Random(a.seed)
    menu_order = list(oracle)
    rng.shuffle(menu_order)
    remaining = list(menu_order)
    history, claims = [], {"c1": "open", "c2": "open", "c3": "open"}
    first_match, settled_at, flips, totals = None, None, 0, {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "planner_input_tokens": 0}

    for it in range(1, a.max_iters + 1):
        menu = "\n".join(f"- {rid}: {oracle[rid]['hypothesis']}" for rid in remaining)
        plan_prompt = (
            f"{CLAIMS}\n\nCurrent claim state: {json.dumps(claims)}\n\nWhat has been run:\n{planner_context(a.arm, history, oracle, rng)}\n\n"
            f"Menu of experiments available to run next (id: hypothesis or setting):\n{menu}\n\n"
            "Pick the single experiment whose result would most change the state of the three claims (highest information value, prefer experiments that can kill a claim, "
            "avoid near-duplicates of what has been run unless a replication is what is missing). Return its exact id in 'pick'."
        )
        plan, pm = call("You are the planner of an autonomous research loop. Answer only via the schema.", plan_prompt, PLAN_SCHEMA, a.model)
        pick = plan["pick"].strip()
        if pick not in remaining:
            cands = [r for r in remaining if r.lower() == pick.lower() or pick.lower().startswith(r.lower())]
            pick = cands[0] if cands else remaining[0]
        remaining.remove(pick)

        an_prompt = (
            f"{CLAIMS}\n\nClaim state before this experiment: {json.dumps(claims)}\n\nCompressed ledger so far:\n{one_liners(history)}\n\n"
            f"New experiment {pick}. Hypothesis or setting: {oracle[pick]['hypothesis']}\n\nRaw result (verdict removed, numbers as recorded):\n{oracle[pick]['result_body']}\n\n"
            "Give the verdict for this experiment against its own hypothesis (pass, kill, inconclusive, or tie), a one-line summary under 35 words with the key numbers, "
            "and the updated state of each claim. Rules: move a claim from open only when the evidence is sufficient (a pre-registered kill rule triggered, or consistent results "
            "across at least two experiments or seeds); you may revert a claim to open or change it if new evidence contradicts the earlier state; leave untouched claims as they were."
        )
        an, am = call("You are the analyst of an autonomous research loop. Answer only via the schema.", an_prompt, ANALYSIS_SCHEMA, a.model)
        new_claims = {k: an["claims"].get(k, claims[k]) for k in claims}
        flips += sum(1 for k in claims if claims[k] != "open" and new_claims[k] != claims[k])
        claims = new_claims
        history.append({"id": pick, "verdict": an["verdict"], "one_line": an["one_line"]})

        for k in ("cost_usd", "input_tokens", "output_tokens"):
            totals[k] += pm[k] + am[k]
        totals["planner_input_tokens"] += pm["input_tokens"]
        rec = {"iter": it, "pick": pick, "why": plan["why"], "header_verdict": oracle[pick]["header_verdict"], "verdict": an["verdict"],
               "one_line": an["one_line"], "claims": claims, "reasoning": an["reasoning"], "planner": pm, "analyst": am}
        trace_f.write(json.dumps(rec) + "\n")
        trace_f.flush()
        print(f"[{a.arm} s{a.seed}] it{it} {pick}: {an['verdict']} claims={claims} cost so far ${totals['cost_usd']:.2f}", flush=True)

        if claims_match(claims):
            if first_match is None:
                first_match = it
            elif settled_at is None:
                settled_at = first_match
                break
        else:
            first_match = None

    result = {"arm": a.arm, "seed": a.seed, "model": a.model, "experiments_to_verdict": settled_at, "censored": settled_at is None,
              "iterations": len(history), "final_claims": claims, "flips": flips, **totals, "order": [h["id"] for h in history]}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    (out / "done.flag").touch()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
