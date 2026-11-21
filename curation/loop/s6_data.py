#!/usr/bin/env python3
"""Step 6 data: rebuild every planner prompt of finished loops and pair it with the idea the planner gave.

    python curation/loop/s6_data.py --roots curation/runs/B1 curation/runs/C0 curation/runs/C1 --pool v2 \
        --out curation/runs/S6/data

The loop does not store prompts, but a prompt is a function of the loop seed, the slot, the best file at the time and the
history, all of which the ledger holds. The rebuilt prompt is checked against the length the call log recorded.
Writes plans.jsonl (every slot), train.jsonl (kept ideas of the strong planner: the training set of the trained planner)
and memory.txt (the best kept ideas as text, for the planner that gets them in its prompt).
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine as E   # noqa: E402

MEMORY_N, MEMORY_CHARS = 3, 1800


def slots_of(run_dir, pool_data, start_dev):
    """One row per slot of a loop: the prompt the planner saw and what came of its idea."""
    cfg = json.loads((run_dir / "config.json").read_text())
    ledger = [json.loads(l) for l in (run_dir / "ledger.jsonl").read_text().splitlines()]
    sent = {}
    for l in (run_dir / "calls.jsonl").read_text().splitlines():
        c = json.loads(l)
        if ":planner:" in c["tag"] and c["ok"]:
            sent[int(c["tag"].rsplit(":", 1)[1])] = c["prompt_chars"]
    lp = object.__new__(E.Loop)
    lp.cfg, lp.docs, lp.target, lp.pool_tokens = cfg, *pool_data
    inc, inc_dev, history, rows = str(E.TASK / "curate.py"), start_dev, [], []
    for rec in ledger:
        lp.s = {"slot": rec["slot"] - 1, "incumbent": inc, "inc_scores": {str(E.SEED_A): inc_dev}, "history": list(history)}
        prompt = lp.plan_prompt()
        planner = next((c["model"] for c in rec["calls"] if c["role"] == "planner"), None)
        coder = next((c["model"] for c in rec["calls"] if c["role"] == "coder"), None)
        rows.append({"loop": cfg["loop_id"], "cell": cfg.get("cell"), "slot": rec["slot"], "planner": planner, "coder": coder,
                     "decision": rec["decision"], "dev": rec.get("dev"), "inc": inc_dev, "idea": rec.get("idea"),
                     "why": rec.get("why"), "prompt": prompt, "chars_logged": sent.get(rec["slot"] - 1)})
        history.append({"slot": rec["slot"], "idea": rec.get("idea") or "(no idea)", "outcome": rec["outcome"]})
        if rec["decision"] == "accept":
            inc, inc_dev = rec["cand"], rec["dev"]
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--pool", default="v2")
    ap.add_argument("--strong", default="fable")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pool = E.ROOT / "data" / "curation" / ("pool_%s" % a.pool)
    pool_data = (json.loads((pool / "docs.json").read_text()), json.loads((pool / "target_sample.json").read_text()),
                 json.loads((pool / "summary.json").read_text())["pool_tokens"])
    dirs = [d for r in a.roots for d in sorted(Path(r).iterdir())
            if (d / "final.json").exists() and (d / "ledger.jsonl").exists() and (d / "calls.jsonl").exists()
            and json.loads((d / "config.json").read_text()).get("pool") == a.pool]
    start = None                                  # dev bpb of the starting file on the decision seed: the same in every loop
    for d in dirs:
        for l in (d / "ledger.jsonl").read_text().splitlines():
            rec = json.loads(l)
            if rec["decision"] == "accept":
                break
            if rec.get("inc") is not None:
                start = rec["inc"]
                break
        if start is not None:
            break
    rows = [r for d in dirs for r in slots_of(d, pool_data, start)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "plans.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    strong = [r for r in rows if r["planner"] == a.strong and r["idea"]]
    kept = [r for r in strong if r["decision"] == "accept"]
    with open(out / "train.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps({"loop": r["loop"], "slot": r["slot"], "system": E.SYSTEM, "user": r["prompt"],
                                "assistant": json.dumps({"idea": r["idea"], "why": r["why"]}, ensure_ascii=False)}) + "\n")
    best = sorted((r for r in kept if r["slot"] == 1), key=lambda r: r["dev"])   # first-slot ideas stand on their own
    lines = ["Notes from earlier runs of this same task on a similar pool (other documents). These changes were kept "
             "there; the dev bpb each one reached is given:"]
    lines += ["- (dev bpb %.4f) %s" % (r["dev"], " ".join(r["idea"].split())[:MEMORY_CHARS]) for r in best[:MEMORY_N]]
    (out / "memory.txt").write_text("\n".join(lines) + "\n")
    # the call log counts the system prompt too, so a faithful rebuild differs from it by one constant in every slot
    diffs = {r["chars_logged"] - len(r["prompt"]) for r in rows if r["chars_logged"] is not None}
    print("loops %d, slots %d, start dev %.4f" % (len(dirs), len(rows), start))
    print("strong-planner slots %d, kept %d (from %d loops)" % (len(strong), len(kept), len({r["loop"] for r in kept})))
    print("logged length minus rebuilt length over %d slots: %s (system prompt: %d characters)" % (
        sum(r["chars_logged"] is not None for r in rows), sorted(diffs), len(E.SYSTEM)))
    print("longest training prompt: %d characters" % max(len(r["prompt"]) for r in kept))
    return 0


if __name__ == "__main__":
    sys.exit(main())
