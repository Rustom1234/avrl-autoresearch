"""L3a summary: joins the fresh-pool and L1B-pool result.json files into runs/L3a/result.json. CPU only.

Final verdict: GATE_FAILED if either pool failed a gate; NOT_RUN if a pool wrote no result; otherwise the fresh-pool
verdict (config A at 5%, floor 40: PASS if the mean gap over split seeds 0 to 7 is above +2.0 pp, else KILL).
"""
import os, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--fresh", required=True); ap.add_argument("--l1b", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def pick(r, tag, scope="all_splits"):
    if r is None or not r.get("configs"): return None
    c = r["configs"][tag]; st = c[scope]
    return {"config": c["config"], "budget_frac": c["budget_frac"], "splits": len(st["per_split_pp"]),
            "mean_pp": st["mean_pp"], "sd_pp": st["sd_pp"], "per_split_pp": st["per_split_pp"],
            "splits_above_0": st["n_positive"], "splits_above_2pp": st["n_above_pass_pp"]}


F, P = load(a.fresh), load(a.l1b)
gates = {"spend_fresh": None if F is None else F["spend_gate"]["pass"],
         "spend_l1b": None if P is None else P["spend_gate"]["pass"],
         "reproduction_l1b": None if P is None else P["repro_gate"]["pass"]}
failed = [x for x in (F, P) if x is not None and x["verdict"] == "GATE_FAILED"]
if failed: verdict = "GATE_FAILED"
elif F is None or P is None: verdict = "NOT_RUN"
else: verdict = F["verdict"]
summary = {"experiment": "L3a", "verdict": verdict,
           "rule": "fresh pool, config A at 5%, refit floor 40: mean gap over split seeds 0 to 7 above +2.0 pp is PASS, else KILL",
           "gates": gates,
           "fresh_pool": {"primary_config_A_5pct": pick(F, "A"), "secondary_config_B_10pct": pick(F, "B"),
                          "floor10_reference": None if F is None else F.get("floor10_reference")},
           "l1b_pool_context": {"config_A_5pct_splits_4_to_7": pick(P, "A", "held_splits_4_to_7"),
                                "config_B_10pct_splits_4_to_7": pick(P, "B", "held_splits_4_to_7"),
                                "config_A_5pct_splits_0_to_7": pick(P, "A"), "config_B_10pct_splits_0_to_7": pick(P, "B"),
                                "floor10_reference": None if P is None else P.get("floor10_reference")},
           "caveats": ["replay only, no GRPO", "one model, one task (GSM8K)", "majority-vote proxy for correctness",
                       "cheap features", "the fresh-pool verdict is one pool"]}
with open(a.out, "w") as fh: json.dump(summary, fh, indent=1)
p = summary["fresh_pool"]["primary_config_A_5pct"]
print(f"L3a verdict {verdict} | gates {gates}", flush=True)
if p: print(f"fresh config A 5% floor 40: mean {p['mean_pp']:+.3f} sd {p['sd_pp']:.3f} per split {[round(x, 3) for x in p['per_split_pp']]} "
            f"above 0 {p['splits_above_0']} above +2 {p['splits_above_2pp']}", flush=True)
print(f"wrote {a.out}", flush=True)
