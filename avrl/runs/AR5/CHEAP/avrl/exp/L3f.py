#!/usr/bin/env python3
"""L3f aggregate (CPU). Reads the fresh-pool rollout summary and structure read (runs/L3f/pool), the E03 arms on the
fresh pool (runs/L3f/e03) and the E01 references (runs/p0, runs/E03), applies the pre-registered L3f rules and writes
runs/L3f/result.json. Writes nothing outside runs/L3f/. Any failed assertion is recorded as status "failed" with the
traceback, and the process exits 1."""
import json
import os
import traceback

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AVRL = os.path.join(ROOT, "avrl")
L3F = os.path.join(AVRL, "runs", "L3f")
BUDGET, CALLS = "0.25", 750
STATUS = {
    "KILL": "two-pool KILL: fresh pool 95% CI entirely above 0, same standard as E01",
    "INCONCLUSIVE": "fresh pool INCONCLUSIVE: 95% CI straddles 0, the E01 kill stays one-pool",
    "PASS_CANDIDATE": "fresh pool 95% CI entirely below 0: conflicts with E01, candidate only, not credited",
}


def load(path):
    with open(path) as f:
        return json.load(f)


def verdict(lo, hi):
    if lo > 0:
        return "KILL"
    if hi < 0:
        return "PASS_CANDIDATE"
    return "INCONCLUSIVE"


def boot_block(d):
    return {"mean": d["mean"], "ci95": d["ci95"], "frac_below_zero": d["frac_below_zero"], "n_resamples": d["n_resamples"]}


def run():
    pool = load(os.path.join(L3F, "pool", "result.json"))
    roll = load(os.path.join(L3F, "pool", "summary.json"))
    e03 = load(os.path.join(L3F, "e03", "result.json"))
    e01 = load(os.path.join(AVRL, "runs", "E03", "result.json"))
    e01_pool = load(os.path.join(AVRL, "runs", "p0", "result.json"))

    feats = pd.read_parquet(os.path.join(L3F, "pool", "feats.parquet"), columns=["question"])
    p0q = pd.read_parquet(os.path.join(AVRL, "runs", "p0", "groups.parquet"), columns=["question"])
    overlap = len(set(feats.question) & set(p0q.question))
    assert len(feats) == 3000 and pool["n_groups"] == 3000, (len(feats), pool["n_groups"])
    assert overlap == 0, f"fresh pool shares {overlap} questions with p0"

    b = e03["per_budget"][BUDGET]
    assert b["verifier_calls_per_run"] == CALLS and b["verifier_calls_equal_across_arms"], b["verifier_calls_per_run"]
    boot = e03["metric"]["bootstrap"]
    assert boot["B"] == 100 and boot["seeds"] == 4 and boot["budget"] == 0.25, boot

    l, g = boot["learned_minus_threshold_admitted"], boot["greedy_minus_threshold_admitted"]
    v_l = verdict(*l["ci95"])
    v_g = verdict(*g["ci95"])

    e01_boot = e01["metric"]["bootstrap"]
    auc_gain = pool["auc_cheap_emb"] - pool["auc_cheap"]
    ratio = pool["neighbor_ratio"]["cell100"]
    kill3 = auc_gain < 0.05 and ratio < 1.5
    conf = pool["confident_subset"]

    return {
        "id": "L3f",
        "pool": {
            "path": os.path.join(L3F, "pool"),
            "n_groups": pool["n_groups"],
            "n_proxy_errors": e03["n_proxy_errors"],
            "proxy_error_rate": pool["proxy_error_rate"],
            "p0_question_overlap": overlap,
            "rollouts": roll,
        },
        "e03_fresh": {
            "budget": 0.25,
            "verifier_calls_per_run": b["verifier_calls_per_run"],
            "errors_per_seed": b["errors_per_seed"],
            "arms": {arm: {"caught_mean": b["arms"][arm]["caught_mean"], "admitted_mean": b["arms"][arm]["admitted_mean"],
                           "admitted_per_seed": b["arms"][arm]["admitted_per_seed"]} for arm in ("random", "threshold", "learned", "greedy")},
            "point_estimate_seeds_0_3": e03["metric"]["point_estimate_seeds_0_to_boot_seeds_minus_1"],
            "bootstrap_B": boot["B"],
            "bootstrap_seeds": boot["seeds"],
            "e03_script_verdict_binary": e03["verdict"],
        },
        "claim2": {
            "learned_minus_threshold_admitted": {**boot_block(l), "verdict": v_l, "status": STATUS[v_l]},
            "greedy_minus_threshold_admitted_diagnostic": {**boot_block(g), "verdict": v_g, "status": STATUS[v_g]},
            "e01_reference": {"learned_minus_threshold_admitted_point": e01["metric"]["point_estimate_seeds_0_to_boot_seeds_minus_1"]["learned_minus_threshold_admitted"],
                              "learned_minus_threshold_admitted_bootstrap": boot_block(e01_boot["learned_minus_threshold_admitted"])},
        },
        "claim3": {
            "fresh": {"auc_cheap": pool["auc_cheap"], "auc_cheap_emb": pool["auc_cheap_emb"], "auc_gain": auc_gain,
                      "neighbor_ratio_cell30": pool["neighbor_ratio"]["cell30"], "neighbor_ratio_cell100": ratio,
                      "neighbor_ratio_cell300": pool["neighbor_ratio"]["cell300"],
                      "confident_subset": {"n": conf["n"], "proxy_error_rate": conf["proxy_error_rate"], "auc_cheap": conf["auc_cheap"],
                                           "auc_cheap_emb": conf["auc_cheap_emb"], "auc_gain": conf["auc_cheap_emb"] - conf["auc_cheap"],
                                           "neighbor_ratio_cell100": conf["neighbor_ratio_cell100"]},
                      "kill_condition_met": kill3},
            "e01_reference": {"auc_gain": e01_pool["auc_cheap_emb"] - e01_pool["auc_cheap"],
                              "neighbor_ratio_cell100": e01_pool["neighbor_ratio"]["cell100"]},
            "status": ("two-pool KILL condition met on fresh pool (AUC gain < 0.05 and cell100 ratio < 1.5)" if kill3
                       else "fresh pool does not meet the kill condition: claim 3 survives on this pool"),
        },
        "wall_s_rollouts": roll["wall_s"],
    }


if __name__ == "__main__":
    os.makedirs(L3F, exist_ok=True)
    try:
        res = run()
        res["status"] = "ok"
    except Exception:
        res = {"id": "L3f", "status": "failed", "error": traceback.format_exc()}
    with open(os.path.join(L3F, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    raise SystemExit(0 if res["status"] == "ok" else 1)
