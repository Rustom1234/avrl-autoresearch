"""L7a smoke aggregator (CPU). Applies the SMOKE BINS S1 (null), S2 (planted s_oth), S3 (planted s_diff) registered in the docstring of exp/sim_warm_boot_j4.py to the per-world result.json files written by exp/l7a_finalize.py --smoke. usage: l7a_smoke_check.py <smoke_dir>"""
import sys, os, json
import numpy as np
S = sys.argv[1]; POOLS = ["rm", "judge3b", "L3b-judge-proxy"]; ld = lambda w: json.load(open(os.path.join(S, w, "result.json")))
out = {}
# S1 null: 6 worlds x 3 pools
w0 = [f"null_s{i}" for i in range(6)]; rs = [ld(w) for w in w0]; xs = [[r["pools"][p]["xfit"]["0.25"] for p in POOLS] for r in rs]; ms = [[r["pools"][p]["main"]["0.25"] for p in POOLS] for r in rs]
pass_fire = [w for w, r in zip(w0, rs) if r["verdict_line"].replace("[SMOKE, no verdict] ", "").startswith("PASS")]; kill_fire = [w for w, r in zip(w0, rs) if r["verdict_line"].replace("[SMOKE, no verdict] ", "").startswith("KILL")]
lo = [x["ratios"]["tb_xfit_over_agree_acc"]["percentile_ci95"][0] for row in xs for x in row]; plug = [x["ratios"]["tb_xfit_over_agree_acc"]["plug_in"] for row in xs for x in row]
lit = [w for w, r in zip(w0, rs) if sum(r["P_literal_pass"].values()) >= 2]; nP = sum(sum(r["P_pass"].values()) for r in rs); nPL = sum(sum(r["P_literal_pass"].values()) for r in rs); nK = sum(sum(r["K_pass"].values()) for r in rs)
gb_aa = [x["ratios"]["greedy_blend_over_agree_acc"]["plug_in"] for row in xs for x in row]; tb_gb = [x["ratios"]["tb_xfit_over_greedy_blend"]["plug_in"] for row in xs for x in row]
s1 = {"worlds_with_PASS_verdict": pass_fire, "worlds_with_KILL_verdict": kill_fire, "n_tb_over_aa_ci_lower_gt1_of_18": int(sum(l > 1.0 for l in lo)), "mean_plugin_tb_xfit_over_agree_acc": float(np.mean(plug)), "sd_plugin": float(np.std(plug, ddof=1)),
      "info_worlds_with_PASS_literal_bin": lit, "info_pool_runs_P": int(nP), "info_pool_runs_P_literal": int(nPL), "info_pool_runs_K": int(nK), "info_mean_greedy_blend_over_agree_acc": float(np.mean(gb_aa)), "info_mean_tb_xfit_over_greedy_blend": float(np.mean(tb_gb))}
s1["PASS"] = bool(not pass_fire and not kill_fire and s1["n_tb_over_aa_ci_lower_gt1_of_18"] <= 1 and abs(s1["mean_plugin_tb_xfit_over_agree_acc"] - 1) <= 0.03); out["S1_null"] = s1
for tag, w, feat in (("S2_planted_soth", "planted_soth_s0", "s_oth"), ("S3_planted_sdiff", "planted_sdiff_s0", "s_diff")):
    r = ld(w); d = {}
    for p in POOLS:
        x = r["pools"][p]["xfit"]["0.25"]; q = x["ratios"]["tb_xfit_over_agree_acc"]; d[p] = {"plug_in": q["plug_in"], "percentile_ci95": q["percentile_ci95"], "choices": [s["tb_xfit_is"] for s in x["selection"]], "auc": [(s["selection"]["auc_soth"], s["selection"]["auc_sdiff"]) for s in x["selection"]],
                                                                       "greedy_blend_over_agree_acc": x["ratios"]["greedy_blend_over_agree_acc"]["plug_in"], "tb_over_gb": x["ratios"]["tb_xfit_over_greedy_blend"]["plug_in"]}
    want = "agree_acc_soth" if feat == "s_oth" else "agree_acc_sdiff"; picks = sum(c == want for p in POOLS for c in d[p]["choices"])
    ok_gain = all(d[p]["plug_in"] > 1.04 and d[p]["percentile_ci95"][0] > 1.0 for p in ("rm", "L3b-judge-proxy")); d["picks_planted_of_6"] = int(picks); d["gain_ok_rm_and_qwen3"] = bool(ok_gain); d["PASS"] = bool(ok_gain and picks >= 5); out[tag] = d
out["ALL_SMOKE_PASS"] = bool(out["S1_null"]["PASS"] and out["S2_planted_soth"]["PASS"] and out["S3_planted_sdiff"]["PASS"])
json.dump(out, open(os.path.join(S, "smoke_check.json"), "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
