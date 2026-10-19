"""L9a-shardB-tiebreak, POST HOC sensitivity (run AFTER the registered final stage; not part of the verdict, not registered): interval types for S_tb (percentile as registered, basic, normal around the plug-in, bias-centred), bootstrap bias (mean minus plug-in),
leave-one-budget-out plug-ins, the budget-window concentration, caught-error gains in absolute errors per budget (8-seed full-pool means), and shard A (pooled accept flags, from the BOOT_B=0 files) against shard B plug-ins. Writes <run_dir>/sensitivity.json. Usage: l9a_sensitivity.py <run_dir>"""
import sys, os, json
import numpy as np
R = sys.argv[1]; FR = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4); fs = lambda f: f"{f:g}"; ARM = {"rm": "agree_acc_soth", "q3_4b": "agree_acc_sdiff"}
ld = lambda w, k, f: json.load(open(os.path.join(R, w, k, f"boot_tb_f{fs(f)}.json")))
def stat(bj, arm):
    fp = bj["full_pool"]; plug = float(np.log(np.mean(fp[arm]) / np.mean(fp["agree_acc"])))
    S = np.log(np.array([np.mean(r[arm]) for r in bj["resamples"]]) / np.array([np.mean(r["agree_acc"]) for r in bj["resamples"]])) if bj["resamples"] else None
    return plug, S
def intervals(S, th):
    q = np.percentile(S, [2.5, 97.5]); sd = S.std(ddof=1); b = S.mean() - th
    return {"plug_in": float(th), "boot_mean": float(S.mean()), "bias_boot_mean_minus_plug_in": float(b), "boot_sd": float(sd), "z_plug_over_sd": float(th / sd), "percentile": [float(q[0]), float(q[1])], "basic": [float(2 * th - q[1]), float(2 * th - q[0])], "normal": [float(th - 1.96 * sd), float(th + 1.96 * sd)], "bias_centred_percentile": [float(q[0] - b), float(q[1] - b)]}
out = {}
for k, arm in ARM.items():
    o = {}
    for w, nm in (("boot", "pooled_AB"), ("boot_B", "shard_B")):
        pl, Ss = zip(*[stat(ld(w, k, f), arm) for f in FR]); pl = np.array(pl); S = np.mean(Ss, axis=0)
        loo = {fs(f): float(np.delete(pl, i).mean()) for i, f in enumerate(FR)}
        o[nm] = {"S_tb": intervals(S, pl.mean()), "per_budget_log_plug_in": {fs(f): float(x) for f, x in zip(FR, pl)}, "n_budget_plug_ins_positive": int((pl > 0).sum()), "leave_one_budget_out_plug_in": loo,
                 "plug_in_without_25_and_30": float(np.mean([x for f, x in zip(FR, pl) if f not in (0.25, 0.3)])), "plug_in_only_25_and_30": float(np.mean([x for f, x in zip(FR, pl) if f in (0.25, 0.3)]))}
        if nm == "pooled_AB": o[nm]["per_budget_ratio"] = {fs(f): intervals(np.log(np.array([np.mean(r[arm]) for r in ld(w, k, f)["resamples"]]) / np.array([np.mean(r["agree_acc"]) for r in ld(w, k, f)["resamples"]])), pl[i]) for i, f in enumerate(FR)}
    pa = np.array([stat(ld("boot_A0", k, f), arm)[0] for f in FR]); o["shard_A_pooled_accept_flags_plug_in"] = {"S_tb": float(pa.mean()), "per_budget_log": {fs(f): float(x) for f, x in zip(FR, pa)}}
    c = {}
    for f in FR:
        fp = ld("boot", k, f)["full_pool"]; m = lambda q: float(np.mean(fp[q])); c[fs(f)] = {"budget": fp["budget"], "n_pool_errors": fp["n_err"], "agree_acc": m("agree_acc"), "tie_break": m(arm), "gain_errors": m(arm) - m("agree_acc"), "gain_pct": 100 * (m(arm) / m("agree_acc") - 1), "greedy_blend7": m("greedy_blend7"), "oracle": m("oracle")}
    o["caught_pooled_per_budget_8seed"] = c; out[k] = o
json.dump(out, open(os.path.join(R, "sensitivity.json"), "w"), indent=1)
for k, o in out.items():
    for nm in ("pooled_AB", "shard_B"):
        s = o[nm]["S_tb"]; print(f"[{k} {nm}] S_tb plug {s['plug_in']:+.4f} bootmean {s['boot_mean']:+.4f} (bias {s['bias_boot_mean_minus_plug_in']:+.4f}) SD {s['boot_sd']:.4f} z {s['z_plug_over_sd']:.2f}; percentile [{s['percentile'][0]:+.4f}, {s['percentile'][1]:+.4f}] basic [{s['basic'][0]:+.4f}, {s['basic'][1]:+.4f}] normal [{s['normal'][0]:+.4f}, {s['normal'][1]:+.4f}] bias-centred [{s['bias_centred_percentile'][0]:+.4f}, {s['bias_centred_percentile'][1]:+.4f}]; positive budgets {o[nm]['n_budget_plug_ins_positive']}/6; w/o 25,30: {o[nm]['plug_in_without_25_and_30']:+.4f}; only 25,30: {o[nm]['plug_in_only_25_and_30']:+.4f}")
        print("   leave-one-budget-out plug-ins: " + ", ".join(f"{b} {v:+.4f}" for b, v in o[nm]["leave_one_budget_out_plug_in"].items()))
    print(f"[{k}] shard A (pooled flags) S_tb plug-in {o['shard_A_pooled_accept_flags_plug_in']['S_tb']:+.4f}")
    print("   caught per budget (pooled 8 seeds): " + "; ".join(f"{b} agree_acc {v['agree_acc']:.1f} tie-break {v['tie_break']:.1f} ({v['gain_errors']:+.1f} errors, {v['gain_pct']:+.1f}%) oracle {v['oracle']:.0f}" for b, v in o["caught_pooled_per_budget_8seed"].items()))
