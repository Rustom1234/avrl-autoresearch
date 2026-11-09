"""L7a-tiebreak critic: seed noise of the registered point-ratio quantities X_p and V_p. Re-runs the cross-fit full-half arms (agree_acc, the two tie-breaks, tb_xfit with the REGISTERED per-half choice, greedy_blend with the registered warm start) on seeds 0..7 (must equal the stored 8-seed values) and then on 200 further seeds, in blocks of 8 seeds. CPU only, reads the j4 source up to the XFIT branch so the policies are the registered ones."""
import sys, os, json, numpy as np
ROOT = "/homes/rdubash/avrl_live/FULL/avrl"; RUN = f"{ROOT}/runs/L7a-tiebreak"
src = open(f"{ROOT}/exp/sim_warm_boot_j4.py").read(); src = src[: src.index("if XFIT:  # cross-fit verdict mode")]
POOLS = {"rm": ("runs/L5c-proxies/rm", "judge_error", "accept"), "judge3b": ("runs/L5c-proxies/judge3b", "judge_error", "accept"), "L3b-judge-proxy": ("runs/L3b-judge-proxy", "judge_error", "accept")}
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 25; out = {}
for pool, (d, lab, acc) in POOLS.items():
    ns = {"__name__": "x"}; sys.argv = ["x", f"{ROOT}/{d}/feats_J.parquet", lab, "cell100", "s_maj,s_oth,s_diff,mean_tokens,std_tokens,any_truncated,log_qlen,agreement,n_distinct"]
    os.environ["ACCEPT_COL"] = acc; os.environ.pop("XFIT", None); import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "j4_head", "exec"), ns)
    X = json.load(open(f"{RUN}/{pool}/boot_xfit_f0.25.json")); sim0 = ns["sim0"]; POL = ns["POL"]; set_pool = ns["set_pool"]
    S = {a: np.zeros((2, 8 * NB + 8)) for a in ("agree_acc", "agree_acc_soth", "agree_acc_sdiff", "tb_xfit", "greedy_blend", "greedy_warm")}
    for h in (0, 1):
        held = sim0[sim0.pid % 2 == h].reset_index(drop=True); set_pool(held); ns["BEST"] = X["halves"][h]["full_half"]["best_fixed"]; b = int(0.25 * len(held)); tbn = X["halves"][h]["tb_xfit_is"]
        for s in range(8 * NB + 8):
            for a in S:
                S[a][h, s] = POL[tbn](b, s) if a == "tb_xfit" else POL[a](b, s)
        stored = X["halves"][h]["full_half"]
        for a in ("agree_acc", "greedy_blend", "tb_xfit"): assert list(S[a][h, :8]) == stored[a], (pool, h, a, list(S[a][h, :8]), stored[a])
    tot = {a: S[a].sum(0) for a in S}  # sum over halves per seed (the two halves use the same seed index, as in the registered run)
    reg = {a: tot[a][:8].mean() for a in S}; conv = {a: tot[a][8:].mean() for a in S}
    blocks = np.array([[tot[a][8 + 8 * k: 16 + 8 * k].mean() for a in ("agree_acc", "greedy_blend", "tb_xfit")] for k in range(NB)])
    V = blocks[:, 0] / blocks[:, 1]; Xr = blocks[:, 2] / blocks[:, 1]
    out[pool] = {"registered_8seed": reg, "converged_seeds8plus": conv, "n_extra_seeds": 8 * NB, "V_conv": conv["agree_acc"] / conv["greedy_blend"], "X_conv": conv["tb_xfit"] / conv["greedy_blend"], "V_blocks_mean_sd": [float(V.mean()), float(V.std(ddof=1))], "X_blocks_mean_sd": [float(Xr.mean()), float(Xr.std(ddof=1))],
                 "frac_blocks_V_lt_098": float((V < 0.98).mean()), "frac_blocks_X_ge_098": float((Xr >= 0.98).mean()), "frac_blocks_V_and_X": float(((V < 0.98) & (Xr >= 0.98)).mean()), "seed_sd_per_arm_sum_over_halves": {a: float(tot[a][8:].std(ddof=1)) for a in S}}
    print(pool, json.dumps(out[pool], indent=1), flush=True)
os.makedirs(f"{RUN}/_critic", exist_ok=True); json.dump(out, open(f"{RUN}/_critic/seed_noise.json", "w"), indent=1)
