#!/usr/bin/env python3
"""Critic check for L3z. Recomputes the numbers in the RESEARCH_LOG.md L3z entry from runs/L3z/*.json,
recomputes every planted-pool verdict from the stored bootstrap draws, checks the injection construction
against the base table, and checks the reading against the pre-registered rules. CPU only, no GPU, no ssh.
Output goes to stdout; the caller redirects it to runs/L3z/critic_check.txt."""
import json
import math
import os
import sys

sys.dont_write_bytecode = True
AVRL = "/homes/rdubash/avrl_live/CHEAP/avrl"
sys.path.insert(0, AVRL)
sys.path.insert(0, os.path.join(AVRL, "exp"))
RUN = os.path.join(AVRL, "runs", "L3z")

import numpy as np  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (("  [" + str(detail) + "]") if detail != "" else ""))
    if not cond:
        fails.append(name)


# Values as printed in the RESEARCH_LOG.md L3z entry table: k, verdict, bias, pct upper, basic upper, A_k
ENTRY = {
    0: ("KILL", 0.00869, 0.00880, -0.01371, -0.03068),
    1: ("INCONCLUSIVE", 0.00614, -0.00710, -0.01566, -0.04386),
    2: ("KILL", 0.00718, 0.00132, -0.01684, -0.03440),
    3: ("KILL", -0.00464, 0.00552, 0.01296, -0.03784),
    4: ("INCONCLUSIVE", 0.01166, -0.00458, -0.02342, -0.04874),
    5: ("INCONCLUSIVE", 0.02134, -0.03317, -0.07846, -0.09461),
    6: ("INCONCLUSIVE", 0.01646, -0.00237, -0.03749, -0.05877),
    7: ("KILL", -0.01760, -0.01031, 0.03029, -0.05075),
    8: ("INCONCLUSIVE", 0.00947, -0.03854, -0.06023, -0.09547),
    9: ("KILL", -0.00760, 0.00003, 0.01381, -0.02064),
    10: ("KILL", -0.01816, 0.00703, 0.04010, -0.02982),
    11: ("INCONCLUSIVE", 0.01677, -0.00851, -0.04323, -0.05361),
    12: ("INCONCLUSIVE", 0.02409, -0.01249, -0.05722, -0.06651),
    13: ("KILL", 0.01754, 0.00111, -0.02704, -0.06078),
    14: ("INCONCLUSIVE", 0.00753, -0.01448, -0.04033, -0.07253),
    15: ("KILL", 0.01805, 0.00117, -0.03369, -0.06021),
    16: ("INCONCLUSIVE", 0.01786, -0.02733, -0.06336, -0.07913),
    17: ("INCONCLUSIVE", 0.00594, -0.02160, -0.03247, -0.07856),
    18: ("PASS", 0.00246, -0.00720, -0.01205, -0.04300),
    19: ("INCONCLUSIVE", 0.01328, -0.00438, -0.03162, -0.05705),
}
TOL5 = 6e-6  # entry values are rounded to 5 decimal places

# ---------- 1. per-pool recomputation from raw result.json ----------
pools = []
for k in range(20):
    r = json.load(open(os.path.join(RUN, f"planted_{k}", "result.json")))
    main = r["main_25"]
    ne = float(r["n_proxy_errors"])
    thr = (ne - np.array(main["threshold"]["caught_per_seed"], dtype=np.float64)) / ne
    g6s = (ne - np.array(main["greedy6s"]["caught_per_seed"], dtype=np.float64)) / ne
    diff8 = g6s - thr
    A_k = float(diff8.mean())
    point = float(diff8[:4].mean())
    draws = np.array(r["metric"]["greedy6s_minus_threshold_admitted_per_resample"], dtype=np.float64)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    pct_up = float(hi)
    basic_up = float(2 * point - lo)
    bias = float(draws.mean() - point)
    if pct_up >= 0 or basic_up >= 0:
        verdict = "KILL"
    elif abs(bias) < 0.003:
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    det = all(r["checks"]["same_pool_determinism"][a]["rerun_equals_main"] for a in ("threshold", "greedy6u", "greedy6s"))
    calls_ok = all(set(main[a]["calls_per_seed"]) == {750} and len(main[a]["calls_per_seed"]) == 8
                   for a in ("threshold", "greedy6u", "greedy6s"))
    cw = r["convergence_warnings"]
    cw_total = 0
    for a in ("threshold", "greedy6s", "greedy6u"):
        cw_total += int(cw[a].get("main", 0)) + int(cw[a].get("bootstrap", 0))
    thr_boot_mean = float(np.mean(r["metric"]["threshold_subsample_mean_seeds_0_to_3"]))
    thr_full = float(r["metric"]["threshold_full_pool_admitted_mean_seeds_0_to_3"])
    pools.append(dict(k=k, seed=r["pool_seed"], strength=r["strength"], verdict=r["verdict"], recomputed=verdict,
                      bias=bias, pct=pct_up, basic=basic_up, A=A_k, point=point, det=det, calls_ok=calls_ok,
                      cw=cw_total, wall=r["wall_s"], sd=float(draws.std(ddof=1)), n_draws=len(draws),
                      thr_shift=thr_boot_mean - thr_full, stored_point=r["metric"]["point_estimate_seeds_0_to_3"],
                      stored_pct=r["metric"]["bootstrap"]["ci95_percentile"][1],
                      stored_basic=r["metric"]["bootstrap"]["ci95_basic"][1],
                      stored_bias=r["metric"]["bootstrap"]["bias"], corr=r["planted_diagnostics"]["planted_corr_with_label"],
                      admitted={a: float(np.mean(main[a]["admitted_per_seed"])) for a in main},
                      caught={a: float(np.mean(main[a]["caught_per_seed"])) for a in main},
                      nerr=r["n_proxy_errors"], ngroups=r["n_groups"], sub=r["bootstrap_subsample_size"],
                      B=r["B"], seeds_boot=r["seeds_bootstrap"], seeds_main=r["seeds_main"]))

print("== 1. per-pool recomputation (20 planted pools)")
for p in pools:
    k = p["k"]
    check(f"k={k} seed/strength/groups/errors", p["seed"] == 40000 + k and p["strength"] == 0.4 and
          p["ngroups"] == 3000 and p["nerr"] == 436 and p["sub"] == 1500 and p["B"] == 200 and
          p["seeds_main"] == 8 and p["seeds_boot"] == 4, "")
    check(f"k={k} recomputed verdict equals stored", p["recomputed"] == p["verdict"], f"{p['recomputed']} vs {p['verdict']}")
    check(f"k={k} stored point/bounds/bias equal recomputed", abs(p["point"] - p["stored_point"]) < 1e-12 and
          abs(p["pct"] - p["stored_pct"]) < 1e-12 and abs(p["basic"] - p["stored_basic"]) < 1e-12 and
          abs(p["bias"] - p["stored_bias"]) < 1e-12, f"d_point {p['point'] - p['stored_point']:+.2e}")
    check(f"k={k} determinism all arms", p["det"], "")
    check(f"k={k} budget 750 calls per main run", p["calls_ok"], "")
    check(f"k={k} no ConvergenceWarnings", p["cw"] == 0, p["cw"])
    check(f"k={k} 200 bootstrap draws", p["n_draws"] == 200, p["n_draws"])
    e = ENTRY[k]
    check(f"k={k} entry row verdict", e[0] == p["verdict"], f"{e[0]} vs {p['verdict']}")
    check(f"k={k} entry row bias/pct/basic/A_k within 5dp", abs(e[1] - p["bias"]) <= TOL5 and
          abs(e[2] - p["pct"]) <= TOL5 and abs(e[3] - p["basic"]) <= TOL5 and abs(e[4] - p["A"]) <= TOL5,
          f"bias {p['bias']:+.6f} pct {p['pct']:+.6f} basic {p['basic']:+.6f} A {p['A']:+.6f}")

# ---------- 2. aggregate numbers used in the entry ----------
print("== 2. aggregate numbers")
verd = [p["verdict"] for p in pools]
counts = {v: verd.count(v) for v in ("PASS", "KILL", "INCONCLUSIVE", "INVALID_CONTROL")}
check("counts PASS 1 KILL 8 INCONCLUSIVE 11 INVALID 0", counts == {"PASS": 1, "KILL": 8, "INCONCLUSIVE": 11, "INVALID_CONTROL": 0}, counts)
bounds_only = sum(1 for p in pools if p["pct"] < 0 and p["basic"] < 0)
check("bounds-only count 12", bounds_only == 12, bounds_only)
bo_inc = sum(1 for p in pools if p["pct"] < 0 and p["basic"] < 0 and p["verdict"] == "INCONCLUSIVE")
check("11 of the 12 bounds-only pools are INCONCLUSIVE", bo_inc == 11, bo_inc)
bias = np.array([p["bias"] for p in pools])
check("mean bias +0.00782", abs(bias.mean() - 0.00782) < 6e-6, f"{bias.mean():+.6f}")
check("SD bias 0.01192 (ddof 1)", abs(bias.std(ddof=1) - 0.01192) < 6e-6, f"{bias.std(ddof=1):.6f}")
check("SE bias 0.00267", abs(bias.std(ddof=1) / math.sqrt(20) - 0.00267) < 6e-6, f"{bias.std(ddof=1) / math.sqrt(20):.6f}")
check("bias positive in 16 of 20", int((bias > 0).sum()) == 16, int((bias > 0).sum()))
check("|bias| < 0.003 in 1 of 20", int((np.abs(bias) < 0.003).sum()) == 1, int((np.abs(bias) < 0.003).sum()))
A = np.array([p["A"] for p in pools])
check("A mean -0.05585 (entry -0.0558)", abs(A.mean() - (-0.05585)) < 1e-5, f"{A.mean():+.6f}")
check("A SD 0.0209 across pools", abs(A.std(ddof=1) - 0.0209) < 6e-5, f"{A.std(ddof=1):.6f}")
on_target = abs(A.mean() + 0.03) <= 0.005
check("OFF_TARGET: |A + 0.03| > 0.005", not on_target, f"|A+0.03| = {abs(A.mean() + 0.03):.5f}")
check("entry says A is 0.026 more negative than target", abs((A.mean() + 0.03) - (-0.026)) < 0.001, f"{A.mean() + 0.03:+.5f}")
sds = np.array([p["sd"] for p in pools])
check("bootstrap SD mean 0.0198, range 0.0159 to 0.0233", abs(sds.mean() - 0.0198) < 5e-5 and
      abs(sds.min() - 0.0159) < 5e-5 and abs(sds.max() - 0.0233) < 5e-5,
      f"mean {sds.mean():.5f} min {sds.min():.5f} max {sds.max():.5f}")
arms = ("threshold", "greedy6u", "greedy6s")
adm = {a: np.mean([p["admitted"][a] for p in pools]) for a in arms}
cgt = {a: np.mean([p["caught"][a] for p in pools]) for a in arms}
check("admitted means threshold 0.7062 greedy6u 0.6486 greedy6s 0.6504",
      abs(adm["threshold"] - 0.7062) < 5e-5 and abs(adm["greedy6u"] - 0.6486) < 5e-5 and abs(adm["greedy6s"] - 0.6504) < 5e-5,
      {a: round(float(v), 5) for a, v in adm.items()})
check("caught means 128.1 / 153.2 / 152.4",
      abs(cgt["threshold"] - 128.1) < 0.06 and abs(cgt["greedy6u"] - 153.2) < 0.06 and abs(cgt["greedy6s"] - 152.4) < 0.06,
      {a: round(float(v), 2) for a, v in cgt.items()})
check("A equals greedy6s minus threshold admitted means (0.6504 - 0.7062)",
      abs((adm["greedy6s"] - adm["threshold"]) - A.mean()) < 1e-4, f"{adm['greedy6s'] - adm['threshold']:+.5f}")
wall = sum(p["wall"] for p in pools)
check("eval pool compute about 16.2 min", abs(wall / 60 - 16.2) < 0.1, f"{wall / 60:.2f} min, per pool {[round(p['wall']) for p in pools]}")
corr = np.array([p["corr"] for p in pools])
check("planted feature-label |corr| roughly 0.13 to 0.16 (entry wording, across the 20 pools)",
      np.abs(corr).min() > 0.12 and np.abs(corr).max() < 0.17,
      f"min {np.abs(corr).min():.3f} max {np.abs(corr).max():.3f}; pools 0 and 18 only: "
      f"{np.abs(pools[0]['corr']).min():.3f} to {np.abs(pools[0]['corr']).max():.3f} and "
      f"{np.abs(pools[18]['corr']).min():.3f} to {np.abs(pools[18]['corr']).max():.3f}")
thr_shift = np.array([p["thr_shift"] for p in pools])
se_l3z = bias.std(ddof=1) / math.sqrt(20)
l3v_mean, l3v_se = 0.00216, 0.00276
print(f"   descriptive: L3z mean bias {bias.mean():+.5f} (SE {se_l3z:.5f}, t {bias.mean() / se_l3z:+.2f}); "
      f"L3v null mean bias {l3v_mean:+.5f} (SE {l3v_se:.5f}, t {l3v_mean / l3v_se:+.2f}); "
      f"difference {bias.mean() - l3v_mean:+.5f}, SE of difference {math.sqrt(se_l3z**2 + l3v_se**2):.5f}, "
      f"z {(bias.mean() - l3v_mean) / math.sqrt(se_l3z**2 + l3v_se**2):+.2f}")
greedy_shift = thr_shift + bias
print(f"   descriptive: threshold-only subsample shift mean {thr_shift.mean():+.5f} SD {thr_shift.std(ddof=1):.5f}; "
      f"bias mean {bias.mean():+.5f}; implied greedy6s subsample shift mean {greedy_shift.mean():+.5f}")

# ---------- 3. pilot ----------
print("== 3. pilot")
pil = json.load(open(os.path.join(RUN, "pilot", "result.json")))
check("pilot status OK, s* = 0.40", pil["status"] == "OK" and pil["chosen_strength"] == 0.4, pil["status"])
rows = {round(r["strength"], 2): r for r in pil["rows"]}
check("pilot statistic at 0.40 = -0.0302", abs(rows[0.4]["pilot_statistic"] - (-0.0302)) < 6e-5,
      f"{rows[0.4]['pilot_statistic']:+.6f}")
check("pilot per-pool diffs at 0.40 are -0.035 and -0.026",
      [round(x, 3) for x in rows[0.4]["per_pool_diff"]] == [-0.035, -0.026], rows[0.4]["per_pool_diff"])
for s, want in [(0.30, -0.0033), (0.35, -0.0155), (0.50, 0.0328), (0.60, -0.0013), (0.70, -0.0407), (0.80, -0.0773)]:
    check(f"pilot curve s={s}", abs(rows[s]["pilot_statistic"] - want) < 6e-5, f"{rows[s]['pilot_statistic']:+.5f}")
check("pilot wall time (entry says 34 s; raw file says 32.8 s)", abs(pil["wall_s"] - 34.0) < 0.5, f"raw wall_s {pil['wall_s']}")
d = {s: abs(r["pilot_statistic"] + 0.03) for s, r in rows.items()}
check("s* is the closest grid value to -0.03", min(d, key=lambda s: (d[s], s)) == 0.4, "")

# ---------- 4. aggregate file consistency ----------
print("== 4. result.json and summary.json consistency")
res = json.load(open(os.path.join(RUN, "result.json")))
summ = json.load(open(os.path.join(RUN, "summary.json")))
m = res["metric"]
check("result.json verdict-level fields", m["on_target"] is False and m["pass_count"] == 1 and m["bounds_only_count"] == 12 and
      m["n_valid_pools"] == 20, "")
check("result.json reading is R1 OFF_TARGET", res["reading"].startswith("R1 OFF_TARGET"), res["reading"])
check("summary.json per-pool verdicts match planted_k", all(summ["metric"] and summ["per_pool"][k]["verdict"] == pools[k]["verdict"]
                                                             for k in range(20)), "")
check("result.json mean bias equals mean of stored per-pool bias", abs(m["mean_bias"] - np.mean([p["stored_bias"] for p in pools])) < 1e-12, "")
check("pool_exits.log all 0", all(l.strip().endswith("exit 0") for l in open(os.path.join(RUN, "pool_exits.log")) if l.strip()), "")
check("exit_code 0", open(os.path.join(RUN, "exit_code")).read().strip() == "0", "")
check("done.flag present", os.path.exists(os.path.join(RUN, "done.flag")), "")

# ---------- 5. injection construction and leakage surface ----------
print("== 5. planted pool construction")
import pandas as pd  # noqa: E402
import l3z_planted_power as L  # noqa: E402

df = pd.read_parquet(L.BASE_FEATS)
base = df[L.FEAT_COLS].to_numpy(dtype=np.float64)
y0 = df["proxy_error"].to_numpy(dtype=np.int64)
sd = base.std(axis=0)
sgn = np.sign([np.corrcoef(base[:, c], y0.astype(np.float64))[0, 1] for c in range(6)])
for pseed in (40000, 40018):
    pool, diag = L.build_planted(pseed, 0.4)
    perm = np.random.default_rng(pseed).permutation(3000)
    want = base[perm] + 0.4 * y0[:, None] * (sgn * sd)[None, :]
    check(f"planted X = base[perm] + s*sgn*sd*y_in_place (seed {pseed})", np.allclose(pool["X"], want, atol=1e-12), "")
    check(f"labels stay in place (seed {pseed})", np.array_equal(pool["y"], y0), int(pool["y"].sum()))
    check(f"agreement/n_distinct fields equal X columns (seed {pseed})",
          np.array_equal(pool["agreement"], pool["X"][:, 0]) and np.array_equal(pool["n_distinct"], pool["X"][:, 1]), "")
real_corr_agree = float(np.corrcoef(base[:, 0], y0.astype(np.float64))[0, 1])
print(f"   descriptive: real agreement vs proxy_error corr {real_corr_agree:+.3f} (entry says -0.56)")
check("entry real agreement corr -0.56", abs(real_corr_agree + 0.56) < 0.01, f"{real_corr_agree:+.4f}")
ASSUMED_RECORD_KEYS = {"idx", "feat_vec", "cell", "agreement", "n_distinct"}
src = open(os.path.join(AVRL, "exp", "l3z_planted_power.py")).read()
check("run_arm record dict uses only gold-free keys", '"idx": j' in src and '"cell": 0' in src and
      '"agreement": float(pool["agreement"][g])' in src and '"n_distinct": float(pool["n_distinct"][g])' in src, "")
check("gold label enters allocators only via verified update tuple",
      src.count('pool["y"][g]') == 2 and "records = [" in src, "")

print("== 6. L3v calibration cited by the entry")
l3v = os.path.join(AVRL, "runs", "L3v", "aggregate.log")
if os.path.exists(l3v):
    print("   " + " | ".join(l.strip() for l in open(l3v) if "bias" in l.lower() or "PASS" in l)[:600])

print("== summary")
print(f"failed checks: {len(fails)} {fails if fails else ''}")
print(f"verdict per pool (recomputed): {verd}")
sys.exit(1 if fails else 0)
