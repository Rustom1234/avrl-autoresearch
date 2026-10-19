#!/usr/bin/env python3
"""L3aa critic check. Independent re-implementation (does not import allocators.py or l3aa_oracle_ceiling.py) of the
threshold and oracle_tiebreak arms, the per-step diagnostics, the leakage test, the brute-force within-block ceiling test,
and the bootstrap designs A and B. Read-only on the pool and on runs/. Writes its report to runs/L3aa/critic_check.txt.
CPU only. No GPU, no pip, no ssh.
"""
import itertools
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
BASE = "/homes/rdubash/avrl_live/CHEAP/avrl"
POOL = os.path.join(BASE, "runs", "p0", "feats.parquet")
L3AA = os.path.join(BASE, "runs", "L3aa", "result.json")
L3S = os.path.join(BASE, "runs", "L3s", "result.json")
L3H = os.path.join(BASE, "runs", "L3h", "result.json")
OUT = os.path.join(BASE, "runs", "L3aa", "critic_check.txt")
STEP = 16
BUDGET = 0.25
lines = []


def out(msg=""):
    print(msg, flush=True)
    lines.append(str(msg))


def n_verify(q_groups, step):
    q = BUDGET * q_groups
    return int(math.floor(q * (step + 1)) - math.floor(q * step))


df = pd.read_parquet(POOL)
N = len(df)
AGR = df["agreement"].to_numpy(dtype=np.float64)
NDI = df["n_distinct"].to_numpy(dtype=np.int64)
Y = df["proxy_error"].to_numpy(dtype=np.int64)
E = int(Y.sum())
out(f"pool groups={N} proxy_errors={E} blocks={len(set(zip(AGR, NDI)))}")


def thr_top(chunk, k, y_unused=None):
    # threshold block order: agreement ascending, n_distinct descending, stable on chunk position.
    order = sorted(range(len(chunk)), key=lambda j: (AGR[chunk[j]], -NDI[chunk[j]]))
    return order[:k]


def orc_top(chunk, k):
    # oracle: same block order, then proxy_error descending (gold), stable.
    order = sorted(range(len(chunk)), key=lambda j: (AGR[chunk[j]], -NDI[chunk[j]], -Y[chunk[j]]))
    return order[:k]


def run(perm, arm, y_src=Y, collect=False):
    caught = 0
    calls = 0
    gains = []
    changed = 0
    nsteps = math.ceil(len(perm) / STEP)
    sel_sets = []
    for t in range(nsteps):
        chunk = perm[t * STEP:(t + 1) * STEP]
        k = n_verify(len(chunk), t)
        if arm == "threshold":
            pick = thr_top(chunk, k)
        else:
            pick = orc_top(chunk, k)
        c = sum(int(y_src[chunk[j]]) for j in pick)
        caught += c
        calls += k
        sel_sets.append(sorted(int(chunk[j]) for j in pick))
        if collect:
            t_pick = thr_top(chunk, k)
            c_thr = sum(int(y_src[chunk[j]]) for j in t_pick)
            differs = set(pick) != set(t_pick)
            changed += int(differs)
            gains.append((t, k, c - c_thr, differs))
    return {"caught": caught, "calls": calls, "gains": gains, "changed": changed, "sel": sel_sets}


# ---- A. main 8 seeds ----
res = json.load(open(L3AA))
l3s = json.load(open(L3S))
thr_c, orc_c, changed_all, gains_all = [], [], [], []
for s in range(8):
    perm = np.random.default_rng(1000 + s).permutation(N)
    t_run = run(perm, "threshold")
    o_run = run(perm, "oracle_tiebreak", collect=True)
    thr_c.append(t_run["caught"])
    orc_c.append(o_run["caught"])
    changed_all.append(o_run["changed"])
    gains_all.extend([g for g in o_run["gains"] if g[3]])
    assert t_run["calls"] == 750 and o_run["calls"] == 750
out("A. main seeds 0-7, independent re-implementation")
out(f"  threshold caught {thr_c} mean {np.mean(thr_c):.3f}")
out(f"  oracle    caught {orc_c} mean {np.mean(orc_c):.3f}")
diff_c = [o - t for o, t in zip(orc_c, thr_c)]
out(f"  paired caught diff {diff_c} mean {np.mean(diff_c):.4f}")
D = float(np.mean([-d / E for d in diff_c]))  # admitted diff = -(caught diff) / errors
out(f"  D (admitted diff, denominator 436) = {D:+.6f}  reading = {'HEADROOM' if D <= -0.005 else 'CAPPED'}")
out(f"  changed steps per seed {changed_all} total {sum(changed_all)}")
gain_vals = [g[2] for g in gains_all]
out(f"  per-step caught gain over changed steps: min {min(gain_vals)} max {max(gain_vals)} "
    f"histogram {dict(zip(*np.unique(gain_vals, return_counts=True)))}")
out(f"  total caught gain over changed steps = {sum(gain_vals)} (diagnostic claims 67)")
out(f"  oracle per-step gain >= 0 everywhere: {all(g[2] >= 0 for g in gains_all)}")

# compare against result.json
ok_thr = thr_c == res["main_25"]["threshold"]["caught_per_seed"]
ok_orc = orc_c == res["main_25"]["oracle_tiebreak"]["caught_per_seed"]
ok_l3s = thr_c == l3s["main_25"]["threshold"]["caught_per_seed"]
out(f"  matches result.json threshold: {ok_thr}; oracle: {ok_orc}; L3s threshold: {ok_l3s}")
out(f"  matches result.json changed_steps_per_seed: {changed_all == res['oracle_diagnostic']['changed_steps_per_seed']}")

# ---- B. leakage test: threshold selection must not depend on the labels ----
rng = np.random.default_rng(777)
leak_ok = True
for s in range(8):
    perm = np.random.default_rng(1000 + s).permutation(N)
    y_shuf = rng.permutation(Y)
    a = run(perm, "threshold", y_src=Y)["sel"]
    b = run(perm, "threshold", y_src=y_shuf)["sel"]
    leak_ok = leak_ok and (a == b)
out("B. leakage: threshold selected sets identical under a label permutation (all 8 seeds): " + str(leak_ok))

# ---- C. brute-force within-block ceiling on a sample of steps ----
# Within-block reorderings of the threshold order: any k-subset that contains every group in the blocks strictly
# before the boundary block, plus any m-subset of the boundary block (m = slots left). Oracle must equal the max errors.
bad = 0
checked = 0
for s in range(3):
    perm = np.random.default_rng(1000 + s).permutation(N)
    for t in range(0, math.ceil(N / STEP), 5):
        chunk = perm[t * STEP:(t + 1) * STEP]
        k = n_verify(len(chunk), t)
        if k == 0:
            continue
        ordered = sorted(range(len(chunk)), key=lambda j: (AGR[chunk[j]], -NDI[chunk[j]]))
        keys = [(AGR[chunk[j]], -NDI[chunk[j]]) for j in ordered]
        bkey = keys[k - 1]
        before = [j for j, kk in zip(ordered, keys) if kk < bkey]
        boundary = [j for j, kk in zip(ordered, keys) if kk == bkey]
        m = k - len(before)
        best = -1
        for combo in itertools.combinations(boundary, m):
            cnt = sum(int(Y[chunk[j]]) for j in before) + sum(int(Y[chunk[j]]) for j in combo)
            best = max(best, cnt)
        orc = sum(int(Y[chunk[j]]) for j in orc_top(chunk, k))
        checked += 1
        if orc != best:
            bad += 1
out(f"C. brute-force within-block ceiling: checked {checked} steps (seeds 0-2, every 5th step), mismatches {bad}")

# ---- D. bootstrap designs A and B, independent ----
def paired_boot(replace, size, n_boot):
    diffs = []
    thr_means = []
    for b in range(n_boot):
        ids = np.random.default_rng(10000 + b).choice(N, size=size, replace=replace)
        Ys = Y[ids]
        Asub = AGR[ids]
        Nsub = NDI[ids]
        e = int(Ys.sum())
        dd = []
        tt = []
        for s in range(4):
            perm = np.random.default_rng(1000 + s).permutation(size)
            ct = co = 0
            calls = 0
            for t in range(math.ceil(size / STEP)):
                chunk = perm[t * STEP:(t + 1) * STEP]
                k = n_verify(len(chunk), t)
                ot = sorted(range(len(chunk)), key=lambda j: (Asub[chunk[j]], -Nsub[chunk[j]]))[:k]
                oo = sorted(range(len(chunk)), key=lambda j: (Asub[chunk[j]], -Nsub[chunk[j]], -Ys[chunk[j]]))[:k]
                ct += sum(int(Ys[chunk[j]]) for j in ot)
                co += sum(int(Ys[chunk[j]]) for j in oo)
                calls += k
            assert calls == round(BUDGET * size)
            ea = (e - ct) / e
            eo = (e - co) / e
            tt.append(ea)
            dd.append(eo - ea)
        diffs.append(float(np.mean(dd)))
        thr_means.append(float(np.mean(tt)))
    return diffs, thr_means


dA, _ = paired_boot(True, N, 100)
dB, tB = paired_boot(False, N // 2, 200)
U_A = float(np.percentile(dA, 97.5))
# admitted diff (oracle minus threshold) = -(caught diff) / errors, seeds 0-3 full-pool point estimate
point = float(np.mean([-(o - t) / E for o, t in zip(orc_c[:4], thr_c[:4])]))
lo_b, hi_b = np.percentile(dB, [2.5, 97.5])
out("D. bootstrap (independent re-implementation)")
out(f"  design A: U_A (97.5th pct) = {U_A:+.6f} (result.json -0.014011); "
    f"CI [{np.percentile(dA, 2.5):+.6f}, {U_A:+.6f}]; below zero {sum(x < 0 for x in dA)}/100")
out(f"  design B: point (seeds 0-3) = {point:+.6f}; percentile CI [{lo_b:+.6f}, {hi_b:+.6f}]; "
    f"basic CI [{2 * point - hi_b:+.6f}, {2 * point - lo_b:+.6f}]; bias {np.mean(dB) - point:+.6f}; "
    f"below zero {sum(x < 0 for x in dB)}/200")
ref_b = res["metric"]["design_B"]
out(f"  result.json design B percentile {ref_b['ci95_percentile']} basic {ref_b['ci95_basic']} bias {ref_b['bias']}")
out(f"  design B threshold means match L3s draws: "
    f"{np.allclose(tB, l3s['metric']['threshold_subsample_mean_seeds_0_to_3'], rtol=0, atol=1e-12)}")

# ---- E. entry numbers from L3s and L3h ----
g_c = l3s["main_25"]["greedy6s"]["caught_per_seed"]
out("E. greedy6s (L3s) vs oracle per seed")
out(f"  greedy6s caught {g_c} mean {np.mean(g_c):.3f}")
above = [i for i, (g, o) in enumerate(zip(g_c, orc_c)) if g > o]
out(f"  seeds where greedy6s > oracle (0-based): {above}; values " + str([(g_c[i], orc_c[i]) for i in above]))
out(f"  greedy6s paired caught diff mean {np.mean([g - t for g, t in zip(g_c, thr_c)]):.4f}")
h_c = json.load(open(L3H))["main_25"]["tiebreak"]["caught_per_seed"]
out(f"  L3h tiebreak caught mean {np.mean(h_c):.3f} (entry 335.13); admitted diff {-np.mean([(x - 334.25) for x in h_c]) / E:+.6f}")

out("")
out("Result of this check is written to " + OUT)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
