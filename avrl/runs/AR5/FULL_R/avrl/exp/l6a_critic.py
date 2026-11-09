"""L6a adversarial critic, part 1: re-derive the L6a numbers from the 64 raw JSONs and runs/L5a with independent code (own bootstrap seeds).
Usage: l6a_critic.py  (CPU only, read-only on runs/)"""
import json, glob, math, os, sys
import numpy as np

R = "/homes/rdubash/avrl_live/FULL_R/avrl/runs/"
J = {"3B": "Qwen2.5-3B-Instruct", "7B": "Qwen2.5-7B-Instruct-AWQ"}
CELLS = [(l, j, B) for l, j in J.items() for B in (62, 125)]
FAM = [f"amort_{s}_{n}" for s in ("seed", "rand", "split") for n in (125, 250, 500)]
FZ = [f"freeze_plain_{n}" for n in (125, 250, 500)]
D = {}
bad = []
for l, j, B in CELLS:
    for o in range(16):
        f = f"{R}L6a/amort_{j}_B{B}_o{o}.json"
        d = json.load(open(f)); D[(l, B, o)] = d
        assert d["budget"] == B and d["m"] == 250 and d["n_batches"] == 12 and d["order_seed"] == o and d["policy_seed"] == o
        be = d["batch_base_errors"]
        for k, v in d["arms"].items():
            assert len(v) == 12, (f, k)
            for b in range(12):
                if not (0 <= v[b] <= min(B, be[b])): bad.append((f, k, b, v[b], B, be[b]))
        assert all(d["checks"].values()), f
        for k, inf in d["arm_info"].items():
            if inf["fallback_thr"]: bad.append((f, k, "fallback"))
print("integrity: files", len(D), "hit-bound violations / fallbacks:", bad[:5], len(bad))
# base errors identical across B for the same judge and order seed
for l, j, B in CELLS:
    for o in range(16):
        assert D[(l, B, o)]["batch_base_errors"] == D[(l, 62, o)]["batch_base_errors"]
print("batch_base_errors equal across B: OK; sums per judge:", {l: sorted({sum(D[(l, 62, o)]['batch_base_errors']) for o in range(16)})[:3] for l in J})

# gate vs L5a
ref = ["thr_agreement", "uniform", "restart_offset", "carry_plain", "carry_offset_lam1", "static_out"]
nd = nv = nvec = 0
for l, j, B in CELLS:
    for o in range(8):
        g = json.load(open(f"{R}L5a/carry_{j}_B{B}_o{o}.json")); d = D[(l, B, o)]
        for k in ref + ["batch_base_errors"]:
            a = d["batch_base_errors"] if k == "batch_base_errors" else d["arms"][k]
            r = g["batch_base_errors"] if k == "batch_base_errors" else g["arms"][k]
            r = r["hits"] if isinstance(r, dict) else r
            nvec += 1; nv += len(r); nd += sum(int(x != y) for x, y in zip(a, r))
print(f"gate: {nd} differences in {nv} values ({nvec} vectors)")

def T(l, B, arm, seeds=range(16)): return np.array([sum(D[(l, B, o)]["arms"][arm]) for o in seeds], float)
def S(l, B, arm, seeds=range(16)): return np.array([sum(D[(l, B, o)]["arms"][arm][6:]) for o in seeds], float)
rng = np.random.default_rng(31337)
def boot(num, den, nb=20000):
    n = len(num); idx = rng.integers(0, n, (nb, n)); rb = num[idx].mean(1) / den[idx].mean(1)
    return num.mean() / den.mean(), np.percentile(rb, 2.5), np.percentile(rb, 97.5)

print("\n== pooled 16 table: arm / thr (ratio), arm / carry_plain")
for arm in ["static_out", "carry_plain", "carry_offset_lam1"] + FAM + FZ:
    row = []
    for l, j, B in CELLS:
        t, c, th = T(l, B, arm), T(l, B, "carry_plain"), T(l, B, "thr_agreement")
        r1 = boot(t, th); row.append(f"{r1[0]:.3f}({r1[1]:.3f}-{r1[2]:.3f}) /{t.mean()/c.mean():.3f}")
    print(f"{arm:20s}", " | ".join(row))
print("thr abs:", [round(T(l, B, "thr_agreement").mean(), 1) for l, j, B in CELLS])

print("\n== verdict fold (select on 0-7 by mean total hits, report 8-15), swapped, pooled")
best = {}
for l, j, B in CELLS:
    a = max(FAM, key=lambda k: T(l, B, k, range(8)).mean()); b = max(FAM, key=lambda k: T(l, B, k, range(8, 16)).mean()); best[(l, B)] = (a, b)
    cp = T(l, B, "carry_plain")
    rA = boot(cp[8:], T(l, B, a)[8:]); rB = boot(cp[:8], T(l, B, b)[:8])
    num = np.concatenate([cp[:8], cp[8:]]); den = np.concatenate([T(l, B, b)[:8], T(l, B, a)[8:]]); rP = boot(num, den)
    print(f"{l} B={B}: A best {a}: {rA[0]:.4f} CI {rA[1]:.4f}-{rA[2]:.4f} best/cp {1/rA[0]:.3f} | B best {b}: {rB[0]:.4f} ({rB[1]:.3f}) | pooled {rP[0]:.4f} ({rP[1]:.3f}-{rP[2]:.3f})")
    pa = (cp > T(l, B, a)).sum(); print(f"    carry_plain > verdict-fold arm on {pa}/16 seeds, min per-seed ratio {(cp/T(l,B,a)).min():.3f}; on seeds 8-15 only: {(cp[8:]>T(l,B,a)[8:]).sum()}/8")
tot = sum(int((T(l, B, 'carry_plain') > T(l, B, best[(l, B)][0])).sum()) for l, j, B in CELLS)
print("carry_plain beats verdict-fold arm in", tot, "of 64 (cell, seed) pairs")
print("post hoc best on all 16:", {(l, B): (max(FAM, key=lambda k: T(l, B, k).mean()), round(max(T(l, B, k).mean() for k in FAM) / T(l, B, 'carry_plain').mean(), 3)) for l, j, B in CELLS})

print("\n== spec-original fold: 8 order seeds only, select on 0-3, report 4-7; swapped; pooled 8")
for l, j, B in CELLS:
    a = max(FAM, key=lambda k: T(l, B, k, range(4)).mean()); b = max(FAM, key=lambda k: T(l, B, k, range(4, 8)).mean()); cp = T(l, B, "carry_plain")
    rA = boot(cp[4:8], T(l, B, a)[4:8]); rB = boot(cp[:4], T(l, B, b)[:4]); rP = boot(np.r_[cp[:4], cp[4:8]], np.r_[T(l, B, b)[:4], T(l, B, a)[4:8]])
    print(f"{l} B={B}: sel0-3 {a}: {rA[0]:.3f} ({rA[1]:.3f}-{rA[2]:.3f}) | swapped {b}: {rB[0]:.3f} ({rB[1]:.3f}) | pooled8 {rP[0]:.3f} ({rP[1]:.3f}) | per-seed min on 4-7: {(cp[4:8]/T(l,B,a)[4:8]).min():.3f}")

print("\n== steady state batches 7-12, ratio to thr; best fixed arm on 16 seeds by steady hits (all 9) and restricted to arms with k<=6")
for l, j, B in CELLS:
    th, cp = S(l, B, "thr_agreement"), S(l, B, "carry_plain")
    ks = {k: math.ceil(int(k.split('_')[-1]) / B) for k in FAM}
    ba = max(FAM, key=lambda k: S(l, B, k).mean()); bk = max([k for k in FAM if ks[k] <= 6], key=lambda k: S(l, B, k).mean())
    print(f"{l} B={B}: carry {cp.mean()/th.mean():.3f}; best of 9 {ba} {S(l,B,ba).mean()/th.mean():.3f} ({S(l,B,ba).mean()/cp.mean():.3f} of cp); best k<=6 {bk} {S(l,B,bk).mean()/th.mean():.3f} ({S(l,B,bk).mean()/cp.mean():.3f} of cp); freeze_500 {S(l,B,'freeze_plain_500').mean()/cp.mean():.3f}")

print("\n== uniform / thr per cell (pooled 16, 12 batches)")
print([round(T(l, B, "uniform").mean() / T(l, B, "thr_agreement").mean(), 3) for l, j, B in CELLS])
print("random-pick error rate (uniform/(12B)):", [round(T(l, B, "uniform").mean() / (12 * B), 3) for l, j, B in CELLS])

print("\n== fit-set error rates (mean over 16 seeds)")
for l, j, B in CELLS:
    s = {k: np.mean([D[(l, B, o)]["arm_info"][k].get("cal_err_rate", D[(l, B, o)]["arm_info"][k].get("fit_err_rate")) for o in range(16)]) for k in FAM + FZ}
    print(l, B, "seed", [round(s[f"amort_seed_{n}"], 2) for n in (125, 250, 500)], "rand", [round(s[f"amort_rand_{n}"], 2) for n in (125, 250, 500)], "split", [round(s[f"amort_split_{n}"], 2) for n in (125, 250, 500)], "freeze", [round(s[k], 2) for k in FZ],
          "n_fit seed", [D[(l, B, 0)]["arm_info"][f"amort_seed_{n}"]["n_fit"] for n in (125, 250, 500)])

print("\n== count of (arm, cell) with arm/thr >= 1.03 (pooled 16) and arm/cp >= 0.99")
c103 = []; c99 = []
for l, j, B in CELLS:
    for k in FAM:
        if T(l, B, k).mean() / T(l, B, "thr_agreement").mean() >= 1.03: c103.append((k, l, B))
        if T(l, B, k).mean() / T(l, B, "carry_plain").mean() >= 0.99: c99.append((k, l, B))
print(len(c103), c103, "| arms reaching 0.99 carry_plain:", c99)
print("ranges arm/cp:", {s: (round(min(T(l,B,f'amort_{s}_{n}').mean()/T(l,B,'carry_plain').mean() for l,j,B in CELLS for n in (125,250,500)),3), round(max(T(l,B,f'amort_{s}_{n}').mean()/T(l,B,'carry_plain').mean() for l,j,B in CELLS for n in (125,250,500)),3)) for s in ('seed','rand','split')})
