"""L6a critic, part 2 (CPU, read-only on runs/): (A) does the verdict survive the spec's 'global label-free standardization' fixed rule (exp/l6a_critic_gz.py, runs/L6a_critic_gz) and the union of both fixed families;
(B) like-for-like post-calibration comparison of amort_{seed,rand,split}_n with freeze_plain_n at equal n (same k) to check the 'which labels, not how many' decomposition."""
import json, math
import numpy as np
R = "/homes/rdubash/avrl_live/FULL_R/avrl/runs/"
J = {"3B": "Qwen2.5-3B-Instruct", "7B": "Qwen2.5-7B-Instruct-AWQ"}; CELLS = [(l, j, B) for l, j in J.items() for B in (62, 125)]
FAM = [f"amort_{s}_{n}" for s in ("seed", "rand", "split") for n in (125, 250, 500)]; GZ = [a.replace("amort_", "amortgz_") for a in FAM]
D, G = {}, {}
for l, j, B in CELLS:
    for o in range(16):
        D[(l, B, o)] = json.load(open(f"{R}L6a/amort_{j}_B{B}_o{o}.json"))["arms"]; G[(l, B, o)] = json.load(open(f"{R}L6a_critic_gz/gz_{j}_B{B}_o{o}.json"))["arms"]
def get(l, B, arm, sl=slice(None), post=None):
    src = G if arm.startswith("amortgz") else D
    return np.array([sum(src[(l, B, o)][arm][post:] if post is not None else src[(l, B, o)][arm]) for o in range(16)], float)[sl]
rng = np.random.default_rng(99)
def boot(num, den, nb=20000):
    idx = rng.integers(0, len(num), (nb, len(num))); rb = num[idx].mean(1) / den[idx].mean(1); return num.mean() / den.mean(), np.percentile(rb, 2.5), np.percentile(rb, 97.5)
A, Bf = slice(0, 8), slice(8, 16)
print("== (A) fixed rule on global label-free standardization (GZ): arm / carry_plain, all 16 seeds")
for arm in GZ:
    print(f"{arm:20s}", " | ".join(f"{get(l,B,arm).mean()/get(l,B,'carry_plain').mean():.3f} (orig {get(l,B,arm.replace('amortgz','amort')).mean()/get(l,B,'carry_plain').mean():.3f})" for l, j, B in CELLS))
for name, fam in (("GZ family only", GZ), ("union of orig + GZ (18 arms)", FAM + GZ)):
    print(f"-- verdict fold, {name}: select on seeds 0-7, report 8-15")
    npass = nkill = 0
    for l, j, B in CELLS:
        a = max(fam, key=lambda k: get(l, B, k, A).mean()); cp = get(l, B, "carry_plain", Bf); r = boot(cp, get(l, B, a, Bf))
        ph = max(fam, key=lambda k: get(l, B, k).mean())
        p = r[0] >= 1.03 and r[1] > 1; kl = 1 / r[0] >= 0.99; npass += p; nkill += kl
        print(f"  {l} B={B}: {a} carry_plain/best {r[0]:.3f} CI {r[1]:.3f}-{r[2]:.3f} best/cp {1/r[0]:.3f} pass={p} kill={kl} | post hoc best on 16: {ph} {get(l,B,ph).mean()/get(l,B,'carry_plain').mean():.3f}")
    print(f"  PASS cells {npass}/4, KILL cells {nkill}/4")

print("\n== (B) like-for-like after calibration (batches k+1..12), pooled 16 seeds, per-batch hits relative to freeze_plain_n and to carry_plain")
print("cell n k | amort_seed/freeze | amort_rand/freeze | amort_split/freeze | freeze/carry_plain | calibration phase (batches 1..k) arm/carry_plain: seed rand split freeze")
for l, j, B in CELLS:
    for n in (125, 250, 500):
        k = math.ceil(n / B)
        if k >= 12: continue
        fz = get(l, B, f"freeze_plain_{n}", post=k).mean(); cp = get(l, B, "carry_plain", post=k).mean()
        post = [get(l, B, f"amort_{s}_{n}", post=k).mean() / fz for s in ("seed", "rand", "split")]
        cal = [(get(l, B, f"amort_{s}_{n}").mean() - get(l, B, f"amort_{s}_{n}", post=k).mean()) / (get(l, B, "carry_plain").mean() - get(l, B, "carry_plain", post=k).mean()) for s in ("seed", "rand", "split")]
        calf = (get(l, B, f"freeze_plain_{n}").mean() - fz * 0 - get(l, B, f"freeze_plain_{n}", post=k).mean()) / (get(l, B, "carry_plain").mean() - get(l, B, "carry_plain", post=k).mean())
        print(f"{l} B={B} n={n} k={k} | " + " | ".join(f"{x:.3f}" for x in post) + f" | {fz/cp:.3f} | " + " ".join(f"{x:.3f}" for x in cal + [calf]))
