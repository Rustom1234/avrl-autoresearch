"""L2a critic, part 3 (CPU): re-run the L2a verdict with the enlarged fixed family (original candidates plus runs/L2a/critic_extra_fixed.json).
Same protocol: choose on policy seeds 0..3 averaged over the 4 splits, report seeds 4..7 over 4 splits, learned candidate set unchanged.
Usage: l2a_critic3.py <runs/L2a> [extra json names]"""
import sys, json, re
import numpy as np
rd = sys.argv[1]; NAMES = {"3B": "Qwen2.5-3B-Instruct", "7B": "Qwen2.5-7B-Instruct-AWQ"}; rng = np.random.default_rng(5)
extra = {}
for fn in sys.argv[2:] or ["critic_extra_fixed.json"]:
    for k, v in json.load(open(f"{rd}/{fn}")).items(): extra.setdefault(k, {}).update(v)
def cl(L, F):
    ci = rng.integers(0, 4, (20000, 4)); cb = L[ci].mean((1, 2)) / F[ci].mean((1, 2)); return np.percentile(cb, 2.5), np.percentile(cb, 97.5)
for lab, nm in NAMES.items():
    D = [json.load(open(f"{rd}/sim_{nm}_split{s}.json")) for s in range(4)]
    for b in ("0.25", "0.5"):
        pols = list(D[0][f"budget_{b}"])
        H = {p: np.array([d[f"budget_{b}"][p]["per_seed"] for d in D], float) for p in pols}
        fixed0 = [p for p in pols if p == "thr_agreement" or re.match(r"(recal|seeded|split)_charged_\d+$", p)]
        learn = [p for p in pols if p.startswith("greedy")]
        HX = {}
        for p in set(k for s in range(4) for k in extra[f"{lab}|{s}|{float(b)}"]):
            if all(p in extra[f"{lab}|{s}|{float(b)}"] for s in range(4)): HX[p] = np.array([extra[f"{lab}|{s}|{float(b)}"][p] for s in range(4)], float)
        import os
        if os.environ.get("CAP"): capn = min(int(float(b) * d["n_sim"]) // 2 for d in D); HX = {k: v for k, v in HX.items() if int(k.rsplit("_", 1)[1]) <= capn}
        HH = {**H, **HX}; fixed1 = fixed0 + list(HX)
        bl = max(learn, key=lambda p: H[p][:, :4].mean()); L = H[bl][:, 4:]
        b0 = max(fixed0, key=lambda p: HH[p][:, :4].mean()); b1 = max(fixed1, key=lambda p: HH[p][:, :4].mean())
        F0, F1 = HH[b0][:, 4:], HH[b1][:, 4:]
        l0, h0 = cl(L, F0); l1, h1 = cl(L, F1)
        ev = lambda p: HH[p][:, 4:].mean()
        mxh = max(fixed1, key=ev)
        G = H["greedy_online"][:, 4:]
        print(f"{lab} {float(b):.0%}: learned {bl} {L.mean():.1f} | orig fixed {b0} {F0.mean():.1f} ratio {L.mean()/F0.mean():.3f} [{l0:.3f},{h0:.3f}] | enlarged fixed {b1} {F1.mean():.1f} ratio {L.mean()/F1.mean():.3f} [{l1:.3f},{h1:.3f}] "
              f"| greedy_online/enlarged {G.mean()/F1.mean():.3f} | held-out max enlarged fixed {mxh} {ev(mxh):.1f} -> learned/that {L.mean()/ev(mxh):.3f}; greedy_online/that {G.mean()/ev(mxh):.3f}")
        print("    extra-n held-out means:", {p: round(ev(p), 1) for p in sorted(HX, key=lambda q: (q.split('_')[0], int(q.rsplit('_', 1)[1])))})
        print("    extra-n sel-seed means:", {p: round(HX[p][:, :4].mean(), 1) for p in sorted(HX, key=lambda q: (q.split('_')[0], int(q.rsplit('_', 1)[1])))})
