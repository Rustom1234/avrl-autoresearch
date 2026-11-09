"""L2a critic, part 5 (CPU): verdict under four grid choices, same selection protocol (choose on seeds 0..3 over 4 splits, report seeds 4..7).
  A original families (reproduces L2a)   B fixed family enlarged   C learned family enlarged   D both enlarged
Fixed enlargement: extra n files given as args (critic_extra_fixed*.json); set CAP=1 to keep only n <= budget // 2. Learned enlargement: critic_learned_<lab>_<split>.json.
Usage: l2a_critic5.py <runs/L2a> [extra fixed json names]"""
import sys, json, re, os
import numpy as np
rd = sys.argv[1]; NAMES = {"3B": "Qwen2.5-3B-Instruct", "7B": "Qwen2.5-7B-Instruct-AWQ"}; rng = np.random.default_rng(11)
extra = {}
for fn in sys.argv[2:]:
    for k, v in json.load(open(f"{rd}/{fn}")).items(): extra.setdefault(k, {}).update(v)
def cl(L, F):
    ci = rng.integers(0, 4, (20000, 4)); cb = L[ci].mean((1, 2)) / F[ci].mean((1, 2)); return np.percentile(cb, 2.5), np.percentile(cb, 97.5)
for lab, nm in NAMES.items():
    D = [json.load(open(f"{rd}/sim_{nm}_split{s}.json")) for s in range(4)]
    LX = [json.load(open(f"{rd}/critic_learned_{lab}_{s}.json")) for s in range(4)]
    for b in ("0.25", "0.5"):
        pols = list(D[0][f"budget_{b}"])
        H = {p: np.array([d[f"budget_{b}"][p]["per_seed"] for d in D], float) for p in pols}
        fixed0 = [p for p in pols if p == "thr_agreement" or re.match(r"(recal|seeded|split)_charged_\d+$", p)]; learn0 = [p for p in pols if p.startswith("greedy")]
        HX = {p: np.array([extra[f"{lab}|{s}|{float(b)}"][p] for s in range(4)], float) for p in set(k for s in range(4) for k in extra.get(f"{lab}|{s}|{float(b)}", {})) if all(p in extra[f"{lab}|{s}|{float(b)}"] for s in range(4))}
        if os.environ.get("CAP"): capn = min(int(float(b) * d["n_sim"]) // 2 for d in D); HX = {k: v for k, v in HX.items() if int(k.rsplit("_", 1)[1]) <= capn}
        HL = {p: np.array([lx[b][p] for lx in LX], float) for p in LX[0][b]}
        HH = {**H, **HX, **HL}; fixed1 = fixed0 + list(HX); learn1 = learn0 + list(HL)
        res = {}
        for tag, fx, ln in (("A orig", fixed0, learn0), ("B fixed+", fixed1, learn0), ("C learned+", fixed0, learn1), ("D both+", fixed1, learn1)):
            bf = max(fx, key=lambda p: HH[p][:, :4].mean()); bl = max(ln, key=lambda p: HH[p][:, :4].mean()); L, F = HH[bl][:, 4:], HH[bf][:, 4:]; lo, hi = cl(L, F)
            res[tag] = f"{tag}: {bl} {L.mean():.1f} / {bf} {F.mean():.1f} = {L.mean()/F.mean():.3f} [{lo:.3f},{hi:.3f}] splits {np.round(L.mean(1)/F.mean(1), 3).tolist()}"
        G = H["greedy_online"][:, 4:]
        print(f"{lab} {float(b):.0%}")
        for v in res.values(): print("   ", v)
        print("    new learned held-out means:", {p: round(HH[p][:, 4:].mean(), 1) for p in HL}, "sel-seed:", {p: round(HH[p][:, :4].mean(), 1) for p in HL})
