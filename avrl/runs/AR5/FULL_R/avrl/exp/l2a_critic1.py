"""L2a critic, part 1 (CPU): recompute every verdict number from the raw sim_*.json files, independent of l2a_summarize.py.
Usage: l2a_critic1.py <runs/L2a>"""
import sys, json, re, itertools
import numpy as np
rd = sys.argv[1]
NAMES = {"3B": "Qwen2.5-3B-Instruct", "7B": "Qwen2.5-7B-Instruct-AWQ"}
rng = np.random.default_rng(777)
def cl(L, F):
    ci = rng.integers(0, 4, (20000, 4)); cb = L[ci].mean((1, 2)) / F[ci].mean((1, 2))
    return np.percentile(cb, 2.5), np.percentile(cb, 97.5)
def pr(L, F):
    l, f = L.ravel(), F.ravel(); ix = rng.integers(0, 16, (20000, 16)); rb = l[ix].mean(1) / f[ix].mean(1)
    return np.percentile(rb, 2.5), np.percentile(rb, 97.5)
for lab, nm in NAMES.items():
    D = [json.load(open(f"{rd}/sim_{nm}_split{s}.json")) for s in range(4)]
    print(lab, "n_sim", [d["n_sim"] for d in D], "errors", [d["hits"] for d in D], "buckets", [d["n_buckets"] for d in D], "label", D[0]["label"], "features", D[0]["features"])
    for b in ("0.1", "0.25", "0.5"):
        pols = list(D[0][f"budget_{b}"])
        H = {p: np.array([d[f"budget_{b}"][p]["per_seed"] for d in D], float) for p in pols}
        budgets = [int(float(b) * d["n_sim"]) for d in D]
        # budget sanity: no policy exceeds its budget in hits
        mx = max(H[p][i].max() / budgets[i] for p in pols for i in range(4))
        fixed = [p for p in pols if p == "thr_agreement" or re.match(r"(recal|seeded|split)_charged_\d+$", p)]
        learn = [p for p in pols if p.startswith("greedy")]
        bf = max(fixed, key=lambda p: H[p][:, :4].mean()); bl = max(learn, key=lambda p: H[p][:, :4].mean())
        L, F = H[bl][:, 4:], H[bf][:, 4:]
        lo, hi = cl(L, F); plo, phi = pr(L, F)
        G = H["greedy_online"][:, 4:]; glo, ghi = cl(G, F)
        print(f" {lab} {float(b):.0%}: nF {len(fixed)} nL {len(learn)} fixed={bf} {F.mean():.1f} learned={bl} {L.mean():.1f} ratio={L.mean()/F.mean():.3f} cluster[{lo:.3f},{hi:.3f}] paired[{plo:.3f},{phi:.3f}] "
              f"splitSD={np.std(L.mean(1)/F.mean(1), ddof=1):.3f} | greedy_online {G.mean():.1f} ratio {G.mean()/F.mean():.3f} cluster[{glo:.3f},{ghi:.3f}] | max hits/budget {mx:.3f}")
        ev = {p: H[p][:, 4:].mean() for p in pols}
        print("    uniform %.1f thr %.1f static %.1f oracle %.1f" % (ev["uniform"], ev["thr_agreement"], ev["static"], ev["oracle"]))
        if b in ("0.25", "0.5"):
            print("    split ratios", np.round(L.mean(1) / F.mean(1), 3).tolist(), " per-split fixed means", np.round(F.mean(1), 1).tolist(), "learned", np.round(L.mean(1), 1).tolist())
            lo_ = [L[[i for i in range(4) if i != j]].mean() / F[[i for i in range(4) if i != j]].mean() for j in range(4)]
            print("    LOSO", np.round(lo_, 3).tolist())
            # held-out max over fixed and the 'best held-out member of one-shot families' relative to selected fixed
            one = [p for p in fixed]
            bho = max(one, key=lambda p: ev[p])
            print(f"    best held-out fixed {bho} {ev[bho]:.1f} -> /selected fixed {ev[bho]/F.mean():.3f}; learned/best-heldout-fixed {L.mean()/ev[bho]:.3f}")
            # sel-seeds vs held-out agreement: rank of the selected
            sel_means = {p: H[p][:, :4].mean() for p in fixed}
            top3 = sorted(sel_means, key=sel_means.get, reverse=True)[:4]
            print("    top4 fixed on sel seeds", [(p, round(sel_means[p], 1), round(ev[p], 1)) for p in top3])
            sel_l = {p: H[p][:, :4].mean() for p in learn}
            top3 = sorted(sel_l, key=sel_l.get, reverse=True)[:4]
            print("    top4 learned on sel seeds", [(p, round(sel_l[p], 1), round(ev[p], 1)) for p in top3])
            # seed-level spread: SD across seeds of the sim means
            print("    learned per-seed (mean over splits) seeds 4-7", np.round(L.mean(0), 1).tolist(), " fixed", np.round(F.mean(0), 1).tolist())
            # sel-seed ratio (the biased in-sample ratio)
            Ls, Fs = H[bl][:, :4], H[bf][:, :4]
            print("    in-selection-seeds ratio", round(Ls.mean() / Fs.mean(), 3))
            # ratio of learned to *each* fixed candidate on held-out: min across fixed candidates
            worst = max(fixed, key=lambda p: ev[p])
            print("    all held-out fixed (top 5):", sorted([(round(ev[p], 1), p) for p in fixed], reverse=True)[:5])
            print("    all held-out learned (top 5):", sorted([(round(ev[p], 1), p) for p in learn], reverse=True)[:5])
