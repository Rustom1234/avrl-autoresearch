"""L2b adversarial critic (CPU only). Re-derives every table number in the RESEARCH_LOG L2b entry from the 96 raw sim files and compares them.
Usage: python exp/l2b_critic.py"""
import json, re, os, sys
import numpy as np
RUN = "/homes/rdubash/avrl_live/FULL_R/avrl/runs/L2b"; LOG = "/homes/rdubash/avrl_live/FULL_R/avrl/RESEARCH_LOG.md"
J = {"3B": "Qwen2.5-3B-Instruct", "7B-AWQ": "Qwen2.5-7B-Instruct-AWQ"}; MS = (250, 500, 1000); BUD = ("0.1", "0.25", "0.5")
rng = np.random.default_rng(777)
fixed_re = re.compile(r"thr_agreement$|(recal|seeded|split)_charged_\d+$"); learn_re = re.compile(r"greedy_online$|greedy_n\d+_R\d+$|greedy_seeded_n25_R25$")
cells = {}
for lab, nm in J.items():
    for m in MS:
        D = [json.load(open(f"{RUN}/sim_{nm}_m{m}_s{s}.json")) for s in range(16)]
        for s, d in enumerate(D):
            assert d["m"] == m and d["subset_seed"] == s and d["n_sim"] == m and d["n_policy_seeds"] == 8 and d["label"] == "judge_error"
            assert d["features"] == "agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std".split(",")
        for b in BUD:
            names = list(D[0][f"budget_{b}"]); assert all(list(d[f"budget_{b}"]) == names for d in D)
            raw = {n: np.array([d[f"budget_{b}"][n]["per_seed"] for d in D], float) for n in names}  # 16 x 8
            for n in names:
                assert raw[n].shape == (16, 8)
                for d in D: assert abs(np.mean(d[f"budget_{b}"][n]["per_seed"]) - d[f"budget_{b}"][n]["mean"]) < 1e-9
            budget = int(float(b) * m)
            cells[(lab, m, b)] = dict(raw=raw, budget=budget, errs=[d["hits"] for d in D])
            # budget sanity: no policy may catch more than the budget or more than the errors in the subset
            for n in names:
                assert (raw[n] <= budget).all(), (lab, m, b, n)
                assert (raw[n] <= np.array([d["hits"] for d in D])[:, None]).all(), (lab, m, b, n)
            fx = [n for n in names if fixed_re.match(n)]; ln = [n for n in names if learn_re.match(n)]
            # all-families coverage check: nothing in the policy list is unclassified
            other = set(names) - set(fx) - set(ln) - {"uniform", "static", "oracle"}
            assert not other, other
            cells[(lab, m, b)].update(fixed=fx, learned=ln)

def foldr(c, sel, rep):
    raw = c["raw"]; per = {n: raw[n].mean(1) for n in raw}
    bf = max(c["fixed"], key=lambda n: per[n][sel].mean()); bl = max(c["learned"], key=lambda n: per[n][sel].mean())
    L, F = per[bl][rep], per[bf][rep]; idx = rng.integers(0, len(L), (20000, len(L))); rb = L[idx].mean(1) / F[idx].mean(1)
    return bf, bl, L.mean() / F.mean(), np.percentile(rb, 2.5), np.percentile(rb, 97.5), L.mean(), F.mean(), L, F
A, Bf = np.arange(0, 8), np.arange(8, 16)
res = {}
for k, c in cells.items():
    a = foldr(c, A, Bf); b = foldr(c, Bf, A)
    pooled = (np.concatenate([a[7], b[7]]).mean() / np.concatenate([a[8], b[8]]).mean())
    pl, pf = np.concatenate([a[7], b[7]]), np.concatenate([a[8], b[8]]); idx = rng.integers(0, 16, (20000, 16)); rb = pl[idx].mean(1) / pf[idx].mean(1)
    res[k] = dict(a=a, b=b, pooled=pooled, pooled_ci=(np.percentile(rb, 2.5), np.percentile(rb, 97.5)))
    # extra: held-out means of reference arms on 8..15
    raw = c["raw"]; res[k]["ho"] = {n: raw[n][8:16].mean() for n in raw}; res[k]["all16"] = {n: raw[n].mean() for n in raw}

# --- compare against the ledger table
txt = open(LOG).read(); ent = txt[txt.index("## L2b 2026-10-18"):]
rows = [l for l in ent.split("\n") if re.match(r"\| (3B|7B-AWQ) \| (250|500|1000) \| \d+% \(B=", l)]
print("ledger table rows found:", len(rows)); bad = 0
for l in rows:
    p = [x.strip() for x in l.strip("|").split("|")]; lab, m, bud = p[0], int(p[1]), {"10": "0.1", "25": "0.25", "50": "0.5"}[re.match(r"(\d+)%", p[2]).group(1)]
    B_abs = int(re.search(r"B=(\d+)", p[2]).group(1)); r = res[(lab, m, bud)]; c = cells[(lab, m, bud)]; ho = r["ho"]
    chk = []
    chk.append(("B", B_abs == c["budget"]))
    for name, col in (("uniform", 3), ("thr_agreement", 4), ("static", 8), ("oracle", 9)):
        v = float(p[col].split()[-1]) if name == "thr_agreement" else float(p[col]); chk.append((name, abs(v - ho[name]) < 0.051))
    fx = p[5].split(); lr = p[6].split(); chk += [("fixed_name", fx[0] == r["a"][0]), ("fixed_val", abs(float(fx[1]) - ho[fx[0]]) < 0.051), ("learned_name", lr[0] == r["a"][1]), ("learned_val", abs(float(lr[1]) - ho[lr[0]]) < 0.051)]
    chk.append(("greedy_online", abs(float(p[7]) - ho["greedy_online"]) < 0.051))
    mm = re.match(r"([\d.]+) \(([\d.]+) to ([\d.]+)\)", p[10]); rt, lo, hi = map(float, mm.groups())
    chk += [("ratio", abs(rt - r["a"][2]) < 0.0006), ("ci_lo", abs(lo - r["a"][3]) < 0.004), ("ci_hi", abs(hi - r["a"][4]) < 0.004), ("swapped", abs(float(p[11]) - r["b"][2]) < 0.0006), ("pooled", abs(float(p[12]) - r["pooled"]) < 0.0006)]
    fails = [n for n, ok in chk if not ok]
    if fails: bad += 1; print("MISMATCH", lab, m, bud, fails, "ledger row:", l[:200])
print("table rows with mismatches:", bad)

print("\nindependent summary (fold A): label m budget B | ratio (CI) | fixed / learned | swapped | pooled (CI)")
for k, r in res.items():
    lab, m, b = k; a = r["a"]
    print(f"{lab:6s} m={m:4d} {float(b):.0%} B={cells[k]['budget']:3d} | {a[2]:.3f} ({a[3]:.3f} to {a[4]:.3f}) | {a[0]} / {a[1]} | {r['b'][2]:.3f} | {r['pooled']:.3f} ({r['pooled_ci'][0]:.3f} to {r['pooled_ci'][1]:.3f})")

# --- verdict
VB = ("0.25", "0.5"); pass_c = {(lab, b): (res[(lab, 500, b)]["a"][2] >= 1.03 and res[(lab, 500, b)]["a"][3] > 1.0) for lab in J for b in VB}
kill_c = {(lab, m, b): res[(lab, m, b)]["a"][2] <= 1.0 for lab in J for m in (250, 500) for b in VB}
print("\nPASS cells", pass_c, sum(pass_c.values())); print("KILL cells", kill_c, sum(kill_c.values()))
# pooled-16 CI variant of the pass rule (spec text says CI over the 16 subsets)
print("pass under pooled-16 rule:", {(lab, b): (res[(lab, 500, b)]["pooled"] >= 1.03 and res[(lab, 500, b)]["pooled_ci"][0] > 1) for lab in J for b in VB})

# --- ledger claims
print("\nclaim: n10_R5 best learned on all-16 mean in 16 of 18 cells")
cnt = 0
for k, c in cells.items():
    a16 = res[k]["all16"]; best = max(c["learned"], key=lambda n: a16[n]); cnt += best == "greedy_n10_R5"
    if best != "greedy_n10_R5": print("  not n10_R5:", k, best, {n: round(a16[n], 1) for n in c["learned"]})
print("  count", cnt)
print("claim: thr_agreement selected on 0-7 in 15 of 18:", sum(res[k]["a"][0] == "thr_agreement" for k in cells), [(k, res[k]["a"][0]) for k in cells if res[k]["a"][0] != "thr_agreement"])
print("claim: static above both arms (held-out) in every cell:", all(res[k]["ho"]["static"] > max(res[k]["ho"][res[k]["a"][0]], res[k]["ho"][res[k]["a"][1]]) for k in cells))
print("static headroom over learner:", {k: round(res[k]["ho"]["static"] / res[k]["ho"][res[k]["a"][1]] - 1, 3) for k in cells if cells[k]["budget"] >= 250})
print("claim: greedy_online/selected fixed at m=500:", {k: round(res[k]["ho"]["greedy_online"] / res[k]["ho"][res[k]["a"][0]], 3) for k in cells if k[1] == 500 and k[2] != "0.1"})
print("claim m500 25% (25,25) vs (10,5), all16:", {lab: (round(res[(lab, 500, "0.25")]["all16"]["greedy_n25_R25"], 1), round(res[(lab, 500, "0.25")]["all16"]["greedy_n10_R5"], 1)) for lab in J})
print("pooled fixed rule m500 25% (pooled cross-fit fixed mean):", {lab: round(float(np.concatenate([res[(lab, 500, "0.25")]["a"][8], res[(lab, 500, "0.25")]["b"][8]]).mean()), 1) for lab in J})
print("claim 7B m500 50% split curve all16:", {n: round(res[("7B-AWQ", 500, "0.5")]["all16"][n], 1) for n in ("split_charged_50", "split_charged_75", "split_charged_100", "thr_agreement")})
print("largest allowed n per cell (m500 50%):", [n for n in cells[("7B-AWQ", 500, "0.5")]["raw"] if n.startswith("split")])
print("claim: 3B m=500 50% static_out:", res[("3B", 500, "0.5")]["ho"]["static"], "7B:", res[("7B-AWQ", 500, "0.5")]["ho"]["static"])
print("mean errors per subset:", {k: round(float(np.mean(cells[k]["errs"])), 1) for k in cells if k[2] == "0.1"})
# per-subset ratio spread at m=500 25% cells and subset-level win counts
for lab in J:
    for b in VB:
        a = res[(lab, 500, b)]["a"]; print(lab, "m500", b, "per-subset ratios fold A:", np.round(a[7] / a[8], 3), "n>1:", int((a[7] / a[8] > 1).sum()))
# overlap of subsets
for m in MS:
    S = [set(np.random.default_rng(s).choice(3000, m, replace=False)) for s in range(16)]
    ov = [len(S[i] & S[j]) / m for i in range(16) for j in range(i + 1, 16)]; print("m", m, "mean pairwise overlap", round(float(np.mean(ov)), 3))
