"""L9a adversarial critic (CPU). Separate implementation: reads only the raw hz_*, gate_*, nr_*, hz120_* JSONs and runs/L7b, recomputes the ledger numbers. Usage: l9a_critic.py <runs/L9a> <runs/L7b>"""
import sys, os, json, glob
import numpy as np

rd, r7 = sys.argv[1], sys.argv[2]
J = {"3B": "Qwen2.5-3B-Instruct", "7B": "Qwen2.5-7B-Instruct-AWQ"}; BS = (62, 125); CELLS = [(a, b) for a in J for b in BS]
CJ = "agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen,judge_top,judge_margin,judge_mean,judge_std".split(",")
fx = [f"amort_{k}_{n}" for k in ("strat", "split", "seed") for n in (125, 250, 500, 1000)]
def ld(stage, j, B, s): return json.load(open(f"{rd}/{stage}_{J[j]}_B{B}_{'o' if stage == 'gate' else 's'}{s}.json"))
D = {st: {(j, B): [ld(st, j, B, s) for s in range(16)] for j, B in CELLS} for st in ("hz", "gate", "nr", "hz120")}

# ---- A. integrity
bad = []
for st, nb in (("hz", 60), ("gate", 12), ("hz120", 120), ("nr", None)):
    for (j, B), docs in D[st].items():
        for s, d in enumerate(docs):
            n = len(d["batch_base_errors"])
            if d["budget"] != B or d["order_seed"] != s or d["features"] != CJ or d["label"] != "judge_error" or d["m"] != 250: bad.append((st, j, B, s, "meta"))
            if nb is not None and (n != nb or d["n_batches"] != nb): bad.append((st, j, B, s, "nb", n))
            if st == "nr" and d["n_batches"] != n: bad.append((st, j, B, s, "nr nb"))
            if d["mode"] != {"hz": "iid", "hz120": "iid", "nr": "iid_nr", "gate": "disjoint12"}[st]: bad.append((st, j, B, s, "mode"))
            if not all(d["checks"].values()): bad.append((st, j, B, s, "checks"))
            for k, v in d["arms"].items():
                if len(v) != n: bad.append((st, j, B, s, k, "len"))
                v = np.array(v); be = np.array(d["batch_base_errors"])
                if (v < 0).any() or (v > B).any() or (v > be).any(): bad.append((st, j, B, s, k, "range"))
print("A. integrity problems:", bad[:10], len(bad))
# base errors independent of B within judge/seed (iid stream defined by seed only), and hz120 prefix == hz
be_ok = all(D["hz"][(j, 62)][s]["batch_base_errors"] == D["hz"][(j, 125)][s]["batch_base_errors"] for j in J for s in range(16)); print("   base errors equal across B (iid):", be_ok)
pref = all(D["hz120"][c][s]["arms"][k][:60] == D["hz"][c][s]["arms"][k] and D["hz120"][c][s]["batch_base_errors"][:60] == D["hz"][c][s]["batch_base_errors"] for c in CELLS for s in range(16) for k in D["hz"][c][s]["arms"])
print("   hz120 prefix-60 == hz for all arms, all 64:", pref)
# unique arms per stage
print("   hz arms:", len(D["hz"][CELLS[0]][0]["arms"]), sorted(D["hz"][CELLS[0]][0]["arms"])[:3], " gate arms:", len(D["gate"][CELLS[0]][0]["arms"]))
# ---- B. gate
nd = nv = nvec = 0; miss = 0
for (j, B), docs in D["gate"].items():
    for o, d in enumerate(docs):
        f = f"{r7}/design_{J[j]}_B{B}_o{o}.json"
        if not os.path.exists(f): miss += 1; continue
        ref = json.load(open(f)); assert set(ref["arms"]) <= set(d["arms"]), set(ref["arms"]) - set(d["arms"])
        extra = set(d["arms"]) - set(ref["arms"])
        for k in list(ref["arms"]) + ["batch_base_errors"]:
            a = d["batch_base_errors"] if k == "batch_base_errors" else d["arms"][k]; r = ref["batch_base_errors"] if k == "batch_base_errors" else ref["arms"][k]
            nvec += 1; nv += len(r); nd += sum(x != y for x, y in zip(a, r)) + abs(len(a) - len(r))
print(f"B. gate: missing {miss}, vectors {nvec}, values {nv}, differences {nd}, arms in ref {len(ref['arms'])}, extra arms in gate not in ref {sorted(extra)}")
# ---- helper arrays
def A(stage, c, arm, H=None):
    a = np.array([d["arms"][arm] for d in D[stage][c]], float); return a if H is None else a[:, :H]
rng = np.random.default_rng(777); NB = 10000; ix8 = rng.integers(0, 8, (NB, 8))
def ci(num, den):
    r = num[ix8].mean(1) / den[ix8].mean(1); return num.mean() / den.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)
FA, FB = slice(0, 8), slice(8, 16)
def best(c, stage, sl, lo, hi, stream=None):
    sc = {k: A(stage, c, k)[sl, lo:hi].sum(1).mean() for k in fx}; m = max(sc.values()); return [k for k in fx if sc[k] == m][0], sc
def seg(stage, c, arm, sl, lo, hi): return A(stage, c, arm)[sl, lo:hi].sum(1)
print("C. headline (carry/best, fold A; CI own bootstrap seed 777)")
cin = {}
for c in CELLS:
    row = []
    for H in (12, 24, 36, 60):
        b, sc = best(c, "hz", FA, 0, H); r = ci(seg("hz", c, "carry_plain", FB, 0, H), seg("hz", c, b, FB, 0, H)); row.append(f"H{H} {b} {r[0]:.3f} ({r[1]:.3f},{r[2]:.3f})")
        if H == 60:
            bB, _ = best(c, "hz", FB, 0, H); rB = seg("hz", c, "carry_plain", FA, 0, H).mean() / seg("hz", c, bB, FA, 0, H).mean()
            num = np.r_[seg("hz", c, "carry_plain", FB, 0, H), seg("hz", c, "carry_plain", FA, 0, H)]; den = np.r_[seg("hz", c, b, FB, 0, H), seg("hz", c, bB, FA, 0, H)]
            row.append(f"[swapped {bB} {rB:.3f}; pooled {num.mean()/den.mean():.3f}; best/carry {1/r[0]:.3f}; best/thr {seg('hz', c, b, FB, 0, H).mean()/seg('hz', c, 'thr_agreement', FB, 0, H).mean():.3f}]")
            cin[c] = (b, bB)
    print("  ", c, " | ".join(row))
print("   totals, mean over 16 seeds (carry, thr, uniform), thr/carry, freeze_1000/carry, freeze_500/carry:")
for c in CELLS:
    t = {k: A("hz", c, k).sum(1).mean() for k in ("carry_plain", "thr_agreement", "uniform", "freeze_plain_1000", "freeze_plain_500", "freeze_plain_250")}
    print("  ", c, {k: round(v, 1) for k, v in t.items()}, f"thr/carry {t['thr_agreement']/t['carry_plain']:.3f} carry/thr {t['carry_plain']/t['thr_agreement']:.3f} fz1000/carry {t['freeze_plain_1000']/t['carry_plain']:.3f} fz500 {t['freeze_plain_500']/t['carry_plain']:.3f} fz250 {t['freeze_plain_250']/t['carry_plain']:.3f}")
print("   H=120 extension (fold A, 120-batch stream):")
for c in CELLS:
    b, _ = best(c, "hz120", FA, 0, 120); r = ci(seg("hz120", c, "carry_plain", FB, 0, 120), seg("hz120", c, b, FB, 0, 120))
    b12, _ = best(c, "hz120", FA, 108, 120); r12 = ci(seg("hz120", c, "carry_plain", FB, 108, 120), seg("hz120", c, b12, FB, 108, 120))
    # pooled
    bB, _ = best(c, "hz120", FB, 0, 120)
    num = np.r_[seg("hz120", c, "carry_plain", FB, 0, 120), seg("hz120", c, "carry_plain", FA, 0, 120)]; den = np.r_[seg("hz120", c, b, FB, 0, 120), seg("hz120", c, bB, FA, 0, 120)]
    print("  ", c, f"H120 {b} {r[0]:.3f} ({r[1]:.3f},{r[2]:.3f}) pooled best/carry {den.mean()/num.mean():.3f}; last12-of-120 {b12} {r12[0]:.3f}")
print("   steady state: last 12 of 60, arm chosen on last 12; arm chosen on H=60; blocks")
for c in CELLS:
    b12, _ = best(c, "hz", FA, 48, 60); r = ci(seg("hz", c, "carry_plain", FB, 48, 60), seg("hz", c, b12, FB, 48, 60))
    b60 = cin[c][0]; r60 = ci(seg("hz", c, "carry_plain", FB, 48, 60), seg("hz", c, b60, FB, 48, 60))
    blk = [seg("hz", c, "carry_plain", FB, lo, lo + 12).mean() / seg("hz", c, b60, FB, lo, lo + 12).mean() for lo in range(0, 60, 12)]
    print("  ", c, f"last12 own-arm {b12} {r[0]:.3f} ({r[1]:.3f},{r[2]:.3f}); H60-arm {r60[0]:.3f}; blocks", " ".join(f"{x:.3f}" for x in blk))
    # post-calibration best / carry at H=60
    kk = -(-int(b60.rpartition('_')[2]) // c[1]); pr = ci(seg("hz", c, b60, FB, kk, 60), seg("hz", c, "carry_plain", FB, kk, 60)); print("      post-cal best/carry", f"{pr[0]:.3f} ({pr[1]:.3f},{pr[2]:.3f}) k={kk}")
print("   verdict-fold arm below carry_plain on report seeds (H=60 totals):")
for c in CELLS:
    b = cin[c][0]; print("  ", c, b, int((seg("hz", c, b, FB, 0, 60) < seg("hz", c, "carry_plain", FB, 0, 60)).sum()), "of 8; swapped fold arm", cin[c][1], int((seg("hz", c, cin[c][1], FA, 0, 60) < seg("hz", c, "carry_plain", FA, 0, 60)).sum()), "of 8")
print("   per-seed oracle best-of-12 fixed / carry_plain (H=60) over 16 seeds: mean, #>=0.99, max; also best-of-12 by mean over all 16 seeds (hindsight) / carry")
for c in CELLS:
    tot = np.array([A("hz", c, k).sum(1) for k in fx]); car = A("hz", c, "carry_plain").sum(1); o = tot.max(0) / car
    ma = tot.mean(1).argmax(); print("  ", c, f"{o.mean():.3f} {int((o >= 0.99).sum())} {o.max():.3f}; hindsight-all-16 arm {fx[ma]} {tot.mean(1)[ma]/car.mean():.3f}")
# ---- F. memorisation
print("F. memorisation (hz): primary gap mean, n seeds >0.02, AUC lab / never, split gap mean, n>0.02, repeat share, hit share on repeats, distinct labelled, never labelled, n_labels")
for c in CELLS:
    m = [d["memorisation"] for d in D["hz"][c]]; s = [d["stream"] for d in D["hz"][c]]
    g = np.array([x["gap_labelled_minus_never"] for x in m]); gs = np.array([x["gap_split_in_minus_held_out"] for x in m])
    print("  ", c, f"{g.mean():.3f} {int((g>0.02).sum())} | {np.mean([x['auc_labelled'] for x in m]):.3f} {np.mean([x['auc_never_labelled'] for x in m]):.3f} | split {gs.mean():.3f} {int((gs>0.02).sum())} | rep {np.mean([x['carry_plain_repeat_pick_share'] for x in s]):.3f} hits-on-rep {np.mean([x['carry_plain_hits_on_repeat_picks']/x['carry_plain_hits_total'] for x in s]):.3f} | lab {np.mean([x['n_distinct_labelled'] for x in m]):.0f} never {np.mean([x['n_never_labelled'] for x in m]):.0f} nl {np.mean([x['n_labels'] for x in m]):.0f}")
print("   no-repeat memorisation / repeat share:")
for c in CELLS:
    m = [d["memorisation"] for d in D["nr"][c]]; s = [d["stream"] for d in D["nr"][c]]; gs = np.array([x["gap_split_in_minus_held_out"] if x["gap_split_in_minus_held_out"] is not None else np.nan for x in m], float); g = np.array([x["gap_labelled_minus_never"] if x["gap_labelled_minus_never"] is not None else np.nan for x in m], float)
    print("   n None primary", int(np.isnan(g).sum()), "n None split", int(np.isnan(gs).sum()))
    print("  ", c, f"nb {min(len(d['batch_base_errors']) for d in D['nr'][c])}-{max(len(d['batch_base_errors']) for d in D['nr'][c])} split gap {np.nanmean(gs):.3f} primary {np.nanmean(g):.3f} rep {np.mean([x['carry_plain_repeat_pick_share'] for x in s]):.3f} (min {min(x['carry_plain_repeat_pick_share'] for x in s):.3f} max {max(x['carry_plain_repeat_pick_share'] for x in s):.3f}) never-labelled mean {np.mean([x['n_never_labelled'] for x in m]):.0f}")
# ---- G. no-repeat
print("G. no-repeat probe: first/last 12 mean errors per batch; carry/best fold A and pooled no-repeat minus iid at H list")
def fold_nr(docs_stage, c, H):
    # all seeds reach H? use first H batches
    a = lambda k: np.array([d["arms"][k][:H] for d in D[docs_stage][c]], float)
    sc = lambda sl: max(fx, key=lambda k: a(k)[sl].sum(1).mean())
    bA, bB = sc(FA), sc(FB)
    rA = a("carry_plain")[FB].sum(1).mean() / a(bA)[FB].sum(1).mean()
    num = np.r_[a("carry_plain")[FB].sum(1), a("carry_plain")[FA].sum(1)]; den = np.r_[a(bA)[FB].sum(1), a(bB)[FA].sum(1)]
    return bA, rA, num.mean() / den.mean()
for c in CELLS:
    be = [d["batch_base_errors"] for d in D["nr"][c]]; Hn = min(len(b) for b in be)
    f12 = np.mean([np.mean(b[:12]) for b in be]); l12 = np.mean([np.mean(b[-12:]) for b in be]); l12m = np.mean([np.mean(b[Hn - 12:Hn]) for b in be])
    out = []
    for H in sorted({12, 24, 36, Hn} & set(range(1, Hn + 1))):
        bn, rAn, rPn = fold_nr("nr", c, H); bi, rAi, rPi = fold_nr("hz", c, H); out.append(f"H{H} nr A {rAn:.3f} ({bn}) pooled {rPn:.3f} | iid A {rAi:.3f} pooled {rPi:.3f} | diff pooled {rPn - rPi:+.3f}")
    print("  ", c, f"Hn(min) {Hn} (per-seed {sorted(set(len(b) for b in be))}) first12 {f12:.1f} last12 (own end) {l12:.1f} last12 (batches Hn-12..Hn) {l12m:.1f}"); [print("       ", o) for o in out]
# ---- H. extrapolation
print("H. extrapolation q=best/carry (a + c/H) through H=12,24,36,60, fold A and pooled")
for c in CELLS:
    qa, qp = [], []
    for H in (12, 24, 36, 60):
        b, _ = best(c, "hz", FA, 0, H); bB, _ = best(c, "hz", FB, 0, H)
        qa.append(seg("hz", c, b, FB, 0, H).mean() / seg("hz", c, "carry_plain", FB, 0, H).mean())
        qp.append(np.r_[seg("hz", c, b, FB, 0, H), seg("hz", c, bB, FA, 0, H)].mean() / np.r_[seg("hz", c, "carry_plain", FB, 0, H), seg("hz", c, "carry_plain", FA, 0, H)].mean())
    out = []
    for nm, q in (("A", qa), ("P", qp)):
        X = np.c_[np.ones(4), 1 / np.array([12., 24, 36, 60])]; a, cc = np.linalg.lstsq(X, np.array(q), rcond=None)[0]
        h103 = cc / (1 / 1.03 - a) if (1 / 1.03 < a and cc < 0) else None; h99 = cc / (0.99 - a) if (0.99 < a and cc < 0) else None
        out.append(f"{nm}: a={a:.3f} (carry/best asympt {1/a:.3f}) fit120 {a + cc/120:.3f} H(carry/best=1.03)={h103 and round(h103)} H(best/carry=.99)={h99 and round(h99)}")
    print("  ", c, " | ".join(out))
print("   measured pooled best/carry at 120 (hz120):")
for c in CELLS:
    b, _ = best(c, "hz120", FA, 0, 120); bB, _ = best(c, "hz120", FB, 0, 120)
    print("  ", c, f"{np.r_[seg('hz120', c, b, FB, 0, 120), seg('hz120', c, bB, FA, 0, 120)].mean() / np.r_[seg('hz120', c, 'carry_plain', FB, 0, 120), seg('hz120', c, 'carry_plain', FA, 0, 120)].mean():.3f}")
