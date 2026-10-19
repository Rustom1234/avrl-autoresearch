"""Append the L2b entry to avrl/RESEARCH_LOG.md. Tables are generated from runs/L2b/sim_*.json (independent of l2b_summarize.py)."""
import json, re, subprocess, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "runs", "L2b")
LOG = os.path.join(HERE, "..", "RESEARCH_LOG.md")
DATE = subprocess.check_output(["date", "+%F"], text=True).strip()
J = {"3B": "Qwen2.5-3B-Instruct", "7B-AWQ": "Qwen2.5-7B-Instruct-AWQ"}
rng = np.random.default_rng(777)


def is_fixed(n): return n == "thr_agreement" or re.match(r"(recal|seeded|split)_charged_\d+$", n) is not None
def is_learned(n): return n == "greedy_online" or n.startswith("greedy_n") or n.startswith("greedy_seeded_")


def rci(l, f):
    idx = rng.integers(0, len(l), size=(10000, len(l)))
    r = l[idx].mean(1) / f[idx].mean(1)
    return l.mean() / f.mean(), *np.percentile(r, [2.5, 97.5])


rows = []
for jl, jn in J.items():
    for m in (250, 500, 1000):
        D = [json.load(open(f"{RUN}/sim_{jn}_m{m}_s{s}.json")) for s in range(16)]
        for b in ("0.1", "0.25", "0.5"):
            names = list(D[0][f"budget_{b}"])
            per = {n: np.array([np.mean(d[f"budget_{b}"][n]["per_seed"]) for d in D]) for n in names}
            fx = [n for n in names if is_fixed(n)]; ln = [n for n in names if is_learned(n)]
            sel = {}
            for fold, (s_, r_) in {"A": (slice(0, 8), slice(8, 16)), "B": (slice(8, 16), slice(0, 8))}.items():
                f = max(fx, key=lambda n: per[n][s_].mean()); l = max(ln, key=lambda n: per[n][s_].mean())
                sel[fold] = (f, l, per[l][r_], per[f][r_])
            f, l, lv, fv = sel["A"]
            r, lo, hi = rci(lv, fv)
            rB = sel["B"][2].mean() / sel["B"][3].mean()
            rp = np.concatenate([sel["A"][2], sel["B"][2]]).mean() / np.concatenate([sel["A"][3], sel["B"][3]]).mean()
            h = lambda n: per[n][8:16].mean()
            B = int(round(float(b) * m))
            rows.append(f"| {jl} | {m} | {int(float(b)*100)}% (B={B}) | {h('uniform'):.1f} | {h('thr_agreement'):.1f} | {f} {fv.mean():.1f} | {l} {lv.mean():.1f} | {h('greedy_online'):.1f} | {h('static'):.1f} | {h('oracle'):.1f} | {r:.3f} ({lo:.3f} to {hi:.3f}) | {rB:.3f} | {rp:.3f} |")
TABLE = ("| label | m | budget | uniform | thr_agreement | selected fixed (chosen on 0-7) | selected learned (chosen on 0-7) | greedy_online (L1b setting) | static_out (free-label ceiling) | oracle | learned / fixed (95% CI, 8 held-out subsets) | swapped fold | pooled 16 |\n"
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))

ENTRY = f"""
## L2b {DATE} inconclusive
Hypothesis: the L2a pass (repeated-refit scorer over the best charged fixed rule, 1.050 / 1.060 / 1.139 / 1.061 at 3B 25%, 3B 50%, 7B-AWQ 25%, 7B-AWQ 50%) depends on a pool of about 2,200 groups in which 25 random start labels are a small share of the budget. At batch scale (m = 250 to 1,000 prompts, budget 25 to 500 verifications) the calibration cost is a much larger share and the edge shrinks to 1.00 or below, so a per-step allocator cannot reach it. Kill reading: best learned variant / best charged fixed member at most 1.00 at m = 500 and m = 250 in all four (judge, budget) cells. Falsifier: ratio at least 1.03 with the subset-cluster CI above 1 at m = 500 in all four cells.

Run (CPU only, no GPU, no generation; about 2 min of compute, cmd.sh 20:27:26 to done.flag 20:29:21, exit_code 0): E04 feature tables (3,000 GSM8K prompts x 8 Qwen2.5-1.5B-Instruct rollouts, ten cheap + judge features), label judge_error from the Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct-AWQ answer-only judges (E01 has no headroom). exp/l2b_sim_alloc.py is exp/l2a_sim_alloc.py with run() and run_online() unchanged (AST diff: identical) and a new population step: m prompts drawn without replacement with default_rng(subset_seed), units = prompts, K = 1. Design: m in 250, 500, 1000 x subset seeds 0 to 15 x 8 policy seeds x budgets 10 / 25 / 50% of m (10% reported, not in the verdict), 96 jobs. Policies fixed in advance: thr_agreement and recal / seeded / split charged_n for n in 10, 15, 25, 50, 75, 100, 150, 200, 250 with n at most budget // 2 (fixed family); greedy_n<n0>_R<R> for (n0, R) in (10,5), (10,10), (15,15), (25,10), (25,25), greedy_online (L1b setting), greedy_seeded_n25_R25 (learned family); uniform, oracle and static_out (logistic fit on the 3,000 - m prompts outside the subset with free labels, ceiling only). ts_online dropped (below greedy_online in every cell of L1b and L2a). Selection: best fixed member and best learned variant chosen on subsets 0 to 7 (mean over subsets and seeds), reported on subsets 8 to 15 (fold A, the verdict fold); fold B swaps the roles; pooled = the 16 held-out values. CI: percentile bootstrap over the 8 held-out subsets, 10,000 resamples. Files: runs/L2b/result.json, summary.log, gate/gate_check.json, 96 sim_<judge>_m<m>_s<subset>.json. Caveat: subsets are draws from the same 3,000 prompts (pairwise overlap about 8%, 17%, 33% at m = 250, 500, 1000), so this is a scale and subset-variance test on one population, not a fresh population.

Audit (I re-derived the numbers from the 96 raw sim files with an independent script, exp/l2b_verify.py, own bootstrap seed, instead of trusting the run report). Regression gate: gate/gate_check.json shows 116 checks, 0 differences (per-seed hits of uniform, static, oracle, thr_agreement, greedy_online, greedy_n25_R25, greedy_seeded_n25_R25 and recal / seeded / split charged_n for n in 25, 50, 100, 200, m = 0, split 0, both judges, against runs/L2a); the gate script is read-and-compare on per_seed lists and the launcher voids the sweep on failure, so the gate is real. All 18 (judge, m, budget) cells reproduce: selected members, ratios and CIs (within 0.003 of summary.log), swapped-fold and pooled ratios, and the verdict counts. Slips in the run report, none changing the verdict: the 190.2 it quotes for 7B-AWQ m = 500 50% static_out is the 3B value (7B is 189.9), and "15 wall minutes" is script-writing time.

Errors caught on held-out subsets 8 to 15 (mean over 8 subsets x 8 policy seeds; B = absolute verification budget):

{TABLE}

Ratio of selected learned to selected fixed against absolute budget B (fold A; cells with equal B but different m in the same row; L2a full-pool values for reference). Post hoc, from the table above:

| B | 3B | 7B-AWQ | cells |
|---|---|---|---|
| 25 | 0.705 | 0.782 | m=250 10% |
| 50 | 0.873 | 0.870 | m=500 10% |
| 62 | 0.880 | 0.905 | m=250 25% |
| 100 | 0.940 | 0.938 | m=1000 10% |
| 125 | 0.989, 0.978 | 1.018, 0.988 | m=250 50%, m=500 25% |
| 250 | 1.039, 1.025 | 1.091, 1.075 | m=500 50%, m=1000 25% |
| 500 | 1.076 | 1.073 | m=1000 50% |
| about 550 and 1,100 (L2a) | 1.050, 1.060 | 1.139, 1.061 | full pool, 25% and 50% |

Findings.
1. The edge is a function of the absolute number of verifications B, not of m or the budget fraction. The two pairs of cells with equal B and different m agree to within 0.03 (B = 125: 0.989 and 0.978 on 3B, 1.018 and 0.988 on 7B-AWQ; B = 250: 1.039 and 1.025, 1.091 and 1.075). The curve crosses 1.00 near B = 125 to 150 and reaches 1.03 near B = 250. At B of 62 or less the learner is 9% to 30% below the agreement threshold, and at B = 100 about 6% below. This is post hoc: 18 cells from one population, m and fraction confounded outside the two matched pairs. It is a hypothesis for L4a and L4b, not a result.
2. The learned optimum is the grid corner, again. n10_R5 (smallest n0 and R) is the best learned variant on the all-subset mean in 16 of 18 cells (exceptions: 3B m = 250 25%, greedy_seeded_n25_R25 by 0.2 hits; 7B-AWQ m = 1000 50%, n10_R10 by 0.8 hits) and the ordering n10_R5 > n10_R10 > n15_R15 > n25_R25 is nearly monotone. At m = 500 25% the step from (25,25) to (10,5) gained 4.7 hits on both judges (3B 90.4 to 95.1, 7B-AWQ 87.9 to 92.6) while the remaining gap to the pooled fixed rule is 2.2 and 2.1 hits (97.3 and 94.7), so the two 25% kill cells at m = 500 (0.978, 0.988) are not robust to extending the grid below n0 = 10, R = 5. The kill is therefore weaker than 5 of 8 suggests, and the pass failures are not a clean refutation either.
3. greedy_online at its L1b setting (R = max(budget // 8, 40)) no longer passes at batch scale: ratio to selected fixed at m = 500 is 0.898 (3B 25%), 1.007 (3B 50%), 0.904 (7B-AWQ 25%), 1.050 (7B-AWQ 50%). The edge is carried by the small-n0, frequent-refit variants.
4. Fixed side. thr_agreement is the selected fixed rule on subsets 0 to 7 in 15 of 18 cells; a charged rule is selected only at 50% budget (7B-AWQ m = 500 split_charged_100, 7B-AWQ m = 1000 split_charged_150, 3B m = 1000 seeded_charged_250). The cap n at most budget // 2 truncates the grid: at m = 500 50% the largest allowed n is 100, which is the 7B-AWQ pick, and that curve is still creeping up (split_charged_50, 75, 100: 159.9, 160.4, 162.7 on the all-subset mean against thr_agreement 162.0). The 1.091 there is partly a truncated-family number; a fixed rule worth 3 more hits would still leave about 1.07.
5. static_out (free labels) is above both arms in every cell: 1% to 7% headroom over the learner at B >= 250 (3B m = 500 50%: 190.2 against 179.6; 3B m = 1000 50%: 374.9 against 369.6), so the repeated refit does not reach the label-free ceiling.
6. Inference limits. Subsets overlap, so the 8 held-out subsets are not independent (most at m = 1000, 33% overlap) and the CIs are optimistic there; policy seeds are the same across subsets; every CI here speaks to subset variance of one 3,000-prompt population.

Verdict against the pre-registered thresholds: INCONCLUSIVE. The regression gate passed (a failed gate would have voided the run). Pass needs ratio at least 1.03 with the subset-cluster CI lower bound above 1 at m = 500 in all four cells; 2 of 4 hold (3B 50%: 1.039, CI 1.019 to 1.062; 7B-AWQ 50%: 1.091, CI 1.078 to 1.106) and both 25% cells fail (3B: 0.978, CI 0.956 to 1.002; 7B-AWQ: 0.988, CI 0.968 to 1.010). Kill needs ratio at most 1.00 at m = 250 and m = 500 in all four cells (8 cells); 5 of 8 hold (the three exceptions are 3B m = 500 50% 1.039, 7B-AWQ m = 250 50% 1.018 with CI 0.993 to 1.043, and 7B-AWQ m = 500 50% 1.091). The swapped fold (0.966, 1.041, 0.977, 1.113 at m = 500), the pooled 16 subsets (0.978, 1.040, 0.977, 1.102) and the held-out-max sensitivity give the same pass and kill patterns, so the inconclusive outcome does not depend on the fold. Neither the m = 500 falsifier nor the kill reading is established.

What it changes. Claim 2 (a learned allocator admits fewer corrupted rewards than any fixed rule at equal budget) keeps its L2a support but is now conditional: in the offline replay it holds when the allocator has about 250 or more verified labels in hand (all 6 L2b cells at B >= 250 and all 4 L2a cells have ratio above 1.00, and 9 of those 10 are at or above 1.03; the exception is 3B m = 1000 25% at 1.025, with swapped and pooled 1.044 and 1.034) and fails below about 125, where thr_agreement beats a per-step restart learner (ratio 0.70 to 0.99 in 11 of the 12 cells at B <= 125; the exception is 7B-AWQ m = 250 50% at 1.018 with a CI that includes 1). The thesis claim as written ("any fixed rule at the same budget") is therefore not shown for a per-step allocator at small batches, and the Phase 1 design has to persist labels across steps (or pool verification budget over steps) for claim 2 to be testable there; a fresh allocator per step at B under 100 is expected to lose to the threshold arm in the Phase 1 kill test. The Phase 1 refit cadence read off the grid is "as frequent as the grid allows" (every 5 labels from a 10-label start was best in 16 of 18 cells), unbracketed. Claim 3 is untouched: L2b used no embedding or prompt-space features, and the edge it measures is a repeated-refit effect on cheap + judge features (E01 and E04 kills stand; ts_online was dropped because it lost in L1b and L2a). Claim 1 is untouched (needs the Phase 1 harness). The fixed agreement threshold stays the default deliverable at small B.

Follow-ups, ranked by expected information value (full specs in lit/avrl/EXPERIMENT_BACKLOG.md under 'Proposed by the loop', Iteration 4).
1. L4a (CPU, about 15 to 20 min): bracket the learner's small end (n0 in 3, 5, 10; R in 1, 2, 5) and add a threshold-offset warm-start learner (score = agreement offset plus a ridge-penalized correction on the other nine features, so it equals thr_agreement at zero labels). Highest value: it decides whether the small-B loss is a fixable cold-start cost or a hard floor near B = 125 to 150 for this feature set, and it removes the grid-corner objection that has now been raised twice.
2. L4b (CPU, about 10 min): cross-step carry-over replay. 12 disjoint batches of 250 prompts, allocator keeps its labels across batches (and a sliding-window variant, W = 125, 250, 500, as a drift proxy), against thr_agreement and a calibrate-once fixed rule. Tests whether label history, not batch size, is the driver of finding 1. Medium to high value: the unbounded-carry arm is expected to recover the B >= 250 ratios (stationary by construction), so the information is the window curve (how stale the labels may be) and the batches-to-break-even count; it feeds E13.
3. L4c (GPU, two jobs, about 35 to 45 min; needs Rustom's explicit override of the E04 stop rule): L2c redesigned for B >= 250 (3 disjoint blocks of 1,000 fresh GSM8K prompts, n10_R5 and n10_R10 pre-specified). Medium value, the only item that addresses the one-population caveat; ask after L4a so the arms tested are final.
"""

with open(LOG, "a") as fh:
    fh.write(ENTRY)
print(TABLE)
