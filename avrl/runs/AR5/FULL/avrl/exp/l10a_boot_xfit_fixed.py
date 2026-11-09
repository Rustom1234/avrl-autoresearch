"""L10a-xfit-fixed-comparator, replay half. Copy of exp/l8c_boot_forced.py (L8c-A / L9c, left untouched) with ONE added mode, XFIT=2. The existing XFIT=1 block is unchanged and is NOT used here (it cross-fits only the tie-break feature and still picks its best fixed rule in-sample on the held half).
PRE-REGISTERED here, before any XFIT=2 job was run. What was read before this text: the stored full-pool and bootstrap numbers of earlier experiments (L1a, L2a, L3b, L7a, L8c-A, L9c), all of which compare greedy_blend7 with the IN-SAMPLE best of the seven fixed rules (plug-in greedy_blend7 / best fixed about 1.03 to 1.04 on the 0.5B pools at 25%, 0.96 to 1.09 CI). The cross-fitted statistic defined below has never been computed on any pool, and no script of this experiment had been run on a real label column when this docstring was written.
HYPOTHESIS (claim 2). A learned allocator admits fewer corrupted rewards than any fixed rule at equal audit budget, once the best fixed rule is chosen OUT OF SAMPLE. With the comparator chosen on the other half of the prompts, greedy_blend7 catches more judge-error groups than the best fixed rule, averaged over six budgets and pooled over four (policy, proxy) cells. NULL: the learner's edge over agree_acc (1.04 to 1.14) is a feature blend that a cross-fitted fixed rule already has, so the pooled edge is at most 1%.
CELLS (4, non-independent: one prompt set, two policies, two proxies, all Qwen-family, GSM8K train, 2,000 groups each, pid == arange, MIN_CELL=1 keeps all groups): L9c (Qwen2.5-0.5B policy) and L8c-A (Qwen2.5-1.5B policy), each with the Qwen3-4B Yes/No judge (q3_4b) and the Skywork-Reward-V2-Qwen3-0.6B reward model (rm). Label = judge_error. BUDGETS (6): 10, 15, 20, 25, 30, 40 percent of the held half.
MODE XFIT=2 (own JSON). pids are split by parity into half 0 (even) and half 1 (odd), 1,000 groups each. For each held-out half h, with FRAC the run budget fraction:
  (a) SELECTION on the OTHER half only (its labels only): all SEVEN fixed rules FIXED7 = (agree_thr, judge_thr, blend50, agree_acc, blend50_acc, agree_acc_soth, agree_acc_sdiff) are run on the other half with SEEDS = 8 seeds at budget int(FRAC * len(other)); the cross-fitted best fixed rule CH = argmax of the 8-seed mean caught, the first listed wins ties. Nothing computed on half h enters CH.
  (b) SCORING on half h at budget int(FRAC * len(held)): CH, all seven fixed rules (context), agree_acc, uniform, oracle, and the learner greedy_blend7 = run_greedy_warm(blend=True, best=FORCE_BEST7), warm-started from the FROZEN arm (rm: agree_acc_soth, Qwen3-4B: agree_acc_sdiff, exactly as L8c-A / L9c), NO reselection of the warm start. The in-sample best of the seven on the held half (argmax of the held-half 8-seed mean) is recorded as context only; it is never an arm of the verdict.
  (c) BOOTSTRAP of the held half: B resamples (200) x BOOT_SEEDS (4) seeds, a FRESH default_rng(12345) per half, each resample draws held.iloc[integers(0, len(held), len(held))]. The generator draws nothing else, so the resample indices are identical at every budget and, since all four pools have 2,000 rows and pid == arange, identical across proxies and policies (sha256 of the index stream is stored per half and checked in the finalizer), which makes the resample CI paired across cells. No new random draw is added anywhere: orders(), run() and run_greedy_warm() are byte-for-byte those of l8c_boot_forced.py, so every existing column is bit-identical. The per-resample quantity of each arm is the SUM over the two held-out halves of the seed-mean caught (resample i of half 0 paired with resample i of half 1; the halves are independent data).
  GATES: greedy_warm7_norefit (refit never fires) must equal the frozen arm per seed on the full held half (8 seeds, exit 2) and on the first 20 resamples (exit 3). REPRODUCTION GATE (main mode, XFIT unset, MIN_CELL=1, BOOT_FRAC=0.25, FORCE_BEST7): the new script must reproduce the stored runs/{L9c-weak-policy-tiebreak,L8c-A-fresh-tiebreak}/boot/{q3_4b,rm}/boot_tb_f0.25.json full pool exactly, agree_acc / blend50_acc / greedy_blend7 means L9c q3_4b 192.25 / 214.25 / 220.0, L9c rm 201.875 / 222.75 / 231.0, L8c-A q3_4b 136.5 / 123.0 / 151.0, L8c-A rm 142.0 / 130.5 / 148.0 (and every other full-pool column and all resamples of the run). Any mismatch: the copy changed old behaviour, stop, FAILED.
STATISTICS (exp/l10a_finalize.py). Per cell c, budget f, resample i: G = greedy_blend7 and K = CH, both summed over the two halves (seed-mean per half); ratio = G / K (per cell and budget, percentile CI over the 200 resamples, descriptive). C_cell[i] = mean over the six budgets of log(G_f[i] / K_f[i]). Pooled C[i] = unweighted mean over the four cells of C_cell[i]; the 95% PERCENTILE CI of pooled C is taken over the 200 paired resamples (the basic CI is never used). POINT = plug-in, built the same way from the unresampled 8-seed full-half means (sum over halves), not the bootstrap mean. SD of pooled C = SD over the 200 resamples. Context (never in the verdict): C against agree_acc, against the frozen arm, against the IN-SAMPLE best of seven (held-half choice, the old convention), the number of cells with positive plug-in C, the table of CH by (cell, budget, half), and the winner's-curse gap (in-sample best minus CH, in caught errors and as a share of the in-sample best, plug-in). The L8c-A Qwen3-4B cell (the one cell where the learner edge over the frozen arm replicated in sample, 1.079) is reported alone as a flag.
REGISTERED BINS (verdict, evaluated only if every gate passed and all 24 jobs are present):
  PASS: pooled C point >= +0.02 AND pooled C percentile CI lower end > 0 AND a positive plug-in C_cell in at least 3 of the 4 cells.
  KILL: pooled C percentile CI upper end < +0.01 (the learner's edge over a cross-fitted fixed rule is below 1%): claim 2 as worded is dropped and the deliverable becomes the agreement threshold plus the fixed blend and the dissenting-score tie-break.
  Both PASS and KILL cannot hold (point >= 0.02 with upper < 0.01 is impossible for a percentile CI that contains its own point estimate; if the point lies outside the CI the verdict is INCONCLUSIVE, DISCORDANT).
  Anything else: INCONCLUSIVE. PRE-STATED POWER FLAG: if the pooled C CI half-width ((upper - lower) / 2) exceeds 0.03 and neither bin fires, the reading is labelled INCONCLUSIVE, POWER-LIMITED.
  FAILED (not a verdict): any gate failed or missing, a Traceback or GATE FAILED in any log, a missing job, the resample-index check failing, or the reproduction gate mismatching.
  A PASS would give claim 2 its first cross-fitted evidence and justify carrying the learner (not only the fixed blend) into the Phase 1 GRPO design; it would still be one task (GSM8K), Qwen-family policies and proxies, and four non-independent cells.
CAVEATS carried to the log: the CI conditions on the two choices CH (the selection half is not resampled), as in XFIT=1; only 7 candidate fixed rules (the candidate set is itself post hoc, five from L5c and two tie-breaks found post hoc on these same prompts in L6a / L7a); the frozen warm start (agree_acc_soth / agree_acc_sdiff) was chosen from the same 2,000-prompt pool in L7a, so the learner side keeps a small in-sample advantage that the cross-fit does not remove; the two halves of a pool and the four cells share prompts and errors; the learner refits on audited rows of ONE half (1,000 groups), half the data of the earlier full-pool reads; the budget fractions are applied to half-pools.
Usage: XFIT=2 FORCE_BEST7=<rule> MIN_CELL=1 BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=<f> OMP_NUM_THREADS=1 l10a_boot_xfit_fixed.py <feats_J.parquet> judge_error cell100 <feature_cols,...> <out.json>
"""
"""L8c-A-fresh-tiebreak, replay half. Copy of sim_warm_boot_j4.py (L7a) with ONE change: env FORCE_BEST7=<rule> sets BEST7 to that fixed rule instead of selecting it on the pool (no reselection, no cross-fit).
The natural in-sample argmax is still computed and stored as full_pool["best_fixed7_natural"]. Everything else (every arm, every draw, the gates, default_rng(12345), the ALL / TB arm lists) is unchanged, so with FORCE_BEST7 equal to the natural BEST7
of a pool the output equals the j4 output (gate (i) of L8c checks that on the pids 0..1999 pools against runs/L7a-tiebreak/<pool>/boot_tb_f0.25.json). Frozen rules for the fresh pool: rm -> agree_acc_soth, Qwen3-4B -> agree_acc_sdiff.
Second change (found by the pool-size check of l8c_analyze.py on the first fresh run, which replayed 1998 groups): env MIN_CELL (default 3, as j4) sets the minimum cell size of the group filter; the fresh runs use MIN_CELL=1 so all 2,000 prompts and 500 audits are used (that first 1998-group run is kept in boot_cellfilter1998_first_run/). The j4 docstring follows unchanged.
"""
"""L7a-tiebreak, replay half. Copy of sim_warm_boot_j3.py (L6a) with two label-free fixed tie-break policies added, a seven-rule best-fixed mode, and a cross-fit verdict mode (env XFIT=1). PRE-REGISTERED here, before any real label column was read for this experiment:
no L7a script loaded judge_error, judge_error_sens, majority_correct, proxy_error, gold or pass_rate before this text was written (only pid, agreement, n_distinct, s_maj, s_oth, s_diff, accept and cell100 were inspected, for NaN, ties and tier sizes).
PROVENANCE, stated plainly: the idea (the agree_acc gap is a tie-break inside one agreement tier), the two tie-break features and their signs (s_oth descending, s_diff ascending) come from the L6a reviewer's post hoc preview on these same 2,000 prompts (RESEARCH_LOG.md, L6a entry, diagnostics a to c), which DID read labels. The feature scan is post hoc on this pool;
the cross-fit below removes only the choice between the two features and the half it is scored on, not the signs or the idea.
HYPOTHESIS. The learner's 25% edge over agree_acc (greedy_blend / agree_acc plug-in 1.046 / 1.009 / 1.111 on rm K=1,715 / judge3b / Qwen3-4B) is only the choice of 30 to 60 audits inside one agreement tier, and a label-free fixed tie-break on the proxy's score of the dissenting completion reaches 98% of greedy_blend.
NEW FIXED POLICIES (context arms in mode (i); candidates for best fixed in mode (ii)), same lexsort keys as agree_acc with the random draw u4 replaced by a score; rejected groups keep agree_acc's random order (u4); u6 is a NEW draw made after u1..u5 so every old arm is bit-identical, it only breaks exact score ties (bootstrap duplicates):
  agree_acc_soth : accepted groups by agreement ascending, n_distinct descending, then s_oth DESCENDING (higher score of the dissenting completion = audit earlier).   agree_acc_sdiff : same with s_diff ASCENDING (s_diff = s_maj - s_oth exactly in all three pools).
MODES (one file). Main mode (XFIT unset), on the full 2,000-group pool, BOOT_B resamples x BOOT_SEEDS seeds, default_rng(12345): (i) every j3 column (uniform, five fixed rules, cell_rerank, greedy_warm, greedy_blend, oracle) is computed exactly as in j3 with BEST = best of the FIVE j2 rules; the two tie-breaks are CONTEXT arms outside that selection;
HARD GATE: all j3 columns must reproduce runs/L6a-cell-rate-usability/<pool>/boot_cellrerank_f0.25.json (and an unmodified j3 run at 0.1) exactly, full pool and every resample, checked in exp/l7a_finalize.py. (ii) BEST7 = best of all SEVEN fixed rules by full-pool 8-seed mean caught at 25% (first listed wins ties, the five j2 rules are listed first); greedy_warm7 and greedy_blend7 are warm-started from BEST7 (aliased to greedy_warm / greedy_blend when BEST7 == BEST);
gates (exit 4 / 5): greedy_warm7_norefit equals BEST7 per seed on the full pool and on the first 20 resamples; all j2 / j3 gates (exit 2 / 3 / 6 / 7) unchanged.
CROSS-FIT MODE (XFIT=1, own JSON): pids split by parity into halves 0 (even) and 1 (odd), 1,000 groups each. For each held-out half h: the tie-break feature is chosen on the OTHER half using only that half's labels: the boundary tier = the accepted groups whose agreement equals the agreement of the group in the last audited slot of the agree_acc key order (agreement ascending, n_distinct descending) of that half at its budget int(FRAC * 1000);
score s_oth (higher worse) versus -s_diff (lower s_diff worse); within-tier AUC against the label; the larger AUC wins, tie goes to s_oth; if the tier has fewer than 3 errors or fewer than 3 non-errors the choice falls back to s_oth and the fallback is flagged. The chosen tie-break (tb_xfit) is run on half h at FRAC of that half, next to agree_acc, greedy_warm, greedy_blend (warm-started from the best of the five j2 rules chosen on half h's own unresampled 25% means, the j2 convention, which if anything favours the comparator),
both tie-breaks (context), uniform and oracle, all on the SAME half-pool, resamples and budget. Bootstrap per half: default_rng(12345) created fresh for each half, B resamples of that half's groups, BOOT_SEEDS seeds each. Gates: greedy_warm_norefit equals best fixed per seed on each full half-pool (exit 2) and on the first 20 resamples (exit 3).
The CI conditions on the two choices (the selection half is not resampled), reported as a limitation. Per-resample cross-fit quantity = sum over the two held-out halves of the seed-mean caught (resample i of half 0 paired with resample i of half 1; the halves are independent data).
REGISTERED BINS (pools: rm K=1,715, judge3b, Qwen3-4B; verdict at BOOT_FRAC 0.25, B = 200 x 4 seeds; applied by exp/l7a_finalize.py). Per pool p:
  X_p = cross-fit point ratio = (sum over the two held-out halves of the 8-seed full-half mean caught by tb_xfit) / (same for greedy_blend on the same halves) >= 0.98.   C_p = the 95% PERCENTILE CI over the 200 resamples of greedy_blend / tb_xfit includes 1.0 (lower <= 1.0 <= upper); the basic CI is reported beside it, never used.
  V_p = non-vacuity guard: (sum over halves of 8-seed agree_acc) / (sum over halves of 8-seed greedy_blend) < 0.98. P_p = X_p and C_p and V_p. P_p_literal = X_p and C_p without V_p (always reported).   K_p = the 95% percentile CI lower end of greedy_blend7 / BEST7 (mode ii, full pool, per-resample seed means, 25%) is above 1.0.
  PASS = P_p on at least 2 of 3 pools. KILL = K_p on at least 2 of 3 pools. If both hold the verdict is INCONCLUSIVE (DISCORDANT). Anything else is INCONCLUSIVE. FAILED (not a verdict) if any reproduction gate, norefit gate or other gate failed, was skipped or is missing on a real run, or an input file is missing.
  Why V_p was added before any real label was read: the spec asks that no pass fire on null labels, but X_p and C_p alone are met by agree_acc itself on any pool where the learner has no edge over agree_acc (a null world, or a pool whose gap is already inside the 2% tolerance), so the literal bin cannot fail for a do-nothing rule there. V_p keeps PASS to pools where the learner's edge exceeds the bin's own tolerance.
  Registered flags (reported, never change the verdict): TB_ABOVE_LEARNER (the CI upper end of greedy_blend / tb_xfit is below 1.0, i.e. the tie-break is significantly ABOVE the learner, which fails C_p literally but would falsify claim 2 even more strongly); V_p-only differences (P_p_literal != P_p).
  NEVER in the verdict: the 10% budget, the rm K=1,840 sensitivity (ACCEPT_COL=accept_sens, judge_error_sens), basic CIs, the full-pool tie-break numbers of mode (i), the mode (i) greedy arms. Oracle headroom H = (oracle - best fixed) / oracle is reported with the seven-rule best fixed. The best fixed rule is chosen in-sample on the full pool (favours the fixed rules, makes K_p conservative).
SMOKE BINS (registered with the bins, run on synthetic labels on the real pools' non-label features before any real run, exp/l7a_smoke_synth.py): S1 null (6 independent null worlds x 3 pools, error rate 0.10 among accepted, independent of everything): PASS and KILL never fire in any world, tb_xfit / agree_acc percentile CI lower end > 1.0 in at most 1 of 18 pool-runs (nominal 2.5%), mean plug-in tb_xfit / agree_acc within 1 +/- 0.03.
  S2 planted s_oth (accepted groups only, logit p = -3.2 + 9 min(1 - agreement, 0.25) + 2.0 z, z = s_oth standardized within its agreement tier among accepted groups, tiers with fewer than 5 groups get z = 0): on the rm-like and Qwen3-4B-like pools tb_xfit / agree_acc plug-in > 1.04 and percentile CI lower end > 1.0 (judge3b has only 4 boundary slots at 25%, no gain is expected there), and across the 3 pools x 2 halves the cross-fit picks s_oth in at least 5 of 6 fold choices.
  S3 planted s_diff (z = minus s_diff standardized within its agreement tier): same criteria, picking s_diff in at least 5 of 6 fold choices. The synthetic pools reuse each real pool's non-label columns (agreement, n_distinct, s_maj, s_oth, s_diff, accept), so the tier structure is real and only the labels are synthetic. A smoke failure is fixed in the code, not by changing these bins. ERRATUM (first smoke run, before any real run): S3 failed its pick criterion (3 of 6) because within the boundary tier corr(s_oth, -s_diff) is 0.88 / 0.95 / 0.94 on the three real pools (s_diff = s_maj - s_oth), so the planted -s_diff effect was nearly the same signal as s_oth; the synthetic generator now replaces s_diff by an independent column in planted_sdiff (exp/l7a_smoke_synth.py), the S3 bin is unchanged. It also means the real s_oth versus s_diff choice is weakly identifiable.
DEFINITIONS OF THE NUMBERS. Ratios use the mean over seeds per resample as in l5c_finalize.py / l6a_finalize.py; plug-in = ratio of 8-seed full-pool (or full-half) means; B = 200 puts about 0.01 of Monte Carlo noise on percentile CI ends.
CAVEATS carried to the log: the feature scan was post hoc on these same 2,000 prompts and the cross-fit only partly removes that; the three pools share about 70% of their errors (two correlated proxies, not three independent tests); the boundary tier holds only 32 to 34 errors per pool (ratio SD about 0.04, about 5 errors); all proxies and the policy are Qwen-family; one policy (1.5B), GSM8K train.
Usage: [XFIT=1] [ACCEPT_COL=accept] BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 sim_warm_boot_j4.py <feats_J.parquet> <label> <group_col> <feature_cols,...> <out.json>; the file must hold pid, the group column, agreement, n_distinct, s_maj, s_oth, s_diff, the accept column and the label column.
---- j3 text follows ----
L6a-cell-rate-usability, replay half. Copy of sim_warm_boot_j2.py (L5c) with ONE added arm, cell_rerank, and an ACCEPT_COL env var. PRE-REGISTERED here, before any real label was read for this experiment.
NEW arm cell_rerank (fixed order plus online): start from the agree_acc order and audit greedily. The proxy-confident half C = accepted groups with s_maj >= median of s_maj over the accepted groups of the (resampled) pool (label-free). The agree_acc order is read as a
sequence of slots; a slot that agree_acc fills with a group of C is filled instead with the unaudited group of C that has the highest count of audited errors in its own cell100 (ties by agree_acc order); a slot filled with a group outside C is filled with the next unaudited group outside C in
agree_acc order, exactly as agree_acc does. So the share of the budget spent inside C equals agree_acc's by construction and only the choice WITHIN C changes. Error count of a cell = number of DISTINCT audited pids with label 1 in that cell (audit-time labels only, no gold before audit),
excluding the candidate's own pid, so duplicate copies of one group in a bootstrap resample cannot leak a label to each other (on the unresampled pool pids are unique and this is the plain count). cell_rerank is NOT a candidate for the best fixed rule (BEST is selected over the five fixed rules exactly as in j2, so every j2 arm reproduces j2).
Usage: ACCEPT_COL=accept BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 sim_warm_boot_j3.py <feats_J.parquet> <label> <group_col> <feature_cols,...> <out.json>; the file must hold pid, the group column, agreement, n_distinct, s_maj, the accept column (ACCEPT_COL, default accept) and the label column.
REPLAY CRITERION R_p (per pool, verdict interval = percentile): ratio = mean over seeds of cell_rerank / mean over seeds of agree_acc per resample (as l5c_finalize.py), 95% percentile CI over the 200 resamples at BOOT_FRAC = 0.25 (500 audits); R_p = CI lower end > 1.0. The basic CI is reported beside it, never used. Context arms reported: greedy_warm, greedy_blend,
best fixed, oracle headroom H = (oracle - best fixed) / oracle. Pool-level and experiment-level bins (PASS / KILL / INCONCLUSIVE) are registered in the docstring of exp/l6a_cell_rate.py and formed in exp/l6a_finalize.py. B = 200 puts about 0.01 of Monte Carlo noise on the percentile CI ends.
Added gates (exit 6 / 7): cell_rerank with the audit-time labels forced to zero (no audited error ever seen) equals agree_acc per seed on the full pool (8 seeds, exit 6) and on the first 20 resamples (exit 7); all j2 gates unchanged (exit 2/3).
---- j2 text follows ----
L5c / L5a corrected-baseline pool bootstrap. Copy of sim_warm_boot_j.py (L3b) with two extra label-free fixed rules and the learner warm-started from the BEST fixed rule instead of always blend50.
Allocator label J (column given as argv[2], here `judge_error` = proxy accepts the majority completion AND gold wrong); pool = ALL groups of feats_J.parquet (env POOL=held restores the E21 held-out-cells quarter split as a robustness read).
Usage: BOOT_B=200 BOOT_SEEDS=4 BOOT_FRAC=0.25 sim_warm_boot_j2.py <feats_J.parquet> <label> <group_col> <feature_cols,...> <out.json>
Fixed policies (ordering = suspicion, audit first; every draw of randomness is fixed so the learner's warm start reproduces the best fixed rule exactly per seed; u1..u3 are drawn first so agree_thr, judge_thr and
blend50 are identical to sim_warm_boot_j.py for the same seed, the new draws u4, u5 come after):
  agree_thr: agreement ascending, n_distinct descending, random tie-break.   judge_thr: accepted groups by s_maj ascending, rejected groups last in random order.
  blend50: 0.5 * rank(agree_thr ordering) + 0.5 * rank(judge_thr ordering) (rank = (n - position) / n, higher = audit earlier), random tie-break.
  NEW agree_acc: accepted groups by agreement ascending, n_distinct descending (random tie-break), rejected groups last in random order.
  NEW blend50_acc: 0.5 * rank(agree_acc ordering) + 0.5 * rank(judge_thr ordering), random tie-break.
best fixed rule = highest full-pool mean hits at 25% budget over the FIVE fixed rules (8 seeds, first listed wins ties), selected once on the unresampled pool and used for every resample.
Learner: greedy_warm is warm-started from the best fixed rule, refits a balanced logistic on audited rows every R = max(budget // 8, 40) labels and ranks by it; greedy_blend averages the warm rank and the refit rank (as L2a).
Gates (exit 2/3): greedy_warm_norefit == best fixed per seed on the full pool (8 seeds) and on the first 20 resamples. Resample generator default_rng(12345)."""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
MIN, SEEDS = int(os.environ.get("MIN_CELL", 3)), 8; ACC = os.environ.get("ACCEPT_COL", "accept"); XFIT = os.environ.get("XFIT", "0") == "1"
XFIT2 = os.environ.get("XFIT", "0") == "2"  # L10a: cross-fitted best fixed rule mode
path, LABEL, G, cheap = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","); out_path = sys.argv[5] if len(sys.argv) > 5 else None
df = pd.read_parquet(path); cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
if os.environ.get("POOL", "all") == "held":  # same held-out-cells split as sim_e21.py / sim_warm_boot.py
    rng0 = np.random.default_rng(0); groups = df[G].unique(); rng0.shuffle(groups); fit_g = set(groups[: len(groups) // 4]); sim0 = df[~df[G].isin(fit_g)].reset_index(drop=True)
else: sim0 = df

def set_pool(d):
    global sim, Xs, y, agree, ndist, smaj, soth, sdiff, acc, conf, cellc, pidc, ncell, npid
    sim = d.reset_index(drop=True); Xs = sim[cheap].fillna(0).values; y = sim[LABEL].values.astype(int)
    agree, ndist, smaj, acc = sim.agreement.values, sim.n_distinct.values, sim.s_maj.values, sim[ACC].values.astype(bool); soth, sdiff = sim.s_oth.values, sim.s_diff.values
    conf = acc & (smaj >= np.median(smaj[acc])); cellc, pidc = pd.factorize(sim[G])[0], pd.factorize(sim.pid)[0]; ncell, npid = cellc.max() + 1, pidc.max() + 1
set_pool(sim0)
print(f"pool {len(sim)} groups, label {LABEL}, base {y.mean():.4f}, hits {int(y.sum())}, accepted {int(acc.sum())}, features {cheap}", flush=True)

def rank01(x): return np.argsort(np.argsort(x)) / len(x)

def rs(o):  # ordering (best first) -> score in (0, 1], top = 1
    sc = np.empty(len(o)); sc[o] = (len(o) - np.arange(len(o))) / len(o); return sc

def orders(rng):
    n = len(y); u1, u2, u3, u4, u5 = (rng.random(n) for _ in range(5)); u6 = rng.random(n)  # fixed draw order: the learner and the fixed rules see the same tie-breaks; u6 (L7a) comes after u1..u5 so every old arm is bit-identical
    o_agree = np.lexsort((u1, -ndist, agree)); o_judge = np.lexsort((u2, np.where(acc, smaj, np.inf)))
    blend = 0.5 * rs(o_agree) + 0.5 * rs(o_judge); o_blend = np.lexsort((u3, -blend))
    o_agree_acc = np.lexsort((u4, np.where(acc, -ndist, 0), np.where(acc, agree, 0), (~acc).astype(int)))  # primary key last: accepted first, rejected random
    blend_acc = 0.5 * rs(o_agree_acc) + 0.5 * rs(o_judge); o_blend_acc = np.lexsort((u5, -blend_acc))
    tb = lambda score: np.lexsort((u6, np.where(acc, score, u4), np.where(acc, -ndist, 0), np.where(acc, agree, 0), (~acc).astype(int)))  # agree_acc keys, u4 inside accepted ties replaced by a score (ascending = audit first); rejected keep u4
    return {"agree_thr": o_agree, "judge_thr": o_judge, "blend50": o_blend, "agree_acc": o_agree_acc, "blend50_acc": o_blend_acc, "agree_acc_soth": tb(-soth), "agree_acc_sdiff": tb(sdiff)}

def run(policy, budget, seed):
    rng = np.random.default_rng(seed)
    if policy == "uniform": order = rng.permutation(len(y))
    elif policy == "oracle": order = np.argsort(-(y + 1e-6 * rng.random(len(y))))
    else: order = orders(rng)[policy]
    return int(y[order[:budget]].sum())

def run_cell_rerank(budget, seed, y_aud=None):
    """cell_rerank (see docstring). y_aud = labels the audit reveals for the cell counts (default y); the gate passes zeros to check the arm collapses to agree_acc. Hits are always scored with the real y."""
    ya = y if y_aud is None else y_aud; o = orders(np.random.default_rng(seed))["agree_acc"]; isc = conf[o]; cl = o[isc]; oth = o[~isc]
    cell_err = np.zeros(ncell); pid_err = np.zeros(npid, bool); unseen = np.ones(len(cl), bool); oi = hits = 0
    for t in range(budget):
        if isc[t]:
            j = np.flatnonzero(unseen); cand = cl[j]; k = np.argmax(cell_err[cellc[cand]] - pid_err[pidc[cand]]); g = cand[k]; unseen[j[k]] = False  # argmax = first max = agree_acc order
        else: g = oth[oi]; oi += 1
        hits += int(y[g])
        if ya[g] and not pid_err[pidc[g]]: pid_err[pidc[g]] = True; cell_err[cellc[g]] += 1
    return hits

BEST = None  # set from the full-pool 25% means before any learner arm runs

def run_greedy_warm(budget, seed, R=None, blend=False, best=None):
    rng = np.random.default_rng(seed); R = R or max(budget // 8, 40); seen = np.zeros(len(y), bool)
    score = rs(orders(rng)[best or BEST]); warm_score = score.copy(); hits = spent = 0; next_fit = R
    while spent < budget:
        mu = np.where(~seen)[0]; c = mu[np.argsort(-score[mu])][:1]; seen[c] = True; hits += int(y[c].sum()); spent += len(c)
        if spent >= next_fit and 0 < y[seen].sum() < seen.sum():
            score = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")).fit(Xs[seen], y[seen]).predict_proba(Xs)[:, 1]
            if blend: score = 0.5 * rank01(warm_score) + 0.5 * rank01(score)
            next_fit += R
    return hits

FIXED = ("agree_thr", "judge_thr", "blend50", "agree_acc", "blend50_acc")
ALL = ("uniform",) + FIXED + ("cell_rerank", "greedy_warm", "greedy_blend", "oracle")
POL = {p: (lambda b, s, p=p: run(p, b, s)) for p in ("uniform", "oracle") + FIXED}
POL["cell_rerank"] = lambda b, s: run_cell_rerank(b, s); POL["cell_rerank_noerr"] = lambda b, s: run_cell_rerank(b, s, y_aud=np.zeros(len(y), int))
POL["greedy_warm_norefit"] = lambda b, s: run_greedy_warm(b, s, R=10**9)  # sanity: refit never fires, must equal the best fixed rule per seed
POL["greedy_warm"] = lambda b, s: run_greedy_warm(b, s); POL["greedy_blend"] = lambda b, s: run_greedy_warm(b, s, blend=True)
TB = ("agree_acc_soth", "agree_acc_sdiff"); FIXED7 = FIXED + TB; BEST7 = None  # L7a: the two label-free tie-breaks; BEST7 is set from the full-pool 25% means before any mode (ii) learner arm runs
for p in TB: POL[p] = (lambda b, s, p=p: run(p, b, s))
POL["greedy_warm7"] = lambda b, s: run_greedy_warm(b, s, best=BEST7); POL["greedy_blend7"] = lambda b, s: run_greedy_warm(b, s, blend=True, best=BEST7)
POL["greedy_warm7_norefit"] = lambda b, s: run_greedy_warm(b, s, R=10**9, best=BEST7)  # sanity: must equal BEST7 per seed

def pick_tb(d, frac):
    """XFIT: choose s_oth (higher worse) versus s_diff (lower worse) on pool d from d's own labels only, by within-boundary-tier AUC (tier = agreement of the group in the last audited slot of the agree_acc key order at budget int(frac * len(d))). Tie goes to s_oth; fewer than 3 errors or 3 non-errors in the tier falls back to s_oth (flagged)."""
    ac, ag, nd, yy = d[ACC].values.astype(bool), d.agreement.values, d.n_distinct.values, d[LABEL].values.astype(int)
    idx = np.flatnonzero(ac); o = idx[np.lexsort((-nd[idx], ag[idx]))]; budget = int(frac * len(d)); last = o[min(budget, len(o)) - 1]
    m = ac & (ag == ag[last]); ym = yy[m]; out = {"tier_agreement": float(ag[last]), "n_tier": int(m.sum()), "n_err_tier": int(ym.sum()), "budget": budget}
    if ym.sum() < 3 or (1 - ym).sum() < 3: out.update(auc_soth=None, auc_sdiff=None, choice="s_oth", fallback=True); return out
    a1, a2 = float(roc_auc_score(ym, d.s_oth.values[m])), float(roc_auc_score(ym, -d.s_diff.values[m])); out.update(auc_soth=a1, auc_sdiff=a2, choice="s_oth" if a1 >= a2 else "s_diff", fallback=False); return out

if XFIT:  # cross-fit verdict mode, own JSON; exits at the end of this block
    B, NS = int(os.environ["BOOT_B"]), int(os.environ.get("BOOT_SEEDS", 4)); FRAC = float(os.environ.get("BOOT_FRAC", 0.25)); XA = ("uniform",) + FIXED + TB + ("tb_xfit", "greedy_warm", "greedy_blend", "oracle"); halves = []
    for h in (0, 1):
        held = sim0[sim0.pid % 2 == h].reset_index(drop=True); other = sim0[sim0.pid % 2 != h].reset_index(drop=True); sel, sel_in = pick_tb(other, FRAC), pick_tb(held, FRAC)  # sel = the registered choice (other half's labels); sel_in = in-sample choice, context only
        TBN = "agree_acc_soth" if sel["choice"] == "s_oth" else "agree_acc_sdiff"; POL["tb_xfit"] = lambda b, s: run(TBN, b, s); set_pool(held)
        b25h = int(0.25 * len(held)); fm = {p: float(np.mean([POL[p](b25h, s) for s in range(SEEDS)])) for p in FIXED}; BEST = max(FIXED, key=lambda p: fm[p])  # best of the five j2 rules on this half, as j2
        budget0 = int(FRAC * len(held)); fp = {"n": int(len(held)), "n_err": int(y.sum()), "budget": budget0, "budget_25": b25h, "fixed_mean_at_25": fm, "best_fixed": BEST}
        for a in XA: fp[a] = [POL[a](budget0, s) for s in range(SEEDS)]
        fp["greedy_warm_norefit"] = [POL["greedy_warm_norefit"](budget0, s) for s in range(SEEDS)]; print("half", h, "selection", json.dumps(sel), "in-sample", json.dumps(sel_in), "full half", json.dumps(fp), flush=True)
        if fp["greedy_warm_norefit"] != fp[BEST]: print("GATE FAILED: greedy_warm_norefit != best fixed on the full half-pool", h, flush=True); sys.exit(2)
        brng = np.random.default_rng(12345); rows = []
        for i in range(B):
            set_pool(held.iloc[brng.integers(0, len(held), len(held))]); budget = int(FRAC * len(sim)); r = {"boot": i, "n_err": int(y.sum()), "budget": budget}
            for a in XA: r[a] = [POL[a](budget, s) for s in range(NS)]
            if i < 20:
                r["greedy_warm_norefit"] = [POL["greedy_warm_norefit"](budget, s) for s in range(NS)]
                if r["greedy_warm_norefit"] != r[BEST]: print("GATE FAILED: norefit != best fixed at half-resample", h, i, r, flush=True); sys.exit(3)
            rows.append(r); print(h, i, json.dumps(r), flush=True)
        halves.append({"held_out_parity": h, "selected_on_parity": 1 - h, "selection": sel, "selection_in_sample_context": sel_in, "tb_xfit_is": TBN, "full_half": fp, "resamples": rows})
    if out_path: json.dump({"mode": "xfit", "n_pool": len(sim0), "accept_col": ACC, "label": LABEL, "features": cheap, "budget_frac": FRAC, "seeds_per_resample": NS, "n_boot": B, "sanity": {"gate_norefit_equals_best_fixed_fullhalf": True, "gate_norefit_equals_best_fixed_first20": True}, "halves": halves}, open(out_path, "w"), indent=1)
    sys.exit(0)

if XFIT2:  # L10a cross-fitted fixed comparator, own JSON; exits at the end of this block. No new random draw: orders() / run() / run_greedy_warm() are those of the main mode.
    import hashlib
    B, NS = int(os.environ["BOOT_B"]), int(os.environ.get("BOOT_SEEDS", 4)); FRAC = float(os.environ.get("BOOT_FRAC", 0.25)); FB = os.environ["FORCE_BEST7"]; assert FB in FIXED7, FB; BEST7 = FB  # BEST7 (global) = the frozen warm-start arm, no reselection
    X2 = ("uniform",) + FIXED7 + ("greedy_blend7", "oracle"); halves = []
    for h in (0, 1):
        held = sim0[sim0.pid % 2 == h].reset_index(drop=True); other = sim0[sim0.pid % 2 != h].reset_index(drop=True)
        set_pool(other); bo = int(FRAC * len(other)); om = {p: float(np.mean([POL[p](bo, s) for s in range(SEEDS)])) for p in FIXED7}; CH = max(FIXED7, key=lambda p: om[p])  # selection on the OTHER half only; first listed wins ties
        sel = {"n": int(len(other)), "n_err": int(y.sum()), "budget": bo, "mean_caught": om, "ch": CH, "tied_with_ch": [p for p in FIXED7 if om[p] == om[CH]]}
        set_pool(held); budget0 = int(FRAC * len(held)); fp = {"n": int(len(held)), "n_err": int(y.sum()), "budget": budget0}
        for a in X2: fp[a] = [POL[a](budget0, s) for s in range(SEEDS)]
        fp["greedy_warm7_norefit"] = [POL["greedy_warm7_norefit"](budget0, s) for s in range(SEEDS)]; hm = {p: float(np.mean(fp[p])) for p in FIXED7}; IS = max(FIXED7, key=lambda p: hm[p])
        fp.update(held_mean_fixed7=hm, in_sample_best=IS, ch=CH, frozen=FB); print("half", h, "selection", json.dumps(sel), "full half", json.dumps(fp), flush=True)
        if fp["greedy_warm7_norefit"] != fp[FB]: print("GATE FAILED: greedy_warm7_norefit != frozen arm on the full held half", h, flush=True); sys.exit(2)
        brng = np.random.default_rng(12345); rows = []; sha = hashlib.sha256()
        for i in range(B):
            idx = brng.integers(0, len(held), len(held)); sha.update(idx.tobytes()); set_pool(held.iloc[idx]); budget = int(FRAC * len(sim)); r = {"boot": i, "n_err": int(y.sum()), "budget": budget}
            for a in X2: r[a] = [POL[a](budget, s) for s in range(NS)]
            if i < 20:
                r["greedy_warm7_norefit"] = [POL["greedy_warm7_norefit"](budget, s) for s in range(NS)]
                if r["greedy_warm7_norefit"] != r[FB]: print("GATE FAILED: greedy_warm7_norefit != frozen arm at half-resample", h, i, r, flush=True); sys.exit(3)
            rows.append(r); print(h, i, json.dumps(r), flush=True)
        halves.append({"held_out_parity": h, "selected_on_parity": 1 - h, "selection": sel, "full_half": fp, "idx_sha256": sha.hexdigest(), "resamples": rows})
    if out_path: json.dump({"mode": "xfit2", "n_pool": len(sim0), "accept_col": ACC, "label": LABEL, "features": cheap, "budget_frac": FRAC, "seeds_per_resample": NS, "seeds_full_half": SEEDS, "n_boot": B, "frozen": FB, "fixed7": list(FIXED7),
                            "sanity": {"gate_norefit7_equals_frozen_fullhalf": True, "gate_norefit7_equals_frozen_first20": True}, "halves": halves}, open(out_path, "w"), indent=1)
    sys.exit(0)

B, NS = int(os.environ["BOOT_B"]), int(os.environ.get("BOOT_SEEDS", 4)); FRAC = float(os.environ.get("BOOT_FRAC", 0.25)); brng = np.random.default_rng(12345); rows = []
# full pool, unresampled, SEEDS (=8) seeds: the 25% budget for the fixed rules (best fixed rule selection), then the FRAC budget for every policy
budget0 = int(FRAC * len(sim0)); fp = {"budget": budget0, "seeds": SEEDS, "n_pool": int(len(sim0)), "n_err": int(y.sum())}
b25 = int(0.25 * len(sim0)); fp["budget_25"] = b25; fp["fixed_mean_at_25"] = {p: float(np.mean([POL[p](b25, s) for s in range(SEEDS)])) for p in FIXED}
BEST = max(FIXED, key=lambda p: fp["fixed_mean_at_25"][p]); fp["best_fixed"] = BEST  # first listed wins ties
for pol in ALL: fp[pol] = [POL[pol](budget0, s) for s in range(SEEDS)]
fp["greedy_warm_norefit"] = [POL["greedy_warm_norefit"](budget0, s) for s in range(SEEDS)]
fp["fixed_mean_at_frac"] = {p: float(np.mean(fp[p])) for p in FIXED}
fp["ratio_greedy_warm_over_best_fixed"] = float(np.mean(fp["greedy_warm"]) / np.mean(fp[BEST])); fp["ratio_greedy_blend_over_best_fixed"] = float(np.mean(fp["greedy_blend"]) / np.mean(fp[BEST]))
fp["cell_rerank_noerr"] = [POL["cell_rerank_noerr"](budget0, s) for s in range(SEEDS)]
# L7a mode (ii): seven fixed rules, BEST7 chosen on the same full-pool 25% means (first listed wins ties, the five j2 rules first); the two tie-breaks are CONTEXT arms in mode (i)
fp["fixed_mean_at_25_7"] = {p: (fp["fixed_mean_at_25"][p] if p in FIXED else float(np.mean([POL[p](b25, s) for s in range(SEEDS)]))) for p in FIXED7}
BEST7 = max(FIXED7, key=lambda p: fp["fixed_mean_at_25_7"][p]); fp["best_fixed7_natural"] = BEST7  # L8c: natural in-sample argmax kept for context
if os.environ.get("FORCE_BEST7"): assert os.environ["FORCE_BEST7"] in FIXED7, os.environ["FORCE_BEST7"]; BEST7 = os.environ["FORCE_BEST7"]  # L8c: frozen rule, no reselection
fp["best_fixed7"] = BEST7; same7 = BEST7 == BEST
for pol in TB: fp[pol] = [POL[pol](budget0, s) for s in range(SEEDS)]
for pol in ("greedy_warm7", "greedy_blend7", "greedy_warm7_norefit"): fp[pol] = list(fp[pol[:-1] if pol != "greedy_warm7_norefit" else "greedy_warm_norefit"]) if same7 else [POL[pol](budget0, s) for s in range(SEEDS)]  # alias when the warm start is the same rule
fp["fixed_mean_at_frac7"] = {p: float(np.mean(fp[p])) for p in FIXED7}; fp["ratio_greedy_blend7_over_best_fixed7"] = float(np.mean(fp["greedy_blend7"]) / np.mean(fp[BEST7]))
san = {"gate_norefit_equals_best_fixed_fullpool": bool(fp["greedy_warm_norefit"] == fp[BEST]), "gate_norefit_equals_best_fixed_first20": None, "gate_cellrerank_noerr_equals_agree_acc_fullpool": bool(fp["cell_rerank_noerr"] == fp["agree_acc"]), "gate_cellrerank_noerr_equals_agree_acc_first20": None, "best_fixed": BEST}
san["gate_norefit7_equals_best_fixed7_fullpool"] = bool(fp["greedy_warm7_norefit"] == fp[BEST7]); san["gate_norefit7_equals_best_fixed7_first20"] = None; san["best_fixed7"] = BEST7
print("full pool", json.dumps(fp), flush=True)
if not san["gate_norefit_equals_best_fixed_fullpool"]: print("GATE FAILED: greedy_warm_norefit != best fixed on the full pool", flush=True); sys.exit(2)
if not san["gate_norefit7_equals_best_fixed7_fullpool"]: print("GATE FAILED: greedy_warm7_norefit != BEST7 on the full pool", flush=True); sys.exit(4)
if not san["gate_cellrerank_noerr_equals_agree_acc_fullpool"]: print("GATE FAILED: cell_rerank with no audited error != agree_acc on the full pool", fp["cell_rerank_noerr"], fp["agree_acc"], flush=True); sys.exit(6)
for i in range(B):
    set_pool(sim0.iloc[brng.integers(0, len(sim0), len(sim0))]); budget = int(FRAC * len(sim)); r = {"boot": i, "n_err": int(y.sum()), "budget": budget}
    for pol in ALL: r[pol] = [POL[pol](budget, s) for s in range(NS)]
    for pol in TB: r[pol] = [POL[pol](budget, s) for s in range(NS)]
    for pol in ("greedy_warm7", "greedy_blend7"): r[pol] = list(r[pol[:-1]]) if same7 else [POL[pol](budget, s) for s in range(NS)]
    if i < 20:
        r["greedy_warm_norefit"] = [POL["greedy_warm_norefit"](budget, s) for s in range(NS)]
        if r["greedy_warm_norefit"] != r[BEST]: print("GATE FAILED: norefit != best fixed at resample", i, r, flush=True); sys.exit(3)
        r["greedy_warm7_norefit"] = list(r["greedy_warm_norefit"]) if same7 else [POL["greedy_warm7_norefit"](budget, s) for s in range(NS)]
        if r["greedy_warm7_norefit"] != r[BEST7]: print("GATE FAILED: norefit7 != BEST7 at resample", i, r, flush=True); sys.exit(5)
        r["cell_rerank_noerr"] = [POL["cell_rerank_noerr"](budget, s) for s in range(NS)]
        if r["cell_rerank_noerr"] != r["agree_acc"]: print("GATE FAILED: cell_rerank with no audited error != agree_acc at resample", i, r, flush=True); sys.exit(7)
    rows.append(r); print(i, json.dumps(r), flush=True)
san["gate_norefit_equals_best_fixed_first20"] = True; san["gate_norefit7_equals_best_fixed7_first20"] = True; san["gate_cellrerank_noerr_equals_agree_acc_first20"] = True
if out_path: json.dump({"n_pool": len(sim0), "pool_mode": os.environ.get("POOL", "all"), "accept_col": os.environ.get("ACCEPT_COL", "accept"), "label": LABEL, "features": cheap, "budget_frac": FRAC, "seeds_per_resample": NS, "n_boot": B,
                        "sanity": san, "full_pool": fp, "resamples": rows}, open(out_path, "w"), indent=1)
