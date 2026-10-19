"""L7a smoke test helper (CPU): writes a SYNTHETIC feats_J.parquet from a real pool's NON-label columns (pid, agreement, n_distinct, token stats, cells, PCs, s_maj, s_oth, s_diff, accept); judge_error is synthetic. No real label column (gold, majority_correct, proxy_error, pass_rate, judge_error*) is loaded.
usage: l7a_smoke_synth.py <real_feats_J.parquet> <out_dir> <variant: null|planted_soth|planted_sdiff> [seed]
null: judge_error ~ Bernoulli(0.10) among accepted groups, independent of everything.
planted_soth: accepted groups only, logit p = -3.2 + 9 min(1 - agreement, 0.25) + 2.0 z, z = s_oth standardized within its agreement tier among accepted groups (tiers with fewer than 5 groups get z = 0).
planted_sdiff: same with z = minus s_diff standardized within its agreement tier (lower s_diff = more likely wrong), AND s_diff is first REPLACED by an independent Gaussian column with the real s_diff's mean and SD (s_oth then carries no information about the label).
ERRATUM (found by the first smoke run, before any real run): with the real s_diff the planted_sdiff world cannot satisfy the registered S3 pick criterion, because s_diff = s_maj - s_oth and within the 0.875 tier corr(s_oth, -s_diff) is 0.878 / 0.951 / 0.940 on rm / judge3b / Qwen3-4B, so a planted effect on -s_diff is almost the same signal as s_oth and the cross-fit picks between two near-duplicates (3 of 6 picks right, a coin flip). The generator was changed as above; the registered S3 bin was not changed.
Registered in the docstring of exp/sim_warm_boot_j4.py before any run."""
import sys, os
import numpy as np, pandas as pd
src, out, variant = sys.argv[1], sys.argv[2], sys.argv[3]; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0; rng = np.random.default_rng(seed); os.makedirs(out, exist_ok=True)
keep = ["pid", "agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "q_len", "log_qlen", "cell30", "cell100", "cell300"] + [f"pc{i}" for i in range(16)] + ["s_maj", "s_oth", "s_diff", "split", "accept"]
F = pd.read_parquet(src, columns=keep).sort_values("pid").reset_index(drop=True); n = len(F); acc = F.accept.values.astype(bool)
def zin(v):  # standardize within agreement tier among accepted groups
    z = np.zeros(n)
    for a in np.unique(F.agreement.values[acc]):
        m = acc & (F.agreement.values == a)
        if m.sum() >= 5 and v[m].std() > 0: z[m] = (v[m] - v[m].mean()) / v[m].std()
    return z
if variant == "planted_sdiff": F["s_diff"] = rng.normal(F.s_diff.mean(), F.s_diff.std(), n)
if variant == "null": p = np.full(n, 0.10)
else:
    z = zin(F.s_oth.values) if variant == "planted_soth" else zin(-F.s_diff.values); lg = -3.2 + 9 * np.minimum(1 - F.agreement.values, 0.25) + 2.0 * z; p = 1 / (1 + np.exp(-lg))
J = (acc & (rng.random(n) < p)).astype(int); F["judge_error"] = J; F["majority_correct"] = 1 - J
F.to_parquet(os.path.join(out, "feats_J.parquet"), index=False); print("wrote", out, variant, "accepted", int(acc.sum()), "errors", int(J.sum()))
