"""L6a smoke test (CPU): writes a SYNTHETIC feats_J.parquet (non-label columns of a real pool, all gold / proxy-error / accept / judge_error columns REPLACED by synthetic ones) so l6a_cell_rate.py and sim_warm_boot_j3.py can be tested before any real label is read.
usage: l6a_smoke_synth.py <real_feats_J.parquet> <out_dir> <variant: null|planted|planted_strong> [seed]. Accept = top 1,715 groups by s_maj (as the rm rule). null: judge_error ~ Bernoulli(0.10) among accepted, independent of everything (expect delta AUC near 0, CI covering 0,
cell_rerank / agree_acc CI covering 1). planted: 12 random cell100 cells ('hot') have error rate 0.40, all other cells 0.06, independent of every other feature (expect JA+cell100 delta AUC >= 0.03 with CI lower end > 0, and cell_rerank / agree_acc percentile CI lower end > 1).
planted_strong: same 12 hot cells at 0.80 and the rest at 0.03 (a power probe for the replay arm, whose bootstrap CI is wide). majority_correct = 1 - judge_error (rejected groups count as right) so the consensus label equals judge_error here. No sens columns."""
import sys, os
import numpy as np, pandas as pd
src, out, variant = sys.argv[1], sys.argv[2], sys.argv[3]; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0; rng = np.random.default_rng(seed); os.makedirs(out, exist_ok=True)
keep = ["pid", "agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "q_len", "log_qlen", "cell30", "cell100", "cell300"] + [f"pc{i}" for i in range(16)] + ["s_maj", "s_oth", "s_diff", "split"]
F = pd.read_parquet(src, columns=keep).sort_values("pid").reset_index(drop=True); n = len(F)
acc = np.zeros(n, bool); acc[np.argsort(-F.s_maj.values, kind="stable")[:1715]] = True
if variant == "null": p = np.full(n, 0.10)
else:
    ph, pc = (0.40, 0.06) if variant == "planted" else (0.80, 0.03); hot = rng.choice(F.cell100.unique(), 12, replace=False); p = np.where(F.cell100.isin(hot).values, ph, pc); print("hot cells", sorted(int(h) for h in hot))
J = (acc & (rng.random(n) < p)).astype(int); F["accept"] = acc.astype(int); F["judge_error"] = J; F["majority_correct"] = 1 - J
F.to_parquet(os.path.join(out, "feats_J.parquet"), index=False); print("wrote", out, variant, "accepted", int(acc.sum()), "errors", int(J.sum()))
