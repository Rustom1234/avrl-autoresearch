"""L5c smoke test (CPU): writes SYNTHETIC scores.parquet + gates.json in the l5c_proxy_score.py schema so l5c_analyze.py can be tested before any real score is read.
usage: l5c_smoke_synth.py <e21_dir> <out_dir> <variant: null|planted|gatefail|failed> [seed]
null: s_maj = 3*correct + N(0, 1.5), +3 for a random 60% of the wrong majorities, independent of the prompt cell (expect NHR near its null, kill-side). planted: additionally, majority-wrong groups in the 12 cell100 cells with the most wrong majorities get s_maj ~ N(8, 0.5)
(very confidently accepted), so the judge-confident errors cluster in those cells (expect NHR well above the null p95 at top50). gatefail: gates.json with all_pass false. failed: only failed.txt."""
import sys, os, json
import numpy as np, pandas as pd
e21, out, variant = sys.argv[1], sys.argv[2], sys.argv[3]; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
os.makedirs(out, exist_ok=True); rng = np.random.default_rng(seed)
if variant == "failed": open(os.path.join(out, "failed.txt"), "w").write("synthetic: OSError: could not load model"); sys.exit(0)
F = pd.read_parquet(os.path.join(e21, "feats_T0.25.parquet")).sort_values("pid").reset_index(drop=True); n = len(F); c = F.majority_correct.values
sm = 3.0 * c + rng.normal(0, 1.5, n); sm[(c == 0) & (rng.random(n) < 0.6)] += 3.0  # 60% of wrong majorities look like right ones (judge blind spots, spread over all prompts)
if variant == "planted":
    hot = F[c == 0].cell100.value_counts().index[:12].values; m = F.cell100.isin(hot).values & (c == 0); sm[m] = rng.normal(8, 0.5, m.sum()); print("hot cells", hot, "planted errors", int(m.sum()))
co = (rng.random(n) < 0.3).astype(int); so = 3.0 * co + rng.normal(0, 1.5, n)
rows = []
for role, s, cc in (("maj", sm, c), ("oth", so, co)):
    rows.append(pd.DataFrame(dict(pid=F.pid.values, k=0, role=role, correct=cc, proxy=1 if role == "maj" else 0, tokens=100, truncated=0, s=s, mass=np.nan, top_tok=-1)))
S = pd.concat(rows, ignore_index=True); S.to_parquet(os.path.join(out, "scores.parquet"), index=False)
y = S.correct.values; rk = pd.Series(S.s.values).rank().values; n1 = int(y.sum()); n0 = len(y) - n1; auc = float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
json.dump({"4_sanity_auc": {"auc_s_vs_correct": auc, "pass": True}, "all_pass": variant != "gatefail", "synthetic": True}, open(os.path.join(out, "gates.json"), "w"), indent=1)
print("wrote", out, variant, "sanity auc", round(auc, 3), "mv-wrong", int((c == 0).sum()))
