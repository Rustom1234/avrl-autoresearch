"""H5: do leaked secrets in source code cluster by repository (and by directory within repo)?
Cheap regex secret proxy over codeparrot/github-code-clean shards. Reports neighbor ratio by repo and repo+dir,
oracle concentration, bucket sizes, and cheap-feature AUC (language, size, path depth, ext) for the label."""
import glob, os, re, sys, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from aa import data as D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

GENERIC = re.compile(r"""(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|client[_-]?secret)\b\s*[:=]\s*['"][^'"\s]{8,}['"]""")
HIGH = D.SECRET  # AWS AKIA, sk-, ghp_, slack xox, private key blocks, Google AIza

files = sorted(glob.glob(os.path.join(ROOT, "data", "github-code", "**", "*.parquet"), recursive=True))
df = pd.concat([pd.read_parquet(f, columns=["code", "repo_name", "path", "language", "size"]) for f in files], ignore_index=True)
print("files", len(files), "rows", len(df), flush=True)
code = df["code"].astype(str)
df["high"] = code.map(lambda t: int(bool(HIGH.search(t))))
df["generic"] = code.map(lambda t: int(bool(GENERIC.search(t))))
df["any_secret"] = ((df.high + df.generic) > 0).astype(int)
df = df.drop(columns=["code"])
df["dir"] = df.repo_name + "/" + df.path.map(lambda p: os.path.dirname(p))
df["ext"] = df.path.map(lambda p: os.path.splitext(p)[1].lower()[:8])
df["depth"] = df.path.str.count("/")
df["log_size"] = np.log1p(df["size"])
df.to_parquet(os.path.join(ROOT, "runs", "code", "feats.parquet"), index=False)

def boot_ratio(y, g, B=200, seed=0):
    """bootstrap over groups (repos) for the neighbor ratio"""
    rng = np.random.default_rng(seed); ug = pd.unique(g); idx = pd.Series(np.arange(len(g))).groupby(g).apply(lambda s: s.values)
    vals = []
    for _ in range(B):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx[k] for k in pick]); vals.append(D.neighbor_hit_ratio(y[ii], g[ii])["ratio"])
    return [float(np.nanquantile(vals, .025)), float(np.nanquantile(vals, .975))]

def multi_hit_frac(y, g):
    h = pd.Series(y).groupby(g).sum(); return float(h[h >= 2].sum() / max(h.sum(), 1))

def bucket_stats(g):
    c = df.groupby(g).size()
    return {"n_groups": int(len(c)), "median": float(c.median()), "p90": float(c.quantile(.9)), "frac_docs_in_groups_ge3": float(c[c >= 3].sum() / c.sum())}

def concentration(y, g):
    agg = pd.DataFrame({"y": y, "g": g}).groupby("g")["y"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    cd = agg["count"].cumsum() / agg["count"].sum(); ch = agg["sum"].cumsum() / max(agg["sum"].sum(), 1)
    return {f"top{p}pct_files": float(ch[cd >= p / 100].iloc[0]) for p in (1, 3, 10)}

out = {"n_files": int(len(df)), "repo_buckets": bucket_stats("repo_name"), "dir_buckets": bucket_stats("dir")}
ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), ["language", "ext"]), ("num", StandardScaler(), ["log_size", "depth"])])
for lab in ["high", "generic", "any_secret"]:
    y = df[lab].values
    rec = {"base_rate": float(y.mean()), "ratio_repo": D.neighbor_hit_ratio(y, df.repo_name.values)["ratio"], "ratio_dir": D.neighbor_hit_ratio(y, df.dir.values)["ratio"], "ci95_repo": boot_ratio(y, df.repo_name.values), "n_hits": int(y.sum()), "frac_hits_in_multi_hit_repos": multi_hit_frac(y, df.repo_name.values),
           "cheap_auc": float(cross_val_score(make_pipeline(ct, LogisticRegression(max_iter=1000)), df[["language", "ext", "log_size", "depth"]], y, cv=5, scoring="roc_auc").mean()) if 0 < y.mean() < 1 else float("nan"),
           **concentration(y, df.repo_name.values)}
    out[lab] = rec; print(lab, json.dumps(rec), flush=True)
json.dump(out, open(os.path.join(ROOT, "runs", "code", "result.json"), "w"), indent=1)
print("\n| label | base rate | hits | neighbor ratio (repo) [95% CI] | hits in multi-hit repos | cheap-feature AUC |\n|---|---|---|---|---|---|")
for lab in ["high", "generic", "any_secret"]:
    r = out[lab]; print(f"| {lab} | {r['base_rate']:.4f} | {r['n_hits']} | {r['ratio_repo']:.1f}x [{r['ci95_repo'][0]:.1f}, {r['ci95_repo'][1]:.1f}] | {r['frac_hits_in_multi_hit_repos']*100:.0f}% | {r['cheap_auc']:.2f} |")
print("buckets:", json.dumps({k: out[k] for k in ("repo_buckets", "dir_buckets")}))
