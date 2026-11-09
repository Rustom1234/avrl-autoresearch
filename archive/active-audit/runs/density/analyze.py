"""Density test: with 10 shards (~10M docs), how big are domain buckets, how autocorrelated is strong_pii,
and how concentrated is it? Compare against the 1-shard numbers in runs/s0-layouts/result.json."""
import glob, json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/homes/rdubash/active-audit")
from aa import data as D

t = time.time()
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("/homes/rdubash/active-audit/runs/density/feats_*.feats.parquet"))], ignore_index=True)
out = {"n_docs": int(len(df)), "n_shards": int(df["shard"].nunique()), "base_strong_pii": float(df["strong_pii"].mean())}
for g in ["domain", "host"]:
    sizes = df.groupby(g).size()
    out[f"{g}_bucket_sizes"] = {"n_groups": int(len(sizes)), "median": float(sizes.median()), "p90": float(sizes.quantile(.9)), "p99": float(sizes.quantile(.99)), "max": int(sizes.max()),
                                "frac_docs_in_groups_ge3": float(sizes[sizes >= 3].sum() / len(df)), "frac_docs_in_groups_ge20": float(sizes[sizes >= 20].sum() / len(df))}
    out[f"{g}_neighbor_ratio_strong_pii"] = D.neighbor_hit_ratio(df["strong_pii"].values, df[g].values)
    out[f"{g}_neighbor_ratio_email"] = D.neighbor_hit_ratio(df["email"].values, df[g].values)
    # concentration: oracle ranks groups by hit rate (groups with >= 3 docs), top x% of docs by volume
    gs = df.groupby(g)["strong_pii"].agg(["sum", "count"]); gs = gs[gs["count"] >= 3].copy(); gs["rate"] = gs["sum"] / gs["count"]
    gs = gs.sort_values("rate", ascending=False); cum_docs = gs["count"].cumsum() / len(df); cum_hits = gs["sum"].cumsum() / df["strong_pii"].sum()
    out[f"{g}_oracle_concentration"] = {f"top_{p}pct_docs": float(np.interp(p / 100, cum_docs.values, cum_hits.values)) for p in (1, 3, 10)}
    # hit-rate dispersion across sizeable buckets: how much of the variance is real vs binomial noise
    big = gs[gs["count"] >= 20]
    if len(big):
        p = big["rate"]; base = df["strong_pii"].mean(); binom_var = (base * (1 - base) / big["count"]).mean()
        out[f"{g}_rate_dispersion_ge20"] = {"n_buckets": int(len(big)), "var_rates": float(p.var()), "expected_binomial_var": float(binom_var), "excess_ratio": float(p.var() / binom_var)}
out["wall_s"] = time.time() - t
json.dump(out, open("/homes/rdubash/active-audit/runs/density/result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
