"""Stage 0: measure autocorrelation of cheap-regex PII labels across candidate corpus layouts.

Agent: s0-layouts. Writes only under runs/s0-layouts/. Uses aa.data shared loader (no
reimplementation of the parquet reader or the regex detector). Fixed seeds throughout.

Run: source .venv/bin/activate && python runs/s0-layouts/stage0.py
"""
from __future__ import annotations
import json, os, sys, time
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aa.data import load_fineweb, add_regex_features, domain_of, neighbor_hit_ratio, stable_hash

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
FEATS_PATH = os.path.join(RUN_DIR, "feats.parquet")
RESULT_PATH = os.path.join(RUN_DIR, "result.json")

SEED = 0
N_FULL = None  # try full shard first; fall back to 400000 if memory is a problem (see load step)
FALLBACK_N = 400000
EMBED_N = 100000
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_TRUNC_TOKENS = 256
KMEANS_K = 1000
N_BOOT = 200

timings = {}
notes = []


def tld_of(url: str) -> str:
    import tldextract
    e = tldextract.extract(url or "")
    return e.suffix or ""


def hostname_of(url: str) -> str:
    try:
        return urlsplit(url or "").netloc
    except Exception:
        return ""


def hostname_path1_of(url: str) -> str:
    try:
        sp = urlsplit(url or "")
        segs = [s for s in sp.path.split("/") if s]
        return sp.netloc + "/" + (segs[0] if segs else "")
    except Exception:
        return ""


def _group_stats(labels: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"y": labels.astype(float), "g": groups})
    return df.groupby("g")["y"].agg(["sum", "count"]).reset_index(drop=True)


def _ratio_from_group_stats(count: np.ndarray, s: np.ndarray) -> float:
    """Vectorized reimplementation of aa.data.neighbor_hit_ratio's leave-one-out math,
    operating on per-group (count, sum) arrays so it can be applied to bootstrap-weighted
    group multisets without rebuilding row-level frames. See main() for a sanity check
    against aa.data.neighbor_hit_ratio on the true (non-resampled) data."""
    base_rate = s.sum() / count.sum()
    multi = count > 1
    c, n_hit = count[multi], s[multi]
    n_nonhit = c - n_hit
    select_hit = np.where(n_hit > 1, n_hit, 0.0)      # hit docs whose OTHER same-group docs include >=1 hit
    select_nonhit = np.where(n_hit > 0, n_nonhit, 0.0)  # non-hit docs whose same-group docs include >=1 hit
    denom = select_hit.sum() + select_nonhit.sum()
    if denom == 0 or base_rate == 0:
        return float("nan")
    p_hit_given_neighbor_hit = select_hit.sum() / denom
    return float(p_hit_given_neighbor_hit / base_rate)


def bootstrap_ratio_ci(labels: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """95% CI on the neighbor-hit ratio, resampling groups (not docs) with replacement.
    Duplicating a group in a bootstrap sample does not change its internal leave-one-out
    values (they depend only on that group's own count/sum), so the resampled statistic can
    be computed from group-level (count, sum) with multiplicities, which is what makes 200
    resamples over corpora with hundreds of thousands of groups tractable."""
    gs = _group_stats(labels, groups)
    count = gs["count"].to_numpy(dtype=float)
    s = gs["sum"].to_numpy(dtype=float)
    n_groups = len(gs)
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_groups, size=n_groups)
        w = np.bincount(idx, minlength=n_groups).astype(float)
        r = _ratio_from_group_stats_weighted(count, s, w)
        if not np.isnan(r):
            ratios.append(r)
    if not ratios:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return [float(lo), float(hi)]


def _ratio_from_group_stats_weighted(count: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    base_rate = (s * w).sum() / (count * w).sum()
    multi = count > 1
    c, n_hit, ww = count[multi], s[multi], w[multi]
    n_nonhit = c - n_hit
    select_hit = np.where(n_hit > 1, n_hit, 0.0) * ww
    select_nonhit = np.where(n_hit > 0, n_nonhit, 0.0) * ww
    denom = select_hit.sum() + select_nonhit.sum()
    if denom == 0 or base_rate == 0:
        return float("nan")
    return float(select_hit.sum() / denom / base_rate)


def coverage(groups: np.ndarray) -> float:
    s = pd.Series(groups)
    counts = s.value_counts()
    multi_groups = counts[counts > 1].index
    return float(s.isin(multi_groups).mean())


def layout_report(name: str, labels: np.ndarray, groups: np.ndarray) -> dict:
    t0 = time.time()
    r = neighbor_hit_ratio(labels, groups)  # authoritative, from aa.data
    # sanity check: the vectorized group-stats formula used for the bootstrap must agree
    # with aa.data.neighbor_hit_ratio on the true (non-resampled) data before it is trusted
    # to drive 200 resamples.
    gs = _group_stats(labels, groups)
    check_ratio = _ratio_from_group_stats(gs["count"].to_numpy(dtype=float), gs["sum"].to_numpy(dtype=float))
    if not (np.isnan(r["ratio"]) and np.isnan(check_ratio)):
        if abs(r["ratio"] - check_ratio) > 1e-6 * max(1.0, abs(r["ratio"])):
            notes.append(f"{name}: bootstrap formula mismatch vs neighbor_hit_ratio ({check_ratio} vs {r['ratio']})")
    r["coverage_multi_doc_groups"] = coverage(groups)
    r["ci95_ratio"] = bootstrap_ratio_ci(labels, groups)
    r["wall_clock_s"] = time.time() - t0
    r["layout"] = name
    return r


def concentration_curve(labels: np.ndarray, groups: np.ndarray, fracs=(0.01, 0.03, 0.10)) -> dict:
    """Oracle triage bound: sort groups by their own hit rate, accumulate docs best-group-first,
    report the fraction of all positive docs captured at each doc-budget fraction (linear
    interpolation within the boundary group, since within-group doc order is unknown)."""
    df = pd.DataFrame({"y": labels.astype(float), "g": groups})
    gs = df.groupby("g")["y"].agg(["sum", "count"]).reset_index()
    gs["hit_rate"] = gs["sum"] / gs["count"]
    gs = gs.sort_values("hit_rate", ascending=False)
    total_docs = gs["count"].sum()
    total_hits = gs["sum"].sum()
    cum_docs = gs["count"].cumsum().to_numpy()
    cum_hits = gs["sum"].cumsum().to_numpy()
    out = {}
    for frac in fracs:
        budget = frac * total_docs
        idx = np.searchsorted(cum_docs, budget)
        if idx == 0:
            prev_docs, prev_hits = 0, 0
        else:
            prev_docs, prev_hits = cum_docs[idx - 1], cum_hits[idx - 1]
        if idx >= len(gs):
            frac_hits = total_hits
        else:
            group_docs = gs["count"].to_numpy()[idx]
            group_hits = gs["sum"].to_numpy()[idx]
            remaining_budget = budget - prev_docs
            interp = group_hits * (remaining_budget / group_docs) if group_docs > 0 else 0.0
            frac_hits = prev_hits + interp
        out[f"top_{int(frac*100)}pct_groups_by_doccount"] = float(frac_hits / total_hits) if total_hits > 0 else float("nan")
    out["n_groups"] = int(len(gs))
    out["total_docs"] = int(total_docs)
    out["total_hits"] = int(total_hits)
    return out


def main():
    result = {"agent": "s0-layouts", "seed": SEED, "script": "runs/s0-layouts/stage0.py"}

    # ---- Step 1: load + featurize (cache) ----
    t0 = time.time()
    n_used = N_FULL
    if os.path.exists(FEATS_PATH):
        feats = pd.read_parquet(FEATS_PATH)
        notes.append(f"loaded cached feats from {FEATS_PATH}, n={len(feats)}")
        n_used = len(feats)
    else:
        try:
            df = load_fineweb(n=N_FULL, seed=SEED, columns=("id", "url", "text", "language", "token_count"))
        except MemoryError:
            notes.append(f"full load hit MemoryError, falling back to n={FALLBACK_N}, seed={SEED}")
            df = load_fineweb(n=FALLBACK_N, seed=SEED, columns=("id", "url", "text", "language", "token_count"))
            n_used = FALLBACK_N
        feats = add_regex_features(df, text_col="text")
        feats = feats.drop(columns=["text"])
        n_used = len(feats)
        feats.to_parquet(FEATS_PATH)
        del df
    timings["load_and_featurize_s"] = time.time() - t0
    result["n_docs"] = int(n_used)
    result["full_shard_used"] = n_used >= 1_000_000

    # ---- layout columns ----
    t0 = time.time()
    feats["domain"] = [domain_of(u) for u in feats["url"].astype(str)]
    feats["hostname"] = [hostname_of(u) for u in feats["url"].astype(str)]
    feats["hostname_path1"] = [hostname_path1_of(u) for u in feats["url"].astype(str)]
    feats["tld"] = [tld_of(u) for u in feats["url"].astype(str)]
    feats["token_decile"] = pd.qcut(feats["token_count"].rank(method="first"), 10, labels=False)
    timings["layout_columns_s"] = time.time() - t0

    strong = feats["strong_pii"].to_numpy()

    layouts = {}
    for name, col in [
        ("a_registered_domain", "domain"),
        ("b_full_hostname", "hostname"),
        ("c_hostname_path1", "hostname_path1"),
        ("d_tld", "tld"),
        ("e_token_count_decile", "token_decile"),
    ]:
        layouts[name] = layout_report(name, strong, feats[col].to_numpy())

    # ---- Step: embedding-cell layout (f) ----
    t0 = time.time()
    embed_result = None
    concentration = {}
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import MiniBatchKMeans

        rng_idx = feats.sample(n=min(EMBED_N, len(feats)), random_state=SEED).index
        # need text back for embedding subsample; re-read just those rows' text from source parquet via id join
        full_ids = None
        sub = feats.loc[rng_idx].copy()
        # text was dropped from cache; re-load text for just this subsample by id
        raw = load_fineweb(n=None, seed=SEED, columns=("id", "text"))
        raw = raw.set_index("id")
        texts = raw.loc[sub["id"], "text"].astype(str).tolist()
        del raw

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(EMBED_MODEL, device=device)
        model.max_seq_length = EMBED_TRUNC_TOKENS
        embs = model.encode(texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True)

        km = MiniBatchKMeans(n_clusters=KMEANS_K, random_state=SEED, n_init=3, batch_size=1024)
        cluster_ids = km.fit_predict(embs)

        sub_labels = sub["strong_pii"].to_numpy()
        layouts["f_embedding_cells"] = layout_report("f_embedding_cells", sub_labels, cluster_ids)
        layouts["f_embedding_cells"]["n_docs_subsample"] = int(len(sub))
        embed_result = {
            "n_embedded": int(len(sub)),
            "model": EMBED_MODEL,
            "truncate_tokens": EMBED_TRUNC_TOKENS,
            "k": KMEANS_K,
            "device": device,
        }
        concentration["f_embedding_cells"] = concentration_curve(sub_labels, cluster_ids)
    except Exception as e:
        notes.append(f"embedding layout failed: {type(e).__name__}: {e}")
    timings["embedding_layout_s"] = time.time() - t0

    # ---- Step 3: concentration curves for (a) ----
    t0 = time.time()
    concentration["a_registered_domain"] = concentration_curve(strong, feats["domain"].to_numpy())
    timings["concentration_curves_s"] = time.time() - t0

    # ---- Step 4: any_pii and email-only, layout (a) only ----
    t0 = time.time()
    extra_labels = {}
    for lname, col in [("any_pii", "any_pii"), ("email_only", "email")]:
        lab = (feats[col].to_numpy() > 0).astype(int) if col == "email" else feats[col].to_numpy()
        extra_labels[lname] = layout_report(f"a_registered_domain__{lname}", lab, feats["domain"].to_numpy())
    timings["extra_labels_s"] = time.time() - t0

    result["layouts"] = layouts
    result["embedding_info"] = embed_result
    result["concentration_curves"] = concentration
    result["extra_labels_domain_layout"] = extra_labels
    result["timings_s"] = timings
    result["notes"] = notes
    result["bootstrap_resamples"] = N_BOOT

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
