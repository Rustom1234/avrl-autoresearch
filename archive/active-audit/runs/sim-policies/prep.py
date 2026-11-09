"""Stage-0 feature cache for the sim-policies run.

Reproduces aa.data.load_fineweb(n=400000, seed=0, columns=(...)) EXACTLY (same
positional sample indices, same order) but streams row-groups so peak RAM stays
under ~1.5 GB instead of loading the whole 2.1 GB shard with text.

Writes feats.parquet (no text) and emb_texts.parquet (50k texts for the optional
embedding step).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from aa.data import FINEWEB, regex_counts, PATTERNS  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
N = 400_000
SEED = 0
COLUMNS = ["id", "url", "text", "language", "language_score", "token_count"]
EMB_N = 50_000


def main():
    t0 = time.time()
    pf = pq.ParquetFile(FINEWEB)
    total = pf.metadata.num_rows
    # identical to pandas .sample(n=N, random_state=SEED) index selection
    locs = np.random.RandomState(SEED).choice(total, size=N, replace=False).astype(np.int64)
    order = np.empty(total, dtype=np.int64)
    order.fill(-1)
    order[locs] = np.arange(N)

    import tldextract
    extract = tldextract.TLDExtract(suffix_list_urls=())

    rows = []
    emb_rows = []
    base = 0
    for rg in range(pf.metadata.num_row_groups):
        tb = pf.read_row_groups([rg], columns=COLUMNS)
        m = tb.num_rows
        pos = order[base:base + m]
        sel = np.nonzero(pos >= 0)[0]
        base += m
        if len(sel) == 0:
            continue
        ids = tb.column("id").take(sel).to_pylist()
        urls = tb.column("url").take(sel).to_pylist()
        texts = tb.column("text").take(sel).to_pylist()
        langs = tb.column("language").take(sel).to_pylist()
        lscore = tb.column("language_score").take(sel).to_numpy(zero_copy_only=False)
        tcount = tb.column("token_count").take(sel).to_numpy(zero_copy_only=False)
        opos = pos[sel]
        for k in range(len(sel)):
            e = extract(urls[k] or "")
            dom = ".".join(p for p in (e.domain, e.suffix) if p)
            c = regex_counts(texts[k])
            r = {"order": int(opos[k]), "id": ids[k], "domain": dom, "tld": e.suffix,
                 "language": langs[k], "language_score": float(lscore[k]),
                 "token_count": int(tcount[k])}
            r.update(c)
            rows.append(r)
            if opos[k] < EMB_N:
                emb_rows.append({"order": int(opos[k]), "text": texts[k][:1000]})
        del tb, texts

    df = pd.DataFrame(rows).sort_values("order").reset_index(drop=True)
    pat = list(PATTERNS)
    df["any_pii"] = (df[pat].sum(axis=1) > 0).astype(int)
    df["strong_pii"] = ((df["email"] > 0) | (df["phone"] > 0) | (df["ssn"] > 0)
                        | (df["credit"] > 0) | (df["secret"] > 0)).astype(int)
    df.to_parquet(os.path.join(OUT, "feats.parquet"), index=False)
    pd.DataFrame(emb_rows).sort_values("order").reset_index(drop=True).to_parquet(
        os.path.join(OUT, "emb_texts.parquet"), index=False)
    with open(os.path.join(OUT, "prep_meta.json"), "w") as f:
        __import__("json").dump({"n": int(len(df)), "seed": SEED,
                                 "wall_clock_sec": round(time.time() - t0, 1)}, f)
    print(f"n={len(df)} strong_pii_rate={df.strong_pii.mean():.5f} "
          f"any_pii_rate={df.any_pii.mean():.5f} domains={df.domain.nunique()} "
          f"wall={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
