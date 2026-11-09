"""Regex-featurize all downloaded FineWeb shards in parallel; write one compact parquet (no text)."""
import glob, os, sys, time
from multiprocessing import Pool
import pandas as pd
sys.path.insert(0, "/homes/rdubash/active-audit")
from aa import data as D
from urllib.parse import urlsplit

def one(path):
    t = time.time()
    df = pd.read_parquet(path, columns=["id", "url", "token_count", "language_score"])
    txt = pd.read_parquet(path, columns=["text"])["text"]
    feats = pd.DataFrame([D.regex_counts(x) for x in txt.astype(str)])
    df["strong_pii"] = ((feats["email"] > 0) | (feats["phone"] > 0) | (feats["ssn"] > 0) | (feats["credit"] > 0) | (feats["secret"] > 0)).astype("int8")
    df["email"] = (feats["email"] > 0).astype("int8")
    df["host"] = [urlsplit(u).netloc.lower().removeprefix("www.") for u in df["url"].astype(str)]
    df["domain"] = [D.domain_of(u) for u in df["url"].astype(str)]
    df["shard"] = os.path.basename(path)[:3]
    out = path.replace("data/fineweb/sample/10BT/", "runs/density/feats_").replace(".parquet", ".feats.parquet")
    df.drop(columns=["url"]).to_parquet(out, index=False)
    return out, len(df), time.time() - t

if __name__ == "__main__":
    paths = sorted(glob.glob("/homes/rdubash/active-audit/data/fineweb/sample/10BT/*.parquet"))
    with Pool(3) as p:
        for out, n, dt in p.imap_unordered(one, paths):
            print(out, n, f"{dt:.0f}s", flush=True)
