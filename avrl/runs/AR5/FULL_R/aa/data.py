"""Shared loader and cheap sensitivity signals for the active-audit experiments.

All agents use this so results are comparable. Cheap regex labels are STATE FEATURES and a proxy
label for stage 0; they are not the expensive detector.
"""
from __future__ import annotations
import glob, hashlib, os, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINEWEB = os.path.join(ROOT, "data", "fineweb", "sample", "10BT", "000_00000.parquet")
STACK_GLOB = os.path.join(ROOT, "data", "stack-smol", "data", "*", "*.parquet")

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\d)")
IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
CREDIT = re.compile(r"(?<!\d)(?:\d{4}[ -]){3}\d{4}(?!\d)")
SECRET = re.compile(r"(?:AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{36}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AIza[0-9A-Za-z_-]{35})")
STREET = re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Blvd\.|Lane|Ln\.|Drive|Dr\.)\b")

PATTERNS = {"email": EMAIL, "phone": PHONE, "ipv4": IPV4, "ssn": SSN, "credit": CREDIT, "secret": SECRET, "street": STREET}


def load_fineweb(n: int | None = None, seed: int = 0, columns=("id", "url", "text", "dump", "language", "token_count")) -> pd.DataFrame:
    """Deterministic subsample of the FineWeb shard. n=None loads everything (about 1M rows, ~2 GB)."""
    df = pd.read_parquet(FINEWEB, columns=list(columns))
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed)
    return df.reset_index(drop=True)


def load_stack(n: int | None = None, seed: int = 0, langs: list[str] | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(STACK_GLOB))
    if langs:
        files = [f for f in files if os.path.basename(os.path.dirname(f)) in langs]
    df = pd.concat([pd.read_parquet(f, columns=["content", "lang", "repository_name", "path", "size"]) for f in files], ignore_index=True)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed)
    return df.reset_index(drop=True)


def regex_counts(text: str) -> dict:
    """Counts per cheap pattern. Cheap state features and the stage-0 proxy label."""
    return {k: len(p.findall(text)) for k, p in PATTERNS.items()}


def add_regex_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    feats = pd.DataFrame([regex_counts(t) for t in df[text_col].astype(str)])
    out = pd.concat([df.reset_index(drop=True), feats], axis=1)
    out["any_pii"] = (feats.sum(axis=1) > 0).astype(int)
    out["strong_pii"] = ((feats["email"] > 0) | (feats["phone"] > 0) | (feats["ssn"] > 0) | (feats["credit"] > 0) | (feats["secret"] > 0)).astype(int)
    return out


def domain_of(url: str) -> str:
    import tldextract
    e = tldextract.extract(url or "")
    return ".".join(p for p in (e.domain, e.suffix) if p)


def add_domains(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    df = df.copy()
    df["domain"] = [domain_of(u) for u in df[url_col].astype(str)]
    df["path_prefix"] = [re.sub(r"^https?://[^/]+", "", u or "").split("/")[1] if "/" in re.sub(r"^https?://[^/]+", "", u or "") else "" for u in df[url_col].astype(str)]
    return df


def neighbor_hit_ratio(labels: np.ndarray, groups: np.ndarray) -> dict:
    """Autocorrelation of a binary label within groups (e.g. domain): P(hit | a same-group neighbor is a hit) / P(hit).

    Leave-one-out within group: for each doc, the hit rate of the other docs in its group.
    """
    df = pd.DataFrame({"y": labels.astype(float), "g": groups})
    base = df["y"].mean()
    gs = df.groupby("g")["y"].agg(["sum", "count"])
    df = df.join(gs, on="g")
    multi = df[df["count"] > 1]
    loo = (multi["sum"] - multi["y"]) / (multi["count"] - 1)
    p_hit_given_neighbor_hit = multi.loc[loo > 0, "y"].mean() if (loo > 0).any() else float("nan")
    return {
        "base_rate": float(base),
        "p_hit_given_group_neighbor_hit": float(p_hit_given_neighbor_hit),
        "ratio": float(p_hit_given_neighbor_hit / base) if base > 0 else float("nan"),
        "n_docs_in_multi_doc_groups": int(len(multi)),
        "n_groups": int(len(gs)),
    }


def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
