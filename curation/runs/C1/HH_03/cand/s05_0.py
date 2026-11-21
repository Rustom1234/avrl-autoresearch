"""Filter-then-rank curator: drop exact duplicates and junk, drop docs whose
surface format is far more punctuation-heavy than the target domain, then
order the surviving pool by TF-IDF cosine similarity to the centroid of the
target sample.

select(docs, target_sample) -> ordered list of indices into docs (no repeats).
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import hashlib
import string

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SPAM_MARKERS = [
    "discount code", "coupon", "cheap", "buy now", "free shipping", "click here",
]
HEADER_RE = re.compile(r"^\s*=\s*[^=\n].*?[^=\n]\s*=\s*$", re.MULTILINE)
WS_RE = re.compile(r"\s+")

PUNCT_CHARS = set(string.punctuation)
FMT_MIN_TOKENS = 50
FMT_TARGET_PERCENTILE = 10


def _seed():
    return int(os.environ.get("CURATE_SEED", 0))


def _normalize(text):
    return WS_RE.sub(" ", text.lower()).strip()


def _non_ascii_share(text):
    if not text:
        return 1.0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii / len(text)


def _is_junk(text):
    if _non_ascii_share(text) > 0.02:
        return True
    lowered = text.lower()
    for marker in SPAM_MARKERS:
        if marker in lowered:
            return True
    if not HEADER_RE.search(text):
        return True
    return False


def _is_punct_token(tok):
    if tok == "@-@":
        return True
    return len(tok) > 0 and all(c in PUNCT_CHARS for c in tok)


def _fmt(text):
    """Return (fmt, n_tokens): share of whitespace tokens that are pure
    punctuation or '@-@', and the total whitespace-token count."""
    toks = text.split()
    if not toks:
        return 0.0, 0
    n_punct = sum(1 for t in toks if _is_punct_token(t))
    return n_punct / len(toks), len(toks)


def select(docs, target_sample):
    seed = _seed()

    # (1) exact duplicates by hash of normalized text
    seen = set()
    survivors = []
    for i, doc in enumerate(docs):
        key = hashlib.sha1(_normalize(doc).encode("utf-8", "replace")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        survivors.append(i)

    # (2) junk filter
    survivors = [i for i in survivors if not _is_junk(docs[i])]

    # (2b) surface-format filter: drop survivors (with >= FMT_MIN_TOKENS tokens)
    # whose punctuation/"@-@" token share is below the 10th percentile of the
    # target sample's share. If this would remove every survivor, keep the
    # pre-filter survivors instead so training still has data.
    target_fmts = [_fmt(t)[0] for t in target_sample]
    if target_fmts and survivors:
        cutoff = float(np.percentile(target_fmts, FMT_TARGET_PERCENTILE))
        kept = []
        for i in survivors:
            f, n_tok = _fmt(docs[i])
            if n_tok >= FMT_MIN_TOKENS and f < cutoff:
                continue
            kept.append(i)
        if kept:
            survivors = kept

    if not survivors:
        # Nothing survived; fall back to the seeded shuffle of all docs so
        # training still has data to read.
        idx = list(range(len(docs)))
        random.Random(seed).shuffle(idx)
        return idx

    # (3) TF-IDF similarity to target centroid
    surv_texts = [docs[i] for i in survivors]
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    try:
        vec.fit(list(target_sample) + surv_texts)
        target_mat = normalize(vec.transform(target_sample))
        centroid = np.asarray(target_mat.mean(axis=0))
        centroid = normalize(centroid)
        doc_mat = normalize(vec.transform(surv_texts))
        scores = np.asarray(doc_mat @ centroid.T).ravel()
    except ValueError:
        # Empty vocabulary (e.g. min_df=2 removes everything); keep survivor order.
        scores = np.zeros(len(survivors))

    # Seeded tie-breaking shuffle, then stable sort by descending score.
    order = list(range(len(survivors)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda k: -scores[k])

    return [survivors[k] for k in order]
