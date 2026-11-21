"""Filter-then-rank curator.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
1. Drop junk: short docs, docs with too many non-text characters, docs with repeated lines.
2. Deduplicate on a hash of the normalized text (lowercased, whitespace collapsed), keeping the first copy.
3. Rank by TF-IDF cosine similarity to the mean of the target_sample vectors, plus a small
   wiki-format bonus when the target sample is predominantly wiki-formatted.
4. Walk the ranking, accumulating gpt2 token counts until T tokens; cycle the selection if short.
Tie-breaking is seeded by CURATE_SEED.
"""
import os
import re
import random
import hashlib
import string
from collections import Counter

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer

from common import T

MIN_CHARS = 200
MIN_CLEAN_FRAC = 0.70          # fraction of chars that must be ASCII letters/digits/space/punct
MIN_REPEAT_LINE_LEN = 10       # ignore very short repeated lines such as "}" or "-"
TFIDF_CHARS = 5000             # cap per-doc text used for TF-IDF to bound runtime
WIKI_BONUS = 0.05              # added to TF-IDF score for wiki-formatted docs
WIKI_GATE_FRAC = 0.80          # bonus applies only if this fraction of target docs are wiki-formatted

_KEEP = set(string.ascii_letters + string.digits + " \t\r\n" + string.punctuation)
_STRIP_KEEP = {ord(c): None for c in _KEEP}  # translate() deletes kept chars, leaving the rest
_WIKI_HEADER_RE = re.compile(r"^= [^=\n]+ =\n")


def wiki_fmt(doc):
    if not isinstance(doc, str):
        return False
    if _WIKI_HEADER_RE.match(doc):
        return True
    if " @-@ " in doc:
        return True
    if doc[:2000].count(" , ") >= 5:
        return True
    return False


def _is_junk(doc):
    if len(doc) < MIN_CHARS:
        return True
    unkept = len(doc.translate(_STRIP_KEEP))
    if (len(doc) - unkept) < MIN_CLEAN_FRAC * len(doc):
        return True
    counts = Counter(line.strip() for line in doc.splitlines())
    for line, c in counts.items():
        if c > 1 and len(line) >= MIN_REPEAT_LINE_LEN:
            return True
    return False


def _key(doc):
    norm = " ".join(doc.lower().split())
    return hashlib.sha1(norm.encode("utf-8", "replace")).hexdigest()


def _scores(docs, keep, target_sample):
    zeros = np.zeros(len(keep))
    texts = [docs[i][:TFIDF_CHARS] for i in keep]
    tgt = [t[:TFIDF_CHARS] for t in target_sample if isinstance(t, str) and t.strip()]
    if not tgt or not texts:
        return zeros
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    try:
        X_all = vec.fit_transform(texts + tgt)
    except ValueError:  # empty vocabulary after min_df pruning
        return zeros
    X = X_all[:len(texts)]
    Tm = X_all[len(texts):]
    c = np.asarray(Tm.mean(axis=0)).ravel()
    norm = np.linalg.norm(c)
    if norm == 0:
        return zeros
    c = c / norm
    # Rows of X and c are L2-normalized, so the dot product is cosine similarity.
    return np.asarray(X.dot(c)).ravel()


def _target_is_wiki(target_sample):
    if not target_sample:
        return False
    n_wiki = sum(1 for t in target_sample if wiki_fmt(t))
    return n_wiki >= WIKI_GATE_FRAC * len(target_sample)


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    # 1 + 2: junk filter and deduplication (first copy wins)
    seen = set()
    keep = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or _is_junk(d):
            continue
        k = _key(d)
        if k in seen:
            continue
        seen.add(k)
        keep.append(i)
    if not keep:
        keep = list(range(n))  # fallback so we never return an empty curriculum

    # 3: rank by similarity to the target centroid plus optional wiki-format bonus
    scores = np.asarray(_scores(docs, keep, target_sample), dtype=float)
    if _target_is_wiki(target_sample):
        bonus = np.array(
            [WIKI_BONUS * float(wiki_fmt(docs[i])) for i in keep], dtype=float
        )
        scores = scores + bonus
    pos = list(range(len(keep)))
    rng.shuffle(pos)                      # random order among equal scores
    pos.sort(key=lambda p: -scores[p])    # stable sort keeps the shuffled order for ties
    ranked = [keep[p] for p in pos]

    # 4: accumulate gpt2 tokens in rank order until T
    enc = tiktoken.get_encoding("gpt2")
    selected = []
    counts = []
    total = 0
    for i in ranked:
        c = len(enc.encode_ordinary(docs[i]))
        selected.append(i)
        counts.append(c)
        total += c
        if total >= T:
            break

    # If still short, cycle the selected list until T tokens
    if total < T and selected and sum(counts) > 0:
        base, base_counts = list(selected), list(counts)
        k = 0
        while total < T:
            j = k % len(base)
            selected.append(base[j])
            total += base_counts[j]
            k += 1

    return selected
