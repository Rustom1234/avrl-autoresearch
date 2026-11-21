"""Filter-then-rank curator.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
1. Drop junk: short docs, docs with too many non-text characters, docs with repeated lines.
2. Deduplicate on a hash of the normalized text (lowercased, whitespace collapsed), keeping the first copy.
3. Format match: if at least 80% of target_sample docs look WikiText-style (wiki_fmt), rank
   wiki-format docs ahead of all others. Otherwise skip this and rank on TF-IDF alone.
4. Rank by TF-IDF cosine similarity to the mean of the target_sample vectors, with seeded
   tie-breaking. Within each format group, TF-IDF order is preserved.
5. Walk the ranking, accumulating gpt2 token counts until T tokens; cycle the selection if short.
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
FMT_TARGET_FRAC = 0.80         # fraction of target docs that must be wiki-format to enable the feature

_KEEP = set(string.ascii_letters + string.digits + " \t\r\n" + string.punctuation)
_STRIP_KEEP = {ord(c): None for c in _KEEP}  # translate() deletes kept chars, leaving the rest


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


def _wiki_fmt(doc):
    if not isinstance(doc, str):
        return False
    if re.match(r"^= [^=\n]+ =\n", doc):
        return True
    if " @-@ " in doc:
        return True
    return doc[:2000].count(" , ") >= 5


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

    # 3: decide whether the format-match feature applies to this target domain
    n_t = len(target_sample)
    hits = sum(1 for t in target_sample if _wiki_fmt(t))
    # Integer comparison avoids float rounding: hits / n_t >= 0.8
    use_fmt = n_t > 0 and 5 * hits >= 4 * n_t
    if use_fmt:
        wiki = [_wiki_fmt(docs[i]) for i in keep]
    else:
        wiki = [False] * len(keep)

    # 4: rank by (not wiki_fmt, -tfidf, seeded shuffle position)
    scores = _scores(docs, keep, target_sample)
    pos = list(range(len(keep)))
    rng.shuffle(pos)                      # seeded shuffle; position in this list breaks ties
    shuffle_rank = [0] * len(keep)
    for r, p in enumerate(pos):
        shuffle_rank[p] = r
    order = sorted(
        range(len(keep)),
        key=lambda p: (not wiki[p], -scores[p], shuffle_rank[p]),
    )
    ranked = [keep[p] for p in order]

    # 5: accumulate gpt2 tokens in rank order until T
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
