"""Filter-then-rank curator.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
1. Drop junk: short docs, docs with too many non-text characters, docs with repeated lines.
2. Deduplicate on a hash of the normalized text (lowercased, whitespace collapsed), keeping the first copy.
2b. Vocabulary-coverage gate: keep docs whose fraction of [a-z]+ tokens found in the
    target vocabulary is at least the 5th percentile of the target docs' own coverage.
3. Rank by TF-IDF cosine similarity to the mean of the target_sample vectors.
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
COV_PERCENTILE = 5             # coverage gate threshold percentile over target docs

_KEEP = set(string.ascii_letters + string.digits + " \t\r\n" + string.punctuation)
_STRIP_KEEP = {ord(c): None for c in _KEEP}  # translate() deletes kept chars, leaving the rest
_ALPHA_RE = re.compile(r"[a-z]+")


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


def _alpha_tokens(text):
    return _ALPHA_RE.findall(text.lower())


def _coverage(text, V):
    toks = _alpha_tokens(text)
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in V) / len(toks)


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

    # 2b: vocabulary-coverage gate
    # V is built from the target docs, and the target docs are then scored against V,
    # so each target doc's own tokens are included. Its coverage is therefore 1.0 and the
    # 5th percentile is typically 1.0 as well. The gate then keeps only docs whose tokens
    # all appear in the target vocabulary. If that would empty the curriculum, the
    # pre-gate set is kept, matching the fallback above.
    tgt_texts = [t for t in target_sample if isinstance(t, str)]
    V = set()
    for t in tgt_texts:
        V.update(_alpha_tokens(t))
    target_covs = [_coverage(t, V) for t in tgt_texts]
    if target_covs:
        threshold = float(np.percentile(target_covs, COV_PERCENTILE))
    else:
        threshold = 0.0
    gated = [i for i in keep if _coverage(docs[i], V) >= threshold]
    if gated:
        keep = gated

    # 3: rank by similarity to the target centroid, seeded tie-breaking
    scores = _scores(docs, keep, target_sample)
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
