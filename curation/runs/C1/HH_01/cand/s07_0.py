"""Filter-then-rank curator.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
1. Drop junk: short docs, docs with too many non-text characters, docs with repeated lines.
2. Deduplicate on a hash of the normalized text (lowercased, whitespace collapsed), keeping the first copy.
3. Remove near-duplicates: drop a doc if its word 5-gram shingle Jaccard similarity with an
   already-kept doc is at least 0.5. Candidates come from an inverted index over hashed shingles,
   and only docs sharing at least 3 shingles are compared.
4. Rank by TF-IDF cosine similarity to the mean of the target_sample vectors.
5. Walk the ranking, accumulating gpt2 token counts until T tokens; cycle the selection if short.
Tie-breaking is seeded by CURATE_SEED.
"""
import os
import random
import hashlib
import string
import zlib
from collections import Counter

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer

from common import T

MIN_CHARS = 500                # minimum document length in characters
MIN_CLEAN_FRAC = 0.70          # fraction of chars that must be ASCII letters/digits/space/punct
MIN_REPEAT_LINE_LEN = 10       # ignore very short repeated lines such as "}" or "-"
TFIDF_CHARS = 5000             # cap per-doc text used for TF-IDF to bound runtime

SHINGLE_N = 5                  # word n-gram size for near-duplicate detection
SHINGLE_CHARS = 5000           # cap per-doc text used for shingling
JACCARD_MIN = 0.5              # drop a doc if Jaccard with a kept doc is at least this
MIN_SHARED_SHINGLES = 3        # only compute Jaccard against docs sharing at least this many shingles

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


def _shingles(doc):
    """Set of crc32 hashes of word SHINGLE_N-grams from the lowercased first SHINGLE_CHARS chars."""
    words = doc[:SHINGLE_CHARS].lower().split()
    if len(words) < SHINGLE_N:
        return set()
    return {
        zlib.crc32(" ".join(words[i:i + SHINGLE_N]).encode("utf-8", "replace"))
        for i in range(len(words) - SHINGLE_N + 1)
    }


def _near_dedup(docs, keep):
    """Keep the first copy (in pool order) of each near-duplicate cluster.

    A doc is dropped if its Jaccard similarity with any already-kept doc is >= JACCARD_MIN.
    Candidate kept docs are found through an inverted index from shingle hash to kept slots.
    The shared-shingle count from the index equals the exact intersection size, since each
    kept doc's shingle set is indexed once per hash, so Jaccard is computed exactly from it.
    """
    index = {}        # shingle hash -> list of kept slot ids
    kept_sets = []    # kept slot id -> shingle set
    out = []
    for i in keep:
        sh = _shingles(docs[i])
        if not sh:
            # Too short to shingle; cannot be compared, so keep it.
            out.append(i)
            continue
        overlap = Counter()
        for h in sh:
            for s in index.get(h, ()):
                overlap[s] += 1
        is_dup = False
        for s, inter in overlap.items():
            if inter < MIN_SHARED_SHINGLES:
                continue
            other = kept_sets[s]
            union = len(sh) + len(other) - inter
            if union > 0 and inter / union >= JACCARD_MIN:
                is_dup = True
                break
        if is_dup:
            continue
        slot = len(kept_sets)
        kept_sets.append(sh)
        for h in sh:
            index.setdefault(h, []).append(slot)
        out.append(i)
    return out


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

    # 1 + 2: junk filter and exact deduplication (first copy wins)
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

    # 3: near-duplicate removal (first copy in pool order wins)
    keep = _near_dedup(docs, keep)

    if not keep:
        keep = list(range(n))  # fallback so we never return an empty curriculum

    # 4: rank by similarity to the target centroid, seeded tie-breaking
    scores = _scores(docs, keep, target_sample)
    pos = list(range(len(keep)))
    rng.shuffle(pos)                      # random order among equal scores
    pos.sort(key=lambda p: -scores[p])    # stable sort keeps the shuffled order for ties
    ranked = [keep[p] for p in pos]

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
