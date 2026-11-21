"""Filter-then-rank curator.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
1. Drop junk: short docs, docs with too many non-text characters, docs with repeated lines.
2. Deduplicate on a hash of the normalized text (lowercased, whitespace collapsed), keeping the first copy.
3. Rank by a discriminative domain classifier: char n-gram TF-IDF + logistic regression
   trained with target_sample as label 1 and kept pool docs as label 0. Each pool doc is
   scored by its decision_function value (log-odds of being target-like).
4. Walk the ranking, accumulating gpt2 token counts until T tokens; cycle the selection if short.
Tie-breaking is seeded by CURATE_SEED.
"""
import os
import random
import hashlib
import string
from collections import Counter

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import T

MIN_CHARS = 200
MIN_CLEAN_FRAC = 0.70          # fraction of chars that must be ASCII letters/digits/space/punct
MIN_REPEAT_LINE_LEN = 10       # ignore very short repeated lines such as "}" or "-"
DOMAIN_CHARS = 2000            # per-doc text used for the domain classifier

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


def _scores(docs, keep, target_sample, seed):
    zeros = np.zeros(len(keep))
    texts = [
        (docs[i] if isinstance(docs[i], str) else "")[:DOMAIN_CHARS] for i in keep
    ]
    tgt = [t[:DOMAIN_CHARS] for t in target_sample if isinstance(t, str) and t.strip()]
    if not tgt or not texts:
        return zeros
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=3,
        max_features=200000,
    )
    try:
        X_all = vec.fit_transform(texts + tgt)
    except ValueError:  # empty vocabulary after min_df pruning
        return zeros
    n_pool = len(texts)
    y = np.concatenate([
        np.zeros(n_pool, dtype=int),        # pool rows -> label 0
        np.ones(len(tgt), dtype=int),       # target rows -> label 1
    ])
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
    )
    try:
        clf.fit(X_all, y)
    except ValueError:  # e.g. only one class present after processing
        return zeros
    # Log-odds of being target-like, for pool rows only.
    scores = np.asarray(clf.decision_function(X_all[:n_pool])).ravel()
    return np.nan_to_num(scores)


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

    # 3: rank by domain-classifier log-odds, seeded tie-breaking
    scores = _scores(docs, keep, target_sample, seed)
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
