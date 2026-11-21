"""Curator: rank pool documents by their average similarity to the individual target documents.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Method
1. Token lengths of every pool and target document are measured with tiktoken ("gpt2").
2. Exact duplicates (after whitespace/case normalisation), empty documents and
   documents with too few words are dropped.
3. Every pool and target document is embedded as an L2-normalised TF-IDF vector
   (hashed word unigrams, sublinear tf). The score of a pool document is the
   average of its cosine similarities to each of the target documents. Since
   all vectors are unit length, this average equals the dot product with the
   mean of the target vectors (the centroid), which is how it is computed.
4. Documents whose tokens-per-character ratio is an outlier relative to the
   target sample (typical of damaged / junk text) receive a heavy penalty that
   grows with the distance from the target range, measured in target standard
   deviations.
5. The best-scoring documents are selected first until the budget of T tokens
   is filled; the selected set is then shuffled (seeded by CURATE_SEED) so that
   training does not see a similarity-sorted curriculum.
"""
import os
import random
import re

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

from common import T

_WS_RE = re.compile(r"\s+")
_SCORE_CHARS = 6000    # only the first chars of each doc are used for scoring (speed)
_MIN_WORDS = 8         # docs with fewer words cannot be scored reliably
_CHUNK = 2000          # docs per tiktoken batch (keeps memory bounded)
_N_FEATURES = 2 ** 18
_PENALTY = 1.0         # penalty per target-std outside the target ratio range
_MAX_PENALTY_STDS = 10.0


def _token_lengths(enc, texts):
    """Token count of each text, computed in batches so token lists are not retained."""
    lengths = []
    for start in range(0, len(texts), _CHUNK):
        chunk = texts[start:start + _CHUNK]
        try:
            encoded = enc.encode_ordinary_batch(chunk, num_threads=8)
            lengths.extend(len(e) for e in encoded)
        except Exception:
            for t in chunk:
                try:
                    lengths.append(len(enc.encode_ordinary(t)))
                except Exception:
                    lengths.append(max(1, len(t) // 4))
    return lengths


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    def _fallback_all():
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # ---- 1. token lengths (pool and target sample) ----
    enc = tiktoken.get_encoding("gpt2")
    doc_len = _token_lengths(enc, docs)
    targets = [t for t in target_sample if t and t.strip()]
    tgt_len = _token_lengths(enc, targets)

    # Tokens-per-character statistics of the target, used to penalise junk text.
    ratios = np.array(
        [tl / len(t) for tl, t in zip(tgt_len, targets) if len(t) > 0 and tl > 0],
        dtype=np.float64,
    )
    if len(ratios) >= 5:
        ratio_lo, ratio_hi = np.percentile(ratios, [2.5, 97.5])
        ratio_std = max(float(ratios.std()), 0.01)
    elif len(ratios) > 0:
        ratio_lo, ratio_hi = float(ratios.min()), float(ratios.max())
        ratio_std = max(float(ratios.std()), 0.02)
    else:
        ratio_lo, ratio_hi, ratio_std = 0.0, float("inf"), 1.0

    # ---- 2. dedupe / filter candidates ----
    seen = set()
    cand = []
    for i, d in enumerate(docs):
        if doc_len[i] <= 0 or not d:
            continue
        head = d[:_SCORE_CHARS]
        if len(head.split()) < _MIN_WORDS:
            continue
        key = hash(_WS_RE.sub(" ", d).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        cand.append(i)

    if not cand:
        return _fallback_all()

    # ---- 3. average similarity to each individual target document ----
    if targets:
        try:
            hv = HashingVectorizer(
                n_features=_N_FEATURES,
                alternate_sign=False,
                norm=None,
                lowercase=True,
                dtype=np.float32,
            )
            X = hv.transform(docs[i][:_SCORE_CHARS] for i in cand)
            Xt = hv.transform(t[:_SCORE_CHARS] for t in targets)
            tfidf = TfidfTransformer(sublinear_tf=True, norm="l2")
            tfidf.fit(X)
            X = tfidf.transform(X)
            Xt = tfidf.transform(Xt)
            # mean over target docs of cos(doc, target_j) == doc . mean_j(target_j)
            centroid = np.asarray(Xt.mean(axis=0)).ravel()
            sim = np.asarray(X @ centroid).ravel().astype(np.float64)
        except Exception:
            sim = np.zeros(len(cand), dtype=np.float64)
    else:
        sim = np.zeros(len(cand), dtype=np.float64)

    # ---- 4. heavy penalty for tokens-per-character outliers ----
    cand_ratio = np.array([doc_len[i] / len(docs[i]) for i in cand], dtype=np.float64)
    excess = np.maximum(cand_ratio - ratio_hi, 0.0) + np.maximum(ratio_lo - cand_ratio, 0.0)
    excess_std = np.minimum(excess / ratio_std, _MAX_PENALTY_STDS)
    score = sim - _PENALTY * excess_std

    # ---- 5. take the best documents first, up to T tokens ----
    tie = np.array([rng.random() for _ in cand])
    order = np.lexsort((tie, -score))

    selected = []
    total = 0
    for j in order:
        if total >= T:
            break
        i = cand[int(j)]
        selected.append(i)
        total += doc_len[i]

    if not selected:
        return _fallback_all()

    rng.shuffle(selected)
    return selected
