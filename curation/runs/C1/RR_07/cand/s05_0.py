"""Curator: rank pool documents by local contextual bigram overlap and vocabulary overlap.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Steps:
  1. Count the tokens of every pool document with tiktoken ("gpt2").
  2. Score every document by its IDF-weighted word overlap with the vocabulary
     of target_sample.
  3. Score every document by bigram overlap: the share of adjacent whitespace-token 
     bigrams that also occur in the target_sample (using a shared bigram set).
  4. Calibrate a bigram_hit threshold without labels: score the real target
     documents and word-shuffled copies of them (leave-one-out bigram set) and
     take the midpoint between the 5th percentile of the real scores and the
     95th percentile of the shuffled scores.
  5. Mark as unusable: documents below the threshold, documents where more than 3% 
     of adjacent tokens are punctuation pairs (', ,', '. ,', ...), exact duplicates 
     and empty documents.
  6. Rank the usable documents by the Local Contextual Overlap score (Word Overlap * 
     Bigram Overlap), pick the best until the training budget of T tokens is filled, 
     then shuffle the picked set with the seed from CURATE_SEED.
"""
import os
import random
import re
from collections import Counter

import numpy as np

from common import T

_TOKEN_PATTERN = r"[a-z0-9']+"
_WS = re.compile(r"\s+")

_BIGRAM_SHRINK = 20.0
_PUNCT_TOKENS = frozenset([",", ".", ";", ":", "!", "?"])
_PUNCT_PAIR_MAX = 0.03


def _token_counts(docs):
    """Number of gpt2 tokens in each document."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        counts = np.empty(len(docs), dtype=np.int64)
        chunk = 2000
        for start in range(0, len(docs), chunk):
            batch = docs[start:start + chunk]
            try:
                encoded = enc.encode_ordinary_batch(batch, num_threads=8)
                lens = [len(e) for e in encoded]
            except Exception:
                lens = []
                for d in batch:
                    try:
                        lens.append(len(enc.encode_ordinary(d)))
                    except Exception:
                        lens.append(max(1, len(d) // 4))
            counts[start:start + len(batch)] = lens
        return counts
    except Exception:
        return np.array([max(1, len(d) // 4) for d in docs], dtype=np.int64)


def _overlap_scores(docs, target_sample):
    """IDF-weighted share of each document's words found in the target vocabulary."""
    from sklearn.feature_extraction.text import CountVectorizer

    n = len(docs)
    vec = CountVectorizer(lowercase=True, token_pattern=_TOKEN_PATTERN, dtype=np.float32)
    try:
        X = vec.fit_transform(docs).tocsr()
    except ValueError:
        return np.zeros(n, dtype=np.float64)

    # Target vocabulary.
    word_re = re.compile(_TOKEN_PATTERN)
    target_vocab = set()
    for t in target_sample:
        target_vocab.update(word_re.findall(t.lower()))

    vocab = vec.vocabulary_
    mask = np.zeros(X.shape[1], dtype=np.float64)
    for w in target_vocab:
        j = vocab.get(w)
        if j is not None:
            mask[j] = 1.0

    # Document frequency -> idf, so rare words (junk, other domains) weigh more.
    df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float64)
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0

    # Sublinear term frequency: repeated words do not dominate.
    X.data = np.log1p(X.data)

    num = np.asarray(X @ (idf * mask)).ravel()
    den = np.asarray(X @ idf).ravel()
    # Small shrinkage so very short documents cannot get a perfect score.
    return num / (den + 30.0)


def _ws_tokens(text):
    """Lowercased whitespace tokens; punctuation tokens (',', '.', '@-@') are kept."""
    return text.lower().split()


def _shuffled_tokens(text, rng):
    """Tokens of the document with everything after the title line shuffled."""
    stripped = text.lstrip()
    head, sep, body = stripped.partition("\n")
    if sep:
        head_toks = _ws_tokens(head)
        body_toks = _ws_tokens(body)
    else:
        head_toks = []
        body_toks = _ws_tokens(stripped)
    rng.shuffle(body_toks)
    return head_toks + body_toks


def _calibrate(target_sample, seed):
    """Return (target bigram set, bigram_hit threshold).

    Real and word-shuffled target documents are both scored against a
    leave-one-out bigram set (bigrams seen in at least one *other* target
    document).  The threshold is the midpoint between the 5th percentile of the
    real scores and the 95th percentile of the shuffled scores.
    """
    tok_lists = [_ws_tokens(t) for t in target_sample]
    doc_sets = [set(zip(toks, toks[1:])) for toks in tok_lists]

    doc_count = Counter()
    for s in doc_sets:
        doc_count.update(s)
    bigram_set = set(doc_count)

    rng = random.Random(seed)
    real_scores = []
    shuf_scores = []
    for text, toks, own in zip(target_sample, tok_lists, doc_sets):
        nb = len(toks) - 1
        if nb <= 0:
            continue
        # Real document: every bigram is in its own set, so it must occur in
        # at least two target documents to be in the leave-one-out set.
        hits = 0
        for b in zip(toks, toks[1:]):
            if doc_count[b] >= 2:
                hits += 1
        real_scores.append(hits / (nb + _BIGRAM_SHRINK))

        stoks = _shuffled_tokens(text, rng)
        snb = len(stoks) - 1
        if snb <= 0:
            continue
        hits = 0
        for b in zip(stoks, stoks[1:]):
            c = doc_count.get(b, 0)
            if c and (c - (1 if b in own else 0)) >= 1:
                hits += 1
        shuf_scores.append(hits / (snb + _BIGRAM_SHRINK))

    if not real_scores or not shuf_scores:
        return bigram_set, 0.0

    lo_real = float(np.percentile(np.asarray(real_scores), 5))
    hi_shuf = float(np.percentile(np.asarray(shuf_scores), 95))
    return bigram_set, 0.5 * (lo_real + hi_shuf)


def _order_stats(docs, bigram_set):
    """Per document: bigram_hit and the fraction of punctuation-pair bigrams."""
    n = len(docs)
    bigram_hit = np.zeros(n, dtype=np.float64)
    punct_frac = np.zeros(n, dtype=np.float64)
    contains = bigram_set.__contains__
    is_punct = _PUNCT_TOKENS.__contains__
    for i, d in enumerate(docs):
        toks = d.lower().split()
        nb = len(toks) - 1
        if nb <= 0:
            continue
        hits = sum(map(contains, zip(toks, toks[1:])))
        bigram_hit[i] = hits / (nb + _BIGRAM_SHRINK)
        flags = list(map(is_punct, toks))
        if any(flags):
            pairs = sum(1 for a, b in zip(flags, flags[1:]) if a and b)
            punct_frac[i] = pairs / nb
    return bigram_hit, punct_frac


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0:
        return []

    counts = _token_counts(docs)
    vocab_scores = _overlap_scores(docs, target_sample)

    # Word-order score and its label-free threshold.
    bigram_set, threshold = _calibrate(target_sample, seed)
    bigram_hit, punct_frac = _order_stats(docs, bigram_set)
    
    if bigram_set:
        order_ok = bigram_hit >= threshold
        # Local Contextual Overlap: word overlap * bigram hit
        final_scores = vocab_scores * bigram_hit
    else:
        # No target bigrams at all: the word-order score carries no information.
        order_ok = np.ones(n, dtype=bool)
        final_scores = vocab_scores

    punct_ok = punct_frac <= _PUNCT_PAIR_MAX

    # Exact duplicates (after whitespace/case normalisation): keep the first copy.
    seen = set()
    unique = np.ones(n, dtype=bool)
    for i, d in enumerate(docs):
        key = hash(_WS.sub(" ", d).strip().lower())
        if key in seen:
            unique[i] = False
        else:
            seen.add(key)

    base = unique & (counts > 0)
    usable = base & order_ok & punct_ok

    if not usable.any():
        # Last resort so the trainer never gets an empty list: nothing passed
        # the word-order threshold, so fall back to the ranking alone.
        usable = base & punct_ok
        if not usable.any():
            usable = base
        if not usable.any():
            return list(range(n))

    order = np.argsort(-final_scores, kind="stable")

    # Fill to T tokens with usable documents only.  If they hold fewer than T
    # tokens the list is returned as is and the trainer cycles it.
    selected = []
    total = 0
    for i in order:
        if not usable[i]:
            continue
        selected.append(int(i))
        total += int(counts[i])
        if total >= T:
            break

    random.Random(seed).shuffle(selected)
    return selected
