"""Curator: rank pool documents by vocabulary diversity, order agreement, and proportions.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import re
import math
from collections import Counter

import numpy as np
from common import T

_TOKEN_PATTERN = r"[a-z0-9']+"
_WS = re.compile(r"\s+")
_BIGRAM_SHRINK = 20.0
_PUNCT_TOKENS = frozenset([",", ".", ";", ":", "!", "?"])
_PUNCT_PAIR_MAX = 0.03

def _token_counts(docs):
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

def _get_vocab_stats(docs, target_sample):
    """Returns target distribution and document token counts."""
    word_re = re.compile(_TOKEN_PATTERN)
    target_counts = Counter()
    for t in target_sample:
        target_counts.update(word_re.findall(t.lower()))
    
    target_total = sum(target_counts.values())
    if target_total == 0:
        return {}, word_re

    # Normalize target distribution
    target_dist = {w: count / target_total for w, count in target_counts.items()}
    return target_dist, word_re

def _get_coherence_score(doc, target_dist, word_re):
    """Calculates MSE of relative frequencies of shared tokens against target distribution."""
    tokens = word_re.findall(doc.lower())
    if not tokens:
        return 0.0
    
    counts = Counter(tokens)
    total = len(tokens)
    
    mse = 0.0
    # We only compare tokens present in target_dist
    for t, target_p in target_dist.items():
        doc_p = counts.get(t, 0) / total
        mse += (target_p - doc_p) ** 2
        
    return 1.0 / (1.0 + mse)

def _overlap_scores(docs, target_dist, word_re):
    """IDF-weighted share of each document's words found in the target vocabulary."""
    from sklearn.feature_extraction.text import CountVectorizer
    n = len(docs)
    vec = CountVectorizer(lowercase=True, token_pattern=_TOKEN_PATTERN, dtype=np.float32)
    try:
        X = vec.fit_transform(docs).tocsr()
    except ValueError:
        return np.zeros(n, dtype=np.float64)

    vocab = vec.vocabulary_
    mask = np.zeros(X.shape[1], dtype=np.float64)
    for w in target_dist:
        j = vocab.get(w)
        if j is not None:
            mask[j] = 1.0

    df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float64)
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
    X.data = np.log1p(X.data)

    num = np.asarray(X @ (idf * mask)).ravel()
    den = np.asarray(X @ idf).ravel()
    return num / (den + 30.0)

def _ws_tokens(text):
    return text.lower().split()

def _shuffled_tokens(text, rng):
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
        if nb <= 0: continue
        hits = sum(1 for b in zip(toks, toks[1:]) if doc_count[b] >= 2)
        real_scores.append(hits / (nb + _BIGRAM_SHRINK))
        stoks = _shuffled_tokens(text, rng)
        snb = len(stoks) - 1
        if snb <= 0: continue
        hits = sum(1 for b in zip(stoks, stoks[1:]) if doc_count.get(b, 0) and (doc_count[b] - (1 if b in own else 0)) >= 1)
        shuf_scores.append(hits / (snb + _BIGRAM_SHRINK))

    if not real_scores or not shuf_scores:
        return bigram_set, 0.0
    lo_real = float(np.percentile(np.asarray(real_scores), 5))
    hi_shuf = float(np.percentile(np.asarray(shuf_scores), 95))
    return bigram_set, 0.5 * (lo_real + hi_shuf)

def _order_stats(docs, bigram_set):
    n = len(docs)
    bigram_hit = np.zeros(n, dtype=np.float64)
    punct_frac = np.zeros(n, dtype=np.float64)
    contains = bigram_set.__contains__
    is_punct = _PUNCT_TOKENS.__contains__
    for i, d in enumerate(docs):
        toks = d.lower().split()
        nb = len(toks) - 1
        if nb <= 0: continue
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
    if n == 0: return []

    target_dist, word_re = _get_vocab_stats(docs, target_sample) # Fixed order
    counts = _token_counts(docs)
    scores = _overlap_scores(docs, target_dist, word_re)
    
    coherence_scores = np.zeros(n)
    for i, d in enumerate(docs):
        coherence_scores[i] = _get_coherence_score(d, target_dist, word_re)

    bigram_set, threshold = _calibrate(target_sample, seed)
    bigram_hit, punct_frac = _order_stats(docs, bigram_set)
    
    if bigram_set:
        order_ok = bigram_hit >= threshold
        final = scores * bigram_hit * coherence_scores
    else:
        order_ok = np.ones(n, dtype=bool)
        final = scores * coherence_scores
        
    punct_ok = punct_frac <= _PUNCT_PAIR_MAX

    seen = set()
    unique = np.ones(n, dtype=bool)
    for i, d in enumerate(docs):
        key = hash(_WS.sub(" ", d).strip().lower())
        if key in seen: unique[i] = False
        else: seen.add(key)

    usable = unique & (counts > 0) & order_ok & punct_ok
    if not usable.any():
        usable = unique & (counts > 0) & punct_ok
        if not usable.any(): usable = unique & (counts > 0)
        if not usable.any(): return list(range(n))

    order = np.argsort(-final, kind="stable")
    selected = []
    total = 0
    for i in order:
        if not usable[i]: continue
        selected.append(int(i))
        total += int(counts[i])
        if total >= T: break

    random.Random(seed).shuffle(selected)
    return selected
