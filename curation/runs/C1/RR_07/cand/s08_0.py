"""Curator: rank pool documents by weighted token-level similarity to a target-domain distribution.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Steps:
  1. Count tokens and estimate a log-probability distribution from target_sample.
  2. Score every document by the sum of log-probabilities of its tokens.
  3. Filter by word-order/punctuation constraints.
  4. Rank and select best documents to fill T tokens.
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
    """Number of gpt2 tokens in each document."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        counts = np.empty(len(docs), dtype=np.int64)
        chunk = 1000
        for start in range(0, len(docs), chunk):
            batch = docs[start:start + chunk]
            try:
                encoded = enc.encode_ordinary_batch(batch, num_threads=1)
                lens = [len(e) for e in encoded]
            except Exception:
                lens = [max(1, len(d) // 4) for d in batch]
            counts[start:start + len(batch)] = lens
        return counts
    except Exception:
        return np.array([max(1, len(d) // 4) for d in docs], dtype=np.int64)

def _get_target_probs(target_sample):
    """Build a Laplacian smoothed multinomial log-probability table."""
    counts = Counter()
    total_tokens = 0
    word_re = re.compile(_TOKEN_PATTERN)
    for text in target_sample:
        words = word_re.findall(text.lower())
        counts.update(words)
        total_tokens += len(words)
    
    # Using Laplacian smoothing: P(w) = (count(w) + 1) / (total_tokens + unique_vocab_size)
    unique_vocab = len(counts)
    denominator = total_tokens + unique_vocab
    
    # We only store the log_probs for the vocabulary; unknown tokens get a constant penalty.
    log_probs = {}
    for word, count in counts.items():
        log_probs[word] = math.log((count + 1.0) / denominator)
    
    # Penalty for tokens not in the target vocabulary (low probability)
    # This should be lower than the smallest log_prob in the vocab.
    min_vocab_log_prob = min(log_probs.values()) if log_probs else -20.0
    unknown_penalty = min(min_vocab_log_prob - 0.5, -20.0)
    
    return log_probs, unknown_penalty

def _calibrate(target_sample, seed):
    """Calculate a bigram threshold to filter out non-coherent/shuffled text."""
    tok_lists = [text.lower().split() for text in target_sample]
    doc_counts = Counter()
    for toks in tok_lists:
        for b in zip(toks, toks[1:]):
            doc_counts[b] += 1
    
    bigram_set = set(doc_counts)
    rng = random.Random(seed)
    real_scores = []
    shuf_scores = []
    
    for text, toks in zip(target_sample, tok_lists):
        nb = len(toks) - 1
        if nb <= 0: continue
        
        # Real scores
        hits = sum(1 for b in zip(toks, toks[1:]) if doc_counts[b] >= 2)
        real_scores.append(hits / (nb + _BIGRAM_SHRINK))
        
        # Shuffled scores
        shuffled_toks = list(toks)
        rng.shuffle(shuffled_toks)
        snb = len(shuffled_toks) - 1
        if snb <= 0: continue
        s_hits = 0
        for b in zip(shuffled_toks, shuffled_toks[1:]):
            if b in bigram_set and doc_counts[b] >= 1:
                s_hits += 1
        shuf_scores.append(s_hits / (snb + _BIGRAM_SHRINK))

    if not real_scores or not shuf_scores:
        return bigram_set, 0.0
    
    lo_real = float(np.percentile(np.asarray(real_scores), 5))
    hi_shuf = float(np.percentile(np.asarray(shuf_scores), 95))
    return bigram_set, 0.5 * (lo_real + hi_shuf)

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0: return []

    counts = _token_counts(docs)
    log_probs, unknown_penalty = _get_target_probs(target_sample)
    word_re = re.compile(_TOKEN_PATTERN)

    # Weighted Token-Level Similarity Score
    # Score = sum(log_p(w_i))
    scores = np.zeros(n, dtype=np.float64)
    for i, d in enumerate(docs):
        words = word_re.findall(d.lower())
        score = 0.0
        for w in words:
            score += log_probs.get(w, unknown_penalty)
        scores[i] = score

    # Coherence filtering
    bigram_set, threshold = _calibrate(target_sample, seed)
    
    order_ok = np.ones(n, dtype=bool)
    punct_ok = np.ones(n, dtype=bool)
    
    for i, d in enumerate(docs):
        toks = d.lower().split()
        nb = len(toks) - 1
        if nb <= 0:
            continue
        
        # Bigram hit score
        hits = sum(1 for b in zip(toks, toks[1:]) if b in bigram_set)
        if bigram_set:
            order_ok[i] = (hits / (nb + _BIGRAM_SHRINK)) >= threshold
        
        # Punctuation penalty
        flags = [1 if t in _PUNCT_TOKENS else 0 for t in toks]
        pairs = sum(1 for j in range(len(flags)-1) if flags[j] and flags[j+1])
        punct_ok[i] = (pairs / nb) <= _PUNCT_PAIR_MAX

    # Deduplication
    unique = np.ones(n, dtype=bool)
    seen = set()
    for i, d in enumerate(docs):
        norm_d = _WS.sub(" ", d).strip().lower()
        h = hash(norm_d)
        if h in seen:
            unique[i] = False
        else:
            seen.add(h)

    usable = unique & (counts > 0) & order_ok & punct_ok
    
    if not usable.any():
        usable = unique & (counts > 0)
        if not usable.any():
            usable = unique

    # Final ranking
    # We use the log-probability sum as the primary score
    order = np.argsort(-scores, kind="stable")
    
    selected = []
    total_tokens = 0
    for i in order:
        if usable[i]:
            selected.append(int(i))
            total_tokens += int(counts[i])
            if total_tokens >= T:
                break
    
    if not selected:
        return list(range(min(n, 1000)))

    random.Random(seed).shuffle(selected)
    return selected
