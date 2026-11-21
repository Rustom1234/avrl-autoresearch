"""Curator: rank pool documents by weighted bigram uniqueness and sentence length variance.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import re
from collections import Counter

import numpy as np

from common import T

_TOKEN_PATTERN = r"[a-z0-9']+"
_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"[.!?]+")

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

def _sentence_stats(target_sample):
    """Return mean and std of sentence lengths in target sample."""
    lens = []
    for t in target_sample:
        sents = _SENT_SPLIT.split(t)
        for s in sents:
            words = _WS.split(s)
            if len(words) > 0:
                lens.append(len(words))
    if not lens:
        return 0.0, 0.0
    return np.mean(lens), np.std(lens)

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0:
        return []

    rng = random.Random(seed)
    counts = _token_counts(docs)
    
    # Target Analysis
    # 1. Weighted Bigrams
    target_bigrams = Counter()
    target_words = set()
    for t in target_sample:
        tokens = _WS.split(t.lower())
        target_words.update(tokens)
        for i in range(len(tokens) - 1):
            target_bigrams[(tokens[i], tokens[i+1])] += 1
    
    # 2. Sentence Length Stats
    mean_len, std_len = _sentence_stats(target_sample)

    # Scoring
    scores = np.zeros(n, dtype=np.float64)
    bigram_scores = np.zeros(n, dtype=np.float64)
    punct_frac = np.zeros(n, dtype=np.float64)
    
    # Use a simple approach for vocabulary overlap as a baseline
    target_vocab_set = set(target_words)
    
    for i, d in enumerate(docs):
        tokens = _WS.split(d.lower())
        if not tokens:
            continue
            
        # Vocabulary overlap
        doc_words = set(tokens)
        overlap = len(doc_words.intersection(target_vocab_set))
        scores[i] = overlap / (len(doc_words) + 1.0)
        
        # Weighted Bigrams (Temporal Coherence)
        # Weight = 1 / (freq + 1)
        b_hits = 0.0
        for j in range(len(tokens) - 1):
            bg = (tokens[j], tokens[j+1])
            if bg in target_bigrams:
                freq = target_bigrams[bg]
                b_hits += (1.0 / (freq + 1.0))
        bigram_scores[i] = b_hits / (len(tokens) - 1 + _BIGRAM_SHRINK)
        
        # Punctuation check
        is_punct = _PUNCT_TOKENS.__contains__
        flags = [is_punct(t) for t in tokens]
        if len(flags) > 1:
            pairs = sum(1 for k in range(len(flags)-1) if flags[k] and flags[k+1])
            punct_frac[i] = pairs / (len(flags) - 1)
        else:
            punct_frac[i] = 0.0

    # Sentence Variance Filter
    def get_doc_variance(text):
        sents = _SENT_SPLIT.split(text)
        lens = [len(_WS.split(s)) for s in sents if _WS.split(s)]
        if len(lens) < 2: return 0.0
        return np.var(lens)

    # We use a heuristic: penalize if doc variance is > 2 std devs from target mean
    # Since we don't have exact target variance, we use the target std_len as a proxy 
    # for the 'allowed' variance. 
    # Actually, the prompt asks to penalize if variance deviates from target distribution.
    # Let's calculate variance of the target sentence lengths:
    target_lens = []
    for t in target_sample:
        sents = _SENT_SPLIT.split(t)
        for s in sents:
            w = _WS.split(s)
            if w: target_lens.append(len(w))
    target_var = np.var(target_lens) if target_lens else 0.0
    target_std = np.std(target_lens) if target_lens else 0.0
    
    # A document is "ok" if its variance is within 2 standard deviations of the target variance
    # However, variance is a measure of spread. If target_std is small, doc variance must be small.
    # Let's stick to the rule: penalize if doc variance > (target_var + 2 * target_std)
    # and treat variance as the standard deviation of lengths for simpler scaling.
    
    valid_mask = np.ones(n, dtype=bool)
    
    # Punctuation filter
    valid_mask &= (punct_frac <= _PUNCT_PAIR_MAX)
    
    # Sentence length variance filter
    for i, d in enumerate(docs):
        # Skip very short docs that can't have variance
        if len(_SENT_SPLIT.split(d)) < 2:
            continue
        d_lens = [len(_WS.split(s)) for s in _SENT_SPLIT.split(d) if _WS.split(s)]
        d_var = np.var(d_lens) if d_lens else 0.0
        # Penalty if variance is too high compared to target
        if d_var > (target_var + 2 * target_std) and target_std > 0:
            valid_mask[i] = False

    # Rank by (Overlap * Weighted Bigram)
    final_scores = scores * bigram_scores
    
    # Deduplication
    unique_indices = []
    seen_hashes = set()
    for i in range(n):
        if valid_mask[i] and counts[i] > 0:
            h = hash(_WS.sub(" ", docs[i]).strip().lower())
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_indices.append(i)
    
    # Sort indices by score
    unique_indices = sorted(unique_indices, key=lambda x: final_scores[x], reverse=True)
    
    # Selection
    selected = []
    current_tokens = 0
    for idx in unique_indices:
        selected.append(idx)
        current_tokens += counts[idx]
        if current_tokens >= T:
            break
            
    rng.shuffle(selected)
    return selected
