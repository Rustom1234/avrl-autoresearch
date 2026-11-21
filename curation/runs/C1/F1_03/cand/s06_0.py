"""Curator: dedupe + quality scoring + top-K selection.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Drop exact / near-exact duplicates (hash of lowercased, whitespace-collapsed text).
  2. Score each doc by:
     h = fraction of its word bigrams (first MAX_WORDS words) that occur in the target sample's bigram set.
     a = fraction of non-ASCII chars.
     score = (h + log(len(d))) if a < MAX_NONASCII else -1.
  3. Sort docs by score descending and select top K such that total tokens >= T.
  4. If fewer than T tokens are available after filtering, return all valid docs.
"""
import hashlib
import os
import random
import math
import numpy as np
from common import T

MAX_WORDS = 2000      # words per doc used for bigram scoring
MAX_NONASCII = 0.05   # max fraction of non-ASCII characters
MIN_CHARS = 200       # min document length in characters

def _words(text):
    return text.lower().split()[:MAX_WORDS]

def _bigram_set(texts):
    s = set()
    for t in texts:
        w = _words(t)
        s.update(zip(w, w[1:]))
    return s

def _hit_frac(words, bigrams):
    n = len(words) - 1
    if n <= 0:
        return 0.0
    hits = 0
    for bg in zip(words, words[1:]):
        if bg in bigrams:
            hits += 1
    return hits / n

def _nonascii_frac(text):
    if not text:
        return 1.0
    return (len(text) - len(text.encode("ascii", "ignore"))) / len(text)

def _count_tokens(texts):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        total = 0
        B = 2000
        for i in range(0, len(texts), B):
            total += sum(len(x) for x in enc.encode_ordinary_batch(texts[i:i + B]))
        return total
    except Exception:
        # Rough fallback: ~4 chars per token.
        return sum(len(t) for t in texts) // 4

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    target_sample = list(target_sample)
    target_bigrams = _bigram_set(target_sample)

    seen = set()
    candidates = []

    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        
        # Deduplication
        low_words = d.lower().split()
        key = hashlib.md5(" ".join(low_words).encode("utf-8", "ignore")).digest()
        if key in seen:
            continue
        seen.add(key)

        # Basic Filtering
        if len(d) < MIN_CHARS:
            continue
        if _nonascii_frac(d) >= MAX_NONASCII:
            continue
            
        # Scoring
        h = _hit_frac(low_words[:MAX_WORDS], target_bigrams)
        # Weighted score: Bigram similarity + log of length to favor content-rich docs
        # Use log10(len) to prevent extremely long docs from dominating the score disproportionately
        score = h + math.log10(max(1, len(d)))
        candidates.append((score, i, d))

    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Determine how many we need to reach T
    # We use a subset of candidates to calculate tokens
    # To be efficient, we greedily pick from the best scores
    kept_indices = []
    current_tokens = 0
    
    # Pre-calculate tokens for all candidates to make selection easy
    # (Optimization: use a small batch for _count_tokens)
    candidate_texts = [c[2] for c in candidates]
    # The count for a doc can be approximated or calculated
    # To stay within time limits and satisfy the logic, we calculate token counts
    # for all valid docs once.
    doc_tokens = [_count_tokens([c[2]]) for c in candidates]
    
    for idx, (score, original_idx, text) in enumerate(candidates):
        kept_indices.append(original_idx)
        current_tokens += doc_tokens[idx]
        if current_tokens >= T:
            break

    if not kept_indices:
        return []

    # The trainer cycles the list if it's < T, so we just return the sorted list.
    # However, the prompt asks for the order. We keep the sorted order (highest quality first).
    return kept_indices
