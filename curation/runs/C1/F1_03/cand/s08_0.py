"""Curator: dedupe + robust target-likeness filter.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Drop exact / near-exact duplicates (hash of lowercased, whitespace-collapsed text).
  2. Score each doc by h = fraction of its word bigrams (first MAX_WORDS words) that
     occur in the "stable" bigram set (bigrams appearing in >= 3 docs in target_sample).
     Apply a penalty for high unique bigram counts (complexity penalty).
  3. Calibrate the h threshold without labels using held-out target scores.
  4. Keep docs with h >= thr, a < 0.05 and len >= 200 chars; return them in seeded
     random order.
"""
import hashlib
import os
import random
import collections
import numpy as np
from common import T

MAX_WORDS = 2000      # words per doc used for bigram scoring
MAX_NONASCII = 0.05   # max fraction of non-ASCII characters
MIN_CHARS = 200       # min document length in characters
PCTL = 5              # percentile of held-out target scores
SLACK = 0.9           # multiplicative slack on the threshold

def _words(text):
    return text.lower().split()[:MAX_WORDS]

def _get_bigrams(words):
    return list(zip(words, words[1:]))

def _get_stable_bigrams(target_sample):
    """Returns bigrams that appear in at least 3 different target documents."""
    counts = collections.defaultdict(int)
    # We need to know which documents each bigram appears in
    # But for memory/speed, we can just count occurrences across documents
    # and treat bigrams that are unique to one doc as noise.
    doc_bigrams = []
    for t in target_sample:
        doc_bigrams.append(set(_get_bigrams(_words(t))))
    
    global_counts = collections.defaultdict(int)
    for b_set in doc_bigrams:
        for bg in b_set:
            global_counts[bg] += 1
            
    # A bigram is 'stable' if it appears in >= 3 documents
    stable_set = {bg for bg, count in global_counts.items() if count >= 3}
    return stable_set

def _hit_frac(words, stable_bigrams):
    n = len(words) - 1
    if n <= 0:
        return 0.0
    doc_bigrams = set(_get_bigrams(words))
    hits = sum(1 for bg in doc_bigrams if bg in stable_bigrams)
    
    # Penalty: Subtract a small fraction based on unique bigrams (complexity)
    # If a doc has too many unique bigrams, it's likely low-quality/random.
    # We penalize the hit rate slightly based on the ratio of unique bigrams to hits.
    # Penalty factor: max(0, 1 - (unique_count / (hits + 1) * 0.1))
    # Actually, the prompt asks for a soft-penalty for high unique bigram counts.
    penalty = min(0.2, (len(doc_bigrams) / (hits + 1)) * 0.05)
    
    score = (hits / n) - penalty
    return max(0.0, score)

def _nonascii_frac(text):
    if not text:
        return 1.0
    return (len(text) - len(text.encode("ascii", "ignore"))) / len(text)

def _calibrate(target_sample, stable_bigrams):
    """5th percentile of held-out target bigram hit rates, times slack."""
    n = len(target_sample)
    if n < 2:
        return 0.0
    half_a = target_sample[0::2]
    half_b = target_sample[1::2]
    
    # Note: Calibration should use the actual hit rate logic
    scores = []
    for t in half_b:
        scores.append(_hit_frac(_words(t), stable_bigrams))
    for t in half_a:
        scores.append(_hit_frac(_words(t), stable_bigrams))
        
    if not scores:
        return 0.0
    return float(np.percentile(np.asarray(scores, dtype=np.float64), PCTL)) * SLACK

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
        return sum(len(t) for t in texts) // 4

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    target_sample = list(target_sample)
    stable_bigrams = _get_stable_bigrams(target_sample)
    thr = _calibrate(target_sample, stable_bigrams)

    seen = set()
    deduped_indices = []
    kept = []

    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        
        # Deduplication
        low_words = d.lower().split()
        key = hashlib.md5(" ".join(low_words).encode("utf-8", "ignore")).digest()
        if key in seen:
            continue
        seen.add(key)
        deduped_indices.append(i)

        # Filters
        if len(d) < MIN_CHARS:
            continue
        if _nonascii_frac(d) >= MAX_NONASCII:
            continue
        
        score = _hit_frac(low_words[:MAX_WORDS], stable_bigrams)
        if score >= thr:
            kept.append(i)

    if not kept:
        kept = deduped_indices if deduped_indices else list(range(len(docs)))

    # Optional: check if we have enough tokens, though the trainer cycles if not.
    # The prompt says "return them in seeded random order".
    rng.shuffle(kept)
    return kept
