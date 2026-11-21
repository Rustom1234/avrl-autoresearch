"""Curator: dedupe + weighted target-likeness filter.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Drop exact / near-exact duplicates (hash of lowercased, whitespace-collapsed text).
  2. Score each doc by a weighted bigram overlap: bigrams occurring in more 
     target documents get higher weights.
  3. Calibrate the threshold without labels: split the target sample in two halves,
     score each half against the other half's weighted set, and take the 5th 
     percentile of those held-out scores times 0.9.
  4. Keep docs with weighted score >= thr, a < 0.05 and len >= 200 chars; 
     return them in seeded random order.
"""
import hashlib
import os
import random
import numpy as np
from collections import Counter

from common import T

MAX_WORDS = 2000      # words per doc used for bigram scoring
MAX_NONASCII = 0.05   # max fraction of non-ASCII characters
MIN_CHARS = 200       # min document length in characters
PCTL = 5              # percentile of held-out target scores
SLACK = 0.9           # multiplicative slack on the threshold

def _words(text):
    return text.lower().split()[:MAX_WORDS]

def _get_weighted_bigrams(texts):
    """Returns a dictionary mapping bigram tuple -> weight (count of docs it appears in)."""
    counts = Counter()
    for t in texts:
        w = _words(t)
        bgs = set(zip(w, w[1:]))
        for bg in bgs:
            counts[bg] += 1
    return counts

def _weighted_hit_frac(words, weighted_bigrams):
    n = len(words) - 1
    if n <= 0:
        return 0.0
    score = 0.0
    for bg in zip(words, words[1:]):
        if bg in weighted_bigrams:
            # weight is the count of documents it appeared in
            score += weighted_bigrams[bg]
    # Normalize by n and by the average weight to keep scale somewhat comparable
    # to the original hit_frac logic, but we'll let calibration handle the scale.
    return score / n

def _nonascii_frac(text):
    if not text:
        return 1.0
    return (len(text) - len(text.encode("ascii", "ignore"))) / len(text)

def _calibrate(target_sample):
    """5th percentile of held-out weighted target scores, times slack."""
    n = len(target_sample)
    if n < 2:
        return 0.0
    half_a = target_sample[0::2]
    half_b = target_sample[1::2]
    
    weights_a = _get_weighted_bigrams(half_a)
    weights_b = _get_weighted_bigrams(half_b)
    
    scores = []
    for t in half_b:
        scores.append(_weighted_hit_frac(_words(t), weights_a))
    for t in half_a:
        scores.append(_weighted_hit_frac(_words(t), weights_b))
        
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
    thr = _calibrate(target_sample)
    weighted_bigrams = _get_weighted_bigrams(target_sample)

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

        # Filtering
        if len(d) < MIN_CHARS:
            continue
        if _nonascii_frac(d) >= MAX_NONASCII:
            continue
        
        score = _weighted_hit_frac(low_words[:MAX_WORDS], weighted_bigrams)
        if score >= thr:
            kept.append(i)

    if not kept:
        kept = deduped_indices if deduped_indices else list(range(len(docs)))

    # Ensure we don't do redundant heavy work if we already have plenty
    # But we need to know the count for the return logic.
    # Since the trainer cycles if < T, we just return the list.
    
    rng.shuffle(kept)
    return kept
