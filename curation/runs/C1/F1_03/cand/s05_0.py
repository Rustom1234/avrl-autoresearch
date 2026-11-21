"""Curator: dedupe + weighted bigram similarity filter.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Drop exact / near-exact duplicates (hash of lowercased, whitespace-collapsed text).
  2. Score each doc by a weighted sum of bigrams: weights are 1/sqrt(freq) 
     in the target sample.
  3. Calibrate threshold using the 5th percentile of cross-held-out scores.
  4. Keep docs with weighted score >= threshold, low non-ASCII, and min length;
     return in seeded random order.
"""
import hashlib
import os
import random
import math
import numpy as np
from common import T

MAX_WORDS = 4000      # Increased window as requested
MAX_NONASCII = 0.05   # max fraction of non-ASCII characters
MIN_CHARS = 200       # min document length in characters
PCTL = 5              # percentile of held-out target scores
SLACK = 0.9           # multiplicative slack on the threshold

def _words(text):
    return text.lower().split()[:MAX_WORDS]

def _get_weighted_bigrams(texts):
    bigrams = {}
    for t in texts:
        w = _words(t)
        for bg in zip(w, w[1:]):
            bigrams[bg] = bigrams.get(bg, 0) + 1
    # Weight is inverse of sqrt of frequency (TF-IDF-lite approach)
    weights = {bg: 1.0 / math.sqrt(count) for bg, count in bigrams.items()}
    return weights

def _hit_score(words, weights):
    n = len(words) - 1
    if n <= 0:
        return 0.0
    score = 0.0
    hits = 0
    for bg in zip(words, words[1:]):
        if bg in weights:
            score += weights[bg]
            hits += 1
    # Normalize by number of actual hits to compare across different lengths/content
    return score / hits if hits > 0 else 0.0

def _nonascii_frac(text):
    if not text:
        return 1.0
    return (len(text) - len(text.encode("ascii", "ignore"))) / len(text)

def _calibrate(target_sample):
    n = len(target_sample)
    if n < 2:
        return 0.0
    half_a = target_sample[0::2]
    half_b = target_sample[1::2]
    weights_a = _get_weighted_bigrams(half_a)
    weights_b = _get_weighted_bigrams(half_b)
    
    scores = []
    for t in half_b:
        scores.append(_hit_score(_words(t), weights_a))
    for t in half_a:
        scores.append(_hit_score(_words(t), weights_b))
        
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
    weights = _get_weighted_bigrams(target_sample)

    seen = set()
    deduped = []
    kept = []
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        
        low_words = d.lower().split()
        key = hashlib.md5(" ".join(low_words).encode("utf-8", "ignore")).digest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(i)

        if len(d) < MIN_CHARS:
            continue
        if _nonascii_frac(d) >= MAX_NONASCII:
            continue
        
        score = _hit_score(low_words[:MAX_WORDS], weights)
        if score < thr:
            continue
        kept.append(i)

    if not kept:
        kept = deduped if deduped else list(range(len(docs)))

    rng.shuffle(kept)
    return kept
