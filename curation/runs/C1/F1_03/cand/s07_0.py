"""Curator: dedupe + smoothed bigram likelihood + novelty penalty.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Drop exact / near-exact duplicates (hash of lowercased, whitespace-collapsed text).
  2. Score each doc by h = hits / ((doc_bg_count + target_bg_count) / 2).
  3. Penalize docs with high 'novelty' (proportion of bigrams not in target_sample).
  4. Calibrate threshold using the same smoothing/novelty logic.
  5. Keep docs with h >= thr, a < 0.05 and len >= 200 chars; return them in seeded
     random order.
"""
import hashlib
import os
import random
import numpy as np
from common import T

MAX_WORDS = 2000      # words per doc used for bigram scoring
MAX_NONASCII = 0.05   # max fraction of non-ASCII characters
MIN_CHARS = 200       # min document length in characters
PCTL = 5              # percentile of held-out target scores
SLACK = 0.9           # multiplicative slack on the threshold

def _words(text):
    return text.lower().split()[:MAX_WORDS]

def _bigram_set(texts):
    s = set()
    for t in texts:
        w = _words(t)
        s.update(zip(w, w[1:]))
    return s

def _score_doc(words, target_bg_set, target_bg_count):
    bg_list = list(zip(words, words[1:]))
    doc_bg_count = len(bg_list)
    if doc_bg_count == 0:
        return 0.0, 0.0
    
    hits = sum(1 for bg in bg_list if bg in target_bg_set)
    
    # Smoothed average: hits / ((doc_bg_count + target_bg_count) / 2)
    # Note: The denominator could theoretically be 0 if target_bg_count is 0.
    denom = (doc_bg_count + target_bg_count) / 2.0
    h = hits / denom if denom > 0 else 0.0
    
    # Novelty: fraction of doc bigrams not in target
    novelty = (doc_bg_count - hits) / doc_bg_count
    # Penalize novelty: reduce score by novelty fraction
    final_score = h * (1.0 - novelty)
    
    return final_score, novelty

def _calibrate(target_sample):
    n = len(target_sample)
    if n < 2:
        return 0.0
    half_a = target_sample[0::2]
    half_b = target_sample[1::2]
    set_a = _bigram_set(half_a)
    set_b = _bigram_set(half_b)
    
    # Need counts to use the new smoothing logic
    def get_count(texts):
        cnt = 0
        for t in texts:
            w = _words(t)
            cnt += len(list(zip(w, w[1:])))
        return cnt

    count_a = get_count(half_a)
    count_b = get_count(half_b)

    scores = []
    # Score half_b against set_a
    for t in half_b:
        w = _words(t)
        score, _ = _score_doc(w, set_a, count_a)
        scores.append(score)
    # Score half_a against set_b
    for t in half_a:
        w = _words(t)
        score, _ = _score_doc(w, set_b, count_b)
        scores.append(score)
        
    if not scores:
        return 0.0
    return float(np.percentile(np.asarray(scores, dtype=np.float64), PCTL)) * SLACK

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
        return sum(len(t) for t in texts) // 4

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    target_sample = list(target_sample)
    thr = _calibrate(target_sample)
    target_bigrams = _bigram_set(target_sample)
    
    # Count unique bigrams in target_sample for smoothing
    target_bg_count = 0
    for t in target_sample:
        w = _words(t)
        target_bg_count += len(list(zip(w, w[1:])))

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
            
        score, _ = _score_doc(low_words[:MAX_WORDS], target_bigrams, target_bg_count)
        if score < thr:
            continue
        kept.append(i)

    if not kept:
        kept = deduped if deduped else list(range(len(docs)))

    kept_tokens = _count_tokens([docs[i] for i in kept])

    rng.shuffle(kept)
    return kept
