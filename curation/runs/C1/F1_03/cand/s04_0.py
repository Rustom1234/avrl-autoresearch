"""Curator: weighted Jaccard similarity and extended token window.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Drop exact / near-exact duplicates (hash of lowercased, whitespace-collapsed text).
  2. Score each doc by j = intersection_size / (doc_bigrams_count + target_bigrams_count - intersection_size)
     where bigrams are from the first 3000 words.
  3. Calibrate the j threshold without labels: split the target sample in two halves,
     score each half against the other half's bigram set, and take the 5th percentile
     of those held-out scores times 0.9.
  4. Keep docs with j >= thr, a < 0.05 and len >= 200 chars; return them in seeded
     random order.
"""
import hashlib
import os
import random
import numpy as np
from common import T

MAX_WORDS = 3000      # Extended window for scoring
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


def _jaccard_score(doc_words, target_bigrams):
    doc_bg_list = list(zip(doc_words, doc_words[1:]))
    doc_bg_set = set(doc_bg_list)
    
    intersection = doc_bg_set.intersection(target_bigrams)
    union = doc_bg_set.union(target_bigrams)
    
    if not union:
        return 0.0
    return len(intersection) / len(union)


def _nonascii_frac(text):
    if not text:
        return 1.0
    return (len(text) - len(text.encode("ascii", "ignore"))) / len(text)


def _calibrate(target_sample):
    """5th percentile of held-out target Jaccard scores, times slack."""
    n = len(target_sample)
    if n < 2:
        return 0.0
    half_a = target_sample[0::2]
    half_b = target_sample[1::2]
    set_a = _bigram_set(half_a)
    set_b = _bigram_set(half_b)
    
    scores = []
    for t in half_b:
        w = _words(t)
        scores.append(_jaccard_score(w, set_a))
    for t in half_a:
        w = _words(t)
        scores.append(_jaccard_score(w, set_b))
        
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
    target_bigrams = _bigram_set(target_sample)

    seen = set()
    deduped = []
    kept = []
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        
        # Deduplication logic
        low_words = d.lower().split()
        key = hashlib.md5(" ".join(low_words).encode("utf-8", "ignore")).digest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(i)

        # Filtering logic
        if len(d) < MIN_CHARS:
            continue
        if _nonascii_frac(d) >= MAX_NONASCII:
            continue
        
        w = _words(d)
        if _jaccard_score(w, target_bigrams) < thr:
            continue
            
        kept.append(i)

    if not kept:
        kept = deduped if deduped else list(range(len(docs)))

    rng.shuffle(kept)
    # The trainer cycles the list if it is shorter than T tokens.
    return kept
