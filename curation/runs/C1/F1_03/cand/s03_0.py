"""Curator: dedupe + weighted bigram-likelihood filter.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Drop exact / near-exact duplicates (hash of lowercased, whitespace-collapsed text).
  2. Score each doc by a weighted hit rate of bigrams:
     weight(bg) = count(bg in target_sample) / total_bigrams_in_target_sample
     score(doc) = sum(weight(bg) for bg in doc_bigrams if bg in target_sample)
  3. Calibrate the threshold using the target sample's own internal self-scoring.
  4. Keep docs with score >= thr, low non-ASCII, and minimum length.
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


def _get_bigrams(texts):
    """Returns a list of bigrams and a counter of their frequencies."""
    all_bigrams = []
    counts = collections.defaultdict(int)
    for t in texts:
        w = _words(t)
        bg_list = list(zip(w, w[1:]))
        all_bigrams.extend(bg_list)
        for bg in bg_list:
            counts[bg] += 1
    return all_bigrams, counts


def _weighted_hit_frac(words, target_counts, total_target_bgrams):
    n_words = len(words)
    if n_words <= 1:
        return 0.0
    
    score = 0.0
    doc_bg_count = 0
    for bg in zip(words, words[1:]):
        if bg in target_counts:
            # weight = count_in_target / total_target_bigrams
            score += target_counts[bg]
            doc_bg_count += 1
            
    if doc_bg_count == 0:
        return 0.0
    # Return the average weight of the bigrams that matched
    return score / doc_bg_count


def _nonascii_frac(text):
    if not text:
        return 1.0
    return (len(text) - len(text.encode("ascii", "ignore"))) / len(text)


def _calibrate(target_sample):
    """Calibrate threshold using internal weighted scores of the target sample."""
    if len(target_sample) < 2:
        return 0.0
    
    all_bg_list, counts = _get_bigrams(target_sample)
    total_bg_count = len(all_bg_list)
    if total_bg_count == 0:
        return 0.0
    
    half_a = target_sample[0::2]
    half_b = target_sample[1::2]
    
    scores = []
    for t in half_b:
        scores.append(_weighted_hit_frac(_words(t), counts, total_bg_count))
    for t in half_a:
        scores.append(_weighted_hit_frac(_words(t), counts, total_bg_count))
        
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
            batch = texts[i:i + B]
            total += sum(len(x) for x in enc.encode_ordinary_batch(batch))
        return total
    except Exception:
        return sum(len(t) for t in texts) // 4


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    target_sample = list(target_sample)
    thr = _calibrate(target_sample)
    
    # Prepare weighted counts for target sample
    _, counts = _get_bigrams(target_sample)
    total_bg_count = sum(counts.values())

    seen = set()
    deduped_indices = []
    kept_indices = []
    
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
        
        score = _weighted_hit_frac(low_words[:MAX_WORDS], counts, total_bg_count)
        if score < thr:
            continue
            
        kept_indices.append(i)

    if not kept_indices:
        kept_indices = deduped_indices if deduped_indices else list(range(len(docs)))

    # Shuffle final selection
    rng.shuffle(kept_indices)
    
    # Check if we have enough tokens (optional, but helps understand flow)
    # The trainer cycles if kept_indices < T tokens.
    # We return the indices in the shuffled order.
    return kept_indices
