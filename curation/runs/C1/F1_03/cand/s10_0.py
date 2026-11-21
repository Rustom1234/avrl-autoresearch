"""Curator: dedupe + target-likeness filter with vocabulary-constrained bigrams.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Drop exact / near-exact duplicates (hash of lowercased, whitespace-collapsed text).
  2. Score each doc by h = fraction of its word bigrams (first MAX_WORDS words) that
     are "valid" target bigrams, and a = fraction of non-ASCII chars.
     A bigram is "valid" if both words in the bigram are present in the target_sample.
  3. Calibrate the h threshold without labels: split the target sample in two halves,
     score each half against the other half's "valid" bigram set, and take the
     5th percentile of those held-out scores times 0.9.
  4. Keep docs with h >= thr, a < 0.05 and len >= 200 chars; return them in seeded
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


def _get_valid_bigrams(texts, target_vocab):
    """Returns bigrams where both words are in the target_vocab."""
    valid_bigrams = set()
    for t in texts:
        w = _words(t)
        for i in range(len(w) - 1):
            w1, w2 = w[i], w[i+1]
            if w1 in target_vocab and w2 in target_vocab:
                valid_bigrams.add((w1, w2))
    return valid_bigrams


def _hit_frac(words, valid_bigrams):
    n = len(words) - 1
    if n <= 0:
        return 0.0
    hits = 0
    # We look for bigrams formed by the document that exist in the target-valid set
    for i in range(len(words) - 1):
        bg = (words[i], words[i+1])
        if bg in valid_bigrams:
            hits += 1
    return hits / n


def _nonascii_frac(text):
    if not text:
        return 1.0
    return (len(text) - len(text.encode("ascii", "ignore"))) / len(text)


def _calibrate(target_sample):
    """5th percentile of held-out target bigram hit rates, times slack."""
    n = len(target_sample)
    if n < 2:
        return 0.0
    
    # Build vocabulary of words actually present in target sample
    target_vocab = set()
    for t in target_sample:
        target_vocab.update(_words(t))
        
    half_a = target_sample[0::2]
    half_b = target_sample[1::2]
    
    valid_bg_a = _get_valid_bigrams(half_a, target_vocab)
    valid_bg_b = _get_valid_bigrams(half_b, target_vocab)
    
    scores = []
    for t in half_b:
        scores.append(_hit_frac(_words(t), valid_bg_a))
    for t in half_a:
        scores.append(_hit_frac(_words(t), valid_bg_b))
        
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
    
    # Build target vocabulary first for calibration and filtering
    target_vocab = set()
    for t in target_sample:
        target_vocab.update(_words(t))
        
    thr = _calibrate(target_sample)
    # valid_bigrams for selection
    valid_bigrams = _get_valid_bigrams(target_sample, target_vocab)

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
        if _hit_frac(low_words[:MAX_WORDS], valid_bigrams) < thr:
            continue
            
        kept.append(i)

    if not kept:
        kept = deduped_indices if deduped_indices else list(range(len(docs)))

    # We don't need to check tokens here because trainer cycles if under T,
    # but we keep the logic for consistency with original flow.
    
    rng.shuffle(kept)
    return kept
