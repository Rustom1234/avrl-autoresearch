"""Curator: dedupe + word-overlap filter.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Drop exact / near-exact duplicates (hash of lowercased, whitespace-collapsed text).
  2. Score each doc by h = fraction of unique words (first 1000 words) that
     appear in the target sample's unique word set, and a = fraction of non-ASCII chars.
  3. Calibrate the h threshold without labels: split the target sample in two halves,
     score each half against the other half's word set, and take the 5th percentile
     of those held-out scores times 0.9.
  4. Keep docs with h >= thr, a < 0.10 and len >= 200 chars; return them in seeded
     random order. If they hold fewer than T tokens the trainer cycles them; rejected
     docs are never used as padding.
"""
import hashlib
import os
import random
import numpy as np

from common import T

MAX_WORDS = 1000      # words per doc used for word-overlap scoring
MAX_NONASCII = 0.10   # max fraction of non-ASCII characters
MIN_CHARS = 200       # min document length in characters
PCTL = 5              # percentile of held-out target scores
SLACK = 0.9           # multiplicative slack on the threshold


def _words(text):
    return text.lower().split()[:MAX_WORDS]


def _word_set(texts):
    s = set()
    for t in texts:
        s.update(_words(t))
    return s


def _hit_frac(words, target_words):
    unique_words = set(words)
    n = len(unique_words)
    if n <= 0:
        return 0.0
    hits = sum(1 for w in unique_words if w in target_words)
    return hits / n


def _nonascii_frac(text):
    if not text:
        return 1.0
    return (len(text) - len(text.encode("ascii", "ignore"))) / len(text)


def _calibrate(target_sample):
    """5th percentile of held-out target word hit rates, times slack."""
    n = len(target_sample)
    if n < 2:
        return 0.0
    half_a = target_sample[0::2]
    half_b = target_sample[1::2]
    set_a = _word_set(half_a)
    set_b = _word_set(half_b)
    scores = [_hit_frac(_words(t), set_a) for t in half_b]
    scores += [_hit_frac(_words(t), set_b) for t in half_a]
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
        # Rough fallback: ~4 chars per token.
        return sum(len(t) for t in texts) // 4


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    target_sample = list(target_sample)
    thr = _calibrate(target_sample)
    target_words = _word_set(target_sample)

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
        if _hit_frac(low_words[:MAX_WORDS], target_words) < thr:
            continue
        kept.append(i)

    if not kept:
        # Degenerate case (e.g. empty target sample): fall back to the deduped pool.
        kept = deduped if deduped else list(range(len(docs)))

    kept_tokens = _count_tokens([docs[i] for i in kept])

    rng.shuffle(kept)
    if kept_tokens >= T:
        # Enough clean data: trainer reads the first T tokens of this order.
        return kept
    # Fewer than T tokens: return only the kept docs; the trainer cycles them.
    return kept
